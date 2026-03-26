#!/usr/bin/env python3
"""
FastNAS prune backbone only for UniV2X infra-side track model.

Key behavior:
1) Build the full model from MMCV config/checkpoint.
2) Remove freeze constraints from config/model (backbone/neck/bn/frozen_stages).
3) Extract `model.model_ego_agent.img_backbone` and prune ONLY this backbone.
4) Trace/search only sees the backbone wrapper.
5) Subnet scoring evaluates the FULL model by re-inserting candidate backbone.
6) Score objective is global detection mAP (`pts_bbox_NuScenes/mAP`).
7) Support subset split modes for calib/score datasets:
   - full (complete)
   - head (fixed scene head)
   - skip_warmup (skip first 4 frames, then fixed head)

Fixes applied (vs original script):
  Fix 1: --min-params-ratio post-pruning check to abort if too aggressive.
  Fix 2: Pre/post export diagnostic evaluations (DIAG-A/B/C) to pinpoint accuracy loss.
  Fix 3: Deep copy backbone to prevent reference aliasing corruption (CRITICAL).
  Fix 4: --min-channel-ratio and --lock-stage-outputs to limit pruning aggressiveness.
  Fix 5: Post-export BN stats / weight sanity validation.

Example (scene 10-frame subset, with safety guards):
    cd /home/lixingfeng/UniAD_exmaine/UniV2X
    python ./univ2x_purned/univ2x-inf-track-prune-backbone_byclaude.py \
      --gpu-id 0 \
      --params-percent 90% \
      --min-params-ratio 0.8 \
      --min-channel-ratio 0.5 \
      --lock-stage-outputs \
      --calib-split train \
      --calib-subset-mode skip_warmup \
      --calib-frames-per-scene 10 \
      --score-split val \
      --score-subset-mode skip_warmup \
      --score-frames-per-scene 10 \
      --eval-mods det
"""

import os
import sys
import copy
import time
import math
import atexit
import signal
import logging
import traceback
import argparse
import warnings
import os.path as osp
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn

import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, scatter
from mmcv.runner import load_checkpoint

from mmdet.apis import set_random_seed
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import modelopt.torch.prune as mtp
import modelopt.torch.opt as mto

# Ensure repo root on PYTHONPATH.
THIS_DIR = osp.dirname(osp.abspath(__file__))
REPO_ROOT = osp.abspath(osp.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.legend").setLevel(logging.ERROR)

RUN_ROOT = (
    "/home/lixingfeng/UniAD_exmaine/UniV2X/"
    "univ2x_purned/modelopt/univ2x_inf_track_prune_backbone_byclaude"
)
DEFAULT_OUT_DIR = osp.join(RUN_ROOT, "out")
DEFAULT_LOG_DIR = osp.join(RUN_ROOT, "logs")


class _Tee:
    """Python-level tee: mirror stdout/stderr to terminal and log file."""

    def __init__(self, stream, log_fh):
        self._stream = stream
        self._log_fh = log_fh

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        if data:
            # Keep tqdm/progress updates visible in logs.
            log_data = data.replace("\r\n", "\n").replace("\r", "\n")
            self._log_fh.write(log_data)
            self._log_fh.flush()

    def flush(self):
        self._stream.flush()
        self._log_fh.flush()

    def __getattr__(self, attr):
        return getattr(self._stream, attr)


def setup_logging(log_root: str, run_ts: str) -> str:
    run_log_dir = osp.join(log_root, run_ts)
    os.makedirs(run_log_dir, exist_ok=True)
    log_path = osp.join(run_log_dir, "run.log")
    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    def _flush_and_close():
        try:
            log_fh.flush()
            log_fh.close()
        except Exception:
            pass

    atexit.register(_flush_and_close)
    orig_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(sig, frame):
        print("\n[INFO] received SIGINT, flushing log ...", file=sys.__stderr__)
        _flush_and_close()
        signal.signal(signal.SIGINT, orig_sigint)
        signal.raise_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, _sigint_handler)
    return log_path


class ScoreFuncTracker:
    """Track score_func calls for debugging/search review."""

    def __init__(self):
        self.call_history: List[Dict[str, Any]] = []

    def record(self, call_id: int, score: float, elapsed_sec: float):
        stack = traceback.extract_stack()
        caller_frames = []
        for frame in stack:
            if "modelopt" in frame.filename or "score_func" in frame.name:
                caller_frames.append(f"  {frame.filename}:{frame.lineno} in {frame.name}")
        caller_info = "\n".join(caller_frames[-5:]) if caller_frames else "(unknown caller)"
        self.call_history.append(
            {
                "call_id": int(call_id),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "score_map": float(score),
                "elapsed_sec": float(elapsed_sec),
                "caller_info": caller_info,
            }
        )

    def print_report(self) -> None:
        print("\n" + "=" * 72)
        print("[TRACKER] score_func report")
        print("=" * 72)
        print(f"[TRACKER] total calls: {len(self.call_history)}")
        if not self.call_history:
            print("[TRACKER][WARN] score_func was never called.")
            print("=" * 72 + "\n")
            return

        scores = [h["score_map"] for h in self.call_history]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_item = self.call_history[best_idx]
        print(f"[TRACKER] score range: [{min(scores):.6f}, {max(scores):.6f}]")
        print(f"[TRACKER] score mean: {sum(scores) / len(scores):.6f}")
        print(
            f"[TRACKER] best call: #{best_item['call_id']:03d}, "
            f"score={best_item['score_map']:.6f}"
        )
        print(f"[TRACKER] first call: {self.call_history[0]['timestamp']}")
        print(f"[TRACKER] last call: {self.call_history[-1]['timestamp']}")
        for h in self.call_history:
            print(
                f"  #{h['call_id']:03d} {h['timestamp']} "
                f"mAP={h['score_map']:.6f} elapsed={h['elapsed_sec']:.2f}s"
            )
        print("=" * 72 + "\n")

    def get_summary(self) -> Dict[str, Any]:
        if not self.call_history:
            return {"num_calls": 0}
        scores = [h["score_map"] for h in self.call_history]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_item = self.call_history[best_idx]
        return {
            "num_calls": len(self.call_history),
            "score_min": float(min(scores)),
            "score_max": float(max(scores)),
            "score_mean": float(sum(scores) / len(scores)),
            "best_call_id": int(best_item["call_id"]),
            "best_score": float(best_item["score_map"]),
        }


def patch_torchprofile_dtype_setter() -> None:
    """Patch torchprofile None dtype issue for some traced ops."""
    try:
        from torchprofile.utils.ir.variable import Variable as _TorchProfileVariable
    except Exception as exc:
        print(f"[WARN] torchprofile patch skipped: {exc}")
        return

    @_TorchProfileVariable.dtype.setter  # type: ignore[attr-defined]
    def _safe_dtype_setter(self, dtype):
        self._dtype = dtype.lower() if dtype is not None else "unknown"

    _TorchProfileVariable.dtype = _safe_dtype_setter
    print("[PATCH] torchprofile Variable.dtype setter patched")


def fixed_head_indices(dataset, frames_per_scene=10, sampling_mode="skip_warmup"):
    """Scene-wise contiguous head sampling."""
    scene_to_indices = defaultdict(list)
    for i, info in enumerate(dataset.data_infos):
        scene_to_indices[info["scene_token"]].append(i)

    scenes = sorted(scene_to_indices.keys())
    selected = []
    if sampling_mode == "head":
        start_idx = 0
    elif sampling_mode == "skip_warmup":
        start_idx = 4
    else:
        raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")

    for s in scenes:
        idxs = scene_to_indices[s]
        selected.extend(idxs[start_idx:start_idx + frames_per_scene])
    selected = sorted(selected)

    scene_counts = defaultdict(int)
    for i in selected:
        scene_counts[dataset.data_infos[i]["scene_token"]] += 1

    print(
        f"[INFO] subset sampling mode={sampling_mode}, start_idx={start_idx}, "
        f"covered scenes={len(scene_counts)}/{len(scenes)}, total={len(selected)}"
    )
    if scene_counts:
        vals = list(scene_counts.values())
        print(
            f"       per-scene frames: min={min(vals)} max={max(vals)} "
            f"avg={np.mean(vals):.1f}"
        )
    return selected


class SubsetDataset(torch.utils.data.Dataset):
    """Subset wrapper preserving dataset evaluate/format behavior."""

    _PASSTHROUGH = [
        "CLASSES",
        "eval_mod",
        "eval_detection_configs",
        "eval_version",
        "modality",
        "overlap_test",
        "version",
        "data_root",
        "nusc",
        "nusc_maps",
        "ErrNameMapping",
        "with_velocity",
        "split_datas_file",
        "class_range",
        "tmp_dataset_type",
        "planning_steps",
    ]

    def __init__(self, dataset, indices):
        self._dataset = dataset
        self._indices = list(indices)
        self.data_infos = [dataset.data_infos[i] for i in self._indices]
        for attr in self._PASSTHROUGH:
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))
        if hasattr(dataset, "flag"):
            self.flag = dataset.flag[np.array(self._indices)]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self._dataset[self._indices[idx]]

    def _swap_and_call(self, method_name, *args, **kwargs):
        orig = self._dataset.data_infos
        self._dataset.data_infos = self.data_infos
        try:
            return getattr(self._dataset, method_name)(*args, **kwargs)
        finally:
            self._dataset.data_infos = orig

    def evaluate(self, results, **kwargs):
        return self._swap_and_call("evaluate", results, **kwargs)

    def format_results(self, results, **kwargs):
        return self._swap_and_call("format_results", results, **kwargs)

    def format_results_det(self, results, **kwargs):
        return self._swap_and_call("format_results_det", results, **kwargs)


def assert_output_token_alignment(eval_dataset, outputs):
    """Optional hard check for output/data_info token alignment."""
    ds_tokens = [info["token"] for info in eval_dataset.data_infos]
    if len(outputs) != len(ds_tokens):
        raise AssertionError(
            f"output count ({len(outputs)}) != dataset count ({len(ds_tokens)})"
        )

    out_tokens = []
    for out in outputs:
        token = None
        if isinstance(out, dict):
            token = out.get("token", None)
        elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            token = out[0].get("token", None)
        out_tokens.append(token)

    if all(t is None for t in out_tokens):
        print("[CHECK][WARN] output token not found; skip token alignment check.")
        return

    missing = [i for i, t in enumerate(out_tokens) if t is None]
    if missing:
        raise AssertionError(f"missing token in outputs, sample idx: {missing[:10]}")

    mismatch = []
    for i, (pred_t, ds_t) in enumerate(zip(out_tokens, ds_tokens)):
        if pred_t != ds_t:
            mismatch.append((i, pred_t, ds_t))
            if len(mismatch) >= 10:
                break
    if mismatch:
        msg = "; ".join([f"idx={i}, pred={p}, ds={d}" for i, p, d in mismatch])
        raise AssertionError(f"output token order mismatch: {msg}")
    print(f"[CHECK] output token alignment passed: {len(ds_tokens)} samples.")


def patch_nuscenes_eval_for_subset():
    """Patch evaluator to support subset tokens, copied from verify_subset_eval."""
    from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval import NuScenesEval_custom
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
    from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval import (
        filter_eval_boxes_by_overlap,
    )
    import copy as _cp

    _orig_init = NuScenesEval_custom.__init__

    def _subset_safe_init(self, *args, **kwargs):
        try:
            _orig_init(self, *args, **kwargs)
        except AssertionError as exc:
            if "Samples in split" not in str(exc):
                raise
            pred_tokens = set(self.pred_boxes.sample_tokens)
            filtered_gt = EvalBoxes()
            for token in pred_tokens:
                boxes = self.gt_boxes.boxes.get(token, [])
                filtered_gt.add_boxes(token, boxes)
            self.gt_boxes = filtered_gt
            print(f"[PATCH] NuScenesEval subset mode: GT filtered to {len(pred_tokens)} pred tokens")

            self.pred_boxes = add_center_dist(self.nusc, self.pred_boxes)
            self.gt_boxes = add_center_dist(self.nusc, self.gt_boxes)
            self.pred_boxes = filter_eval_boxes(
                self.nusc, self.pred_boxes, self.class_range, verbose=self.verbose
            )
            self.gt_boxes = filter_eval_boxes(
                self.nusc, self.gt_boxes, self.class_range, verbose=self.verbose
            )

            if self.overlap_test:
                self.pred_boxes = filter_eval_boxes_by_overlap(self.nusc, self.pred_boxes)
                self.gt_boxes = filter_eval_boxes_by_overlap(
                    self.nusc, self.gt_boxes, verbose=True
                )

            self.all_gt = _cp.deepcopy(self.gt_boxes)
            self.all_preds = _cp.deepcopy(self.pred_boxes)
            self.sample_tokens = self.gt_boxes.sample_tokens

            self.index_map = {}
            for scene in self.nusc.scene:
                first_sample_token = scene["first_sample_token"]
                sample = self.nusc.get("sample", first_sample_token)
                self.index_map[first_sample_token] = 1
                index = 2
                while sample["next"] != "":
                    sample = self.nusc.get("sample", sample["next"])
                    self.index_map[sample["token"]] = index
                    index += 1

    NuScenesEval_custom.__init__ = _subset_safe_init
    print("[PATCH] NuScenesEval_custom patched for subset evaluation")

    from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval import (
        TrackingEval_custom,
        load_gt as _local_load_gt,
    )
    from nuscenes.eval.tracking.data_classes import TrackingBox
    from nuscenes.eval.common.loaders import (
        load_prediction,
        add_center_dist as _add_center_dist,
        filter_eval_boxes as _filter_eval_boxes,
    )
    from nuscenes.eval.tracking.loaders import create_tracks
    from nuscenes import NuScenes

    _orig_track_init = TrackingEval_custom.__init__

    def _subset_safe_track_init(
        self,
        config,
        result_path,
        eval_set,
        output_dir,
        nusc_version,
        nusc_dataroot,
        verbose=True,
        render_classes=None,
        splits={},
        category_to_type_name=None,
        class_range=None,
    ):
        try:
            _orig_track_init(
                self,
                config,
                result_path,
                eval_set,
                output_dir,
                nusc_version,
                nusc_dataroot,
                verbose=verbose,
                render_classes=render_classes,
                splits=splits,
                category_to_type_name=category_to_type_name,
                class_range=class_range,
            )
        except AssertionError as exc:
            if "Samples in split" not in str(exc):
                raise
            nusc = NuScenes(version=nusc_version, verbose=verbose, dataroot=nusc_dataroot)
            pred_boxes, self.meta = load_prediction(
                self.result_path, self.cfg.max_boxes_per_sample, TrackingBox, verbose=verbose
            )
            gt_boxes = _local_load_gt(
                nusc,
                self.eval_set,
                TrackingBox,
                verbose=verbose,
                splits=self.splits,
                category_to_type_name=self.category_to_type_name,
            )

            pred_tokens = set(pred_boxes.sample_tokens)
            filtered_gt = EvalBoxes()
            for token in pred_tokens:
                boxes = gt_boxes.boxes.get(token, [])
                filtered_gt.add_boxes(token, boxes)
            gt_boxes = filtered_gt
            print(f"[PATCH] TrackingEval subset mode: GT filtered to {len(pred_tokens)} pred tokens")

            pred_boxes = _add_center_dist(nusc, pred_boxes)
            gt_boxes = _add_center_dist(nusc, gt_boxes)
            pred_boxes = _filter_eval_boxes(nusc, pred_boxes, self.class_range, verbose=verbose)
            gt_boxes = _filter_eval_boxes(nusc, gt_boxes, self.class_range, verbose=verbose)
            self.sample_tokens = gt_boxes.sample_tokens
            self.tracks_gt = create_tracks(gt_boxes, nusc, self.eval_set, gt=True)
            self.tracks_pred = create_tracks(pred_boxes, nusc, self.eval_set, gt=False)

    TrackingEval_custom.__init__ = _subset_safe_track_init
    print("[PATCH] TrackingEval_custom patched for subset evaluation")


def reset_temporal_states(module: nn.Module) -> None:
    """Reset temporal/cache states for stable repeated evaluations."""
    for m in module.modules():
        for attr in (
            "test_track_instances",
            "prev_bev",
            "scene_token",
            "timestamp",
            "l2g_t",
            "l2g_r_mat",
        ):
            if hasattr(m, attr):
                setattr(m, attr, None)
        if hasattr(m, "prev_frame_info"):
            pfi = getattr(m, "prev_frame_info")
            if isinstance(pfi, dict):
                if "prev_bev" in pfi:
                    pfi["prev_bev"] = None
                if "scene_token" in pfi:
                    pfi["scene_token"] = None
                if "prev_pos" in pfi:
                    pfi["prev_pos"] = 0
                if "prev_angle" in pfi:
                    pfi["prev_angle"] = 0


def flatten_output_tensors(obj) -> Tuple[torch.Tensor, ...]:
    tensors: List[torch.Tensor] = []

    def _collect(x):
        if isinstance(x, torch.Tensor):
            tensors.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                _collect(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                _collect(v)

    _collect(obj)
    if not tensors:
        raise RuntimeError("No tensor found in backbone output during trace.")
    return tuple(tensors)


class BackbonePruneWrapper(nn.Module):
    """Trace/search wrapper so FastNAS sees only backbone graph."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, img: torch.Tensor):
        result = self.backbone(img)
        if torch.jit.is_tracing():
            outs = flatten_output_tensors(result)
            dep = img.float().sum() * 0
            return tuple(t + dep.to(device=t.device, dtype=t.dtype) for t in outs)
        return result


class FullModelWrapper(nn.Module):
    """Wrapper for saving/restoring full model with mto.save/mto.restore."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self._inner = model

    def forward(self, *args, **kwargs):
        return self._inner(*args, **kwargs)


def _collect_bn_stats(module: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """Collect BN running_mean/running_var/weight/bias from all BatchNorm layers."""
    stats = {}
    for name, m in module.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            entry = {}
            if m.running_mean is not None:
                entry["running_mean"] = m.running_mean.detach().cpu().clone()
            if m.running_var is not None:
                entry["running_var"] = m.running_var.detach().cpu().clone()
            if m.weight is not None:
                entry["weight"] = m.weight.detach().cpu().clone()
            if m.bias is not None:
                entry["bias"] = m.bias.detach().cpu().clone()
            stats[name] = entry
    return stats


def _validate_exported_backbone(
    original_backbone: nn.Module,
    exported_backbone: nn.Module,
    original_bn_stats: Dict[str, Dict[str, torch.Tensor]],
) -> bool:
    """Validate exported backbone: shape consistency, BN stats, weight sanity."""
    ok = True

    # Check 1: Layer existence
    orig_layers = {n for n, _ in original_backbone.named_modules()}
    export_layers = {n for n, _ in exported_backbone.named_modules()}
    missing = orig_layers - export_layers - {""}
    extra = export_layers - orig_layers - {""}
    if missing:
        print(f"[VALIDATE][WARN] layers in original but not in exported: {sorted(missing)[:10]}")
    if extra:
        print(f"[VALIDATE][WARN] layers in exported but not in original: {sorted(extra)[:10]}")

    # Check 2: Output feature dimensions for each stage (layer1-layer4)
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        orig_stage = getattr(original_backbone, stage_name, None)
        export_stage = getattr(exported_backbone, stage_name, None)
        if orig_stage is None or export_stage is None:
            continue
        # Check last block's conv3 (bottleneck output)
        orig_blocks = list(orig_stage.children())
        export_blocks = list(export_stage.children())
        if not orig_blocks or not export_blocks:
            continue
        orig_last = orig_blocks[-1]
        export_last = export_blocks[-1]
        if hasattr(orig_last, "conv3") and hasattr(export_last, "conv3"):
            oc_orig = orig_last.conv3.out_channels
            oc_export = export_last.conv3.out_channels
            if oc_orig != oc_export:
                print(
                    f"[VALIDATE][ERROR] {stage_name} output channels mismatch: "
                    f"original={oc_orig}, exported={oc_export}"
                )
                ok = False
            else:
                print(f"[VALIDATE][OK] {stage_name} output channels: {oc_export} (matches original)")

    # Check 3: BN running stats sanity
    export_bn_stats = _collect_bn_stats(exported_backbone)
    for bn_name, export_entry in export_bn_stats.items():
        rm = export_entry.get("running_mean", None)
        rv = export_entry.get("running_var", None)
        if rm is not None:
            if torch.isnan(rm).any() or torch.isinf(rm).any():
                print(f"[VALIDATE][ERROR] {bn_name}.running_mean has NaN/Inf!")
                ok = False
            if rm.abs().max().item() > 100:
                print(
                    f"[VALIDATE][WARN] {bn_name}.running_mean max abs={rm.abs().max().item():.4f} "
                    f"(unusually large)"
                )
        if rv is not None:
            if torch.isnan(rv).any() or torch.isinf(rv).any():
                print(f"[VALIDATE][ERROR] {bn_name}.running_var has NaN/Inf!")
                ok = False
            if (rv < 0).any():
                print(f"[VALIDATE][ERROR] {bn_name}.running_var has negative values!")
                ok = False
            if rv.abs().max().item() > 1000:
                print(
                    f"[VALIDATE][WARN] {bn_name}.running_var max abs={rv.abs().max().item():.4f} "
                    f"(unusually large)"
                )

    # Check 4: Weight all-zero check on conv layers
    for name, m in exported_backbone.named_modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight.detach()
            if w.abs().max().item() == 0:
                print(f"[VALIDATE][ERROR] {name}.weight is all zeros!")
                ok = False
            if torch.isnan(w).any():
                print(f"[VALIDATE][ERROR] {name}.weight has NaN!")
                ok = False

    if ok:
        print("[VALIDATE] all checks passed.")
    else:
        print("[VALIDATE] SOME CHECKS FAILED -- exported backbone may produce garbage features.")
    return ok


def parse_args():
    parser = argparse.ArgumentParser(
        description="FastNAS prune backbone only for univ2x_sub_inf_e2e_track"
    )
    parser.add_argument(
        "--config",
        default="./projects/configs_e2e_univ2x/univ2x_sub_inf_e2e_track.py",
        help="config path",
    )
    parser.add_argument(
        "--checkpoint",
        default="./ckpts/univ2x_sub_inf_stg1.pth",
        help="full model checkpoint path",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="output root dir")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="log root dir")
    parser.add_argument(
        "--out-name-prefix",
        default="univ2x_inf_track_backbone_pruned",
        help="output checkpoint name prefix",
    )
    parser.add_argument(
        "--search-checkpoint",
        default=None,
        help="search checkpoint file name (saved under run timestamp dir)",
    )

    parser.add_argument(
        "--params-percent",
        type=str,
        default="90%",
        help="upper bound of kept params ratio for backbone search",
    )
    parser.add_argument(
        "--channel-divisor",
        type=int,
        default=16,
        help="channel divisor for Conv/BN in fastnas config",
    )
    parser.add_argument(
        "--max-iter-data-loader",
        type=int,
        default=50,
        help="max calibration batches per round in FastNAS; <=0 means full loader",
    )
    parser.add_argument(
        "--trace-max-images",
        type=int,
        default=4,
        help="max images kept in backbone dummy_input/collect_func",
    )

    # --- Fix 1 & 4: constraints and pruning aggressiveness controls ---
    parser.add_argument(
        "--min-params-ratio",
        type=float,
        default=0.0,
        help=(
            "minimum backbone params retention ratio (0.0-1.0). "
            "If pruned backbone falls below this ratio, abort and warn. "
            "E.g., 0.8 means backbone must keep >= 80%% of original params."
        ),
    )
    parser.add_argument(
        "--min-channel-ratio",
        type=float,
        default=0.0,
        help=(
            "minimum per-layer channel retention ratio (0.0-1.0). "
            "When >0, increases channel_divisor per layer so each conv/bn keeps "
            "at least this fraction of original channels. E.g., 0.5 means each "
            "layer keeps >= 50%% of original channels."
        ),
    )
    parser.add_argument(
        "--lock-stage-outputs",
        action="store_true",
        default=True,
        help=(
            "lock ResNet stage output channels (layer1-4 conv3/downsample) to original values. "
            "Prevents dimension mismatch with downstream FPN/neck. (default: enabled)"
        ),
    )
    parser.add_argument(
        "--no-lock-stage-outputs",
        dest="lock_stage_outputs",
        action="store_false",
        help="disable locking stage output channels",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        default=False,
        help="skip pre/post export diagnostic evaluations (saves time but less debug info)",
    )

    parser.add_argument(
        "--calib-split",
        choices=["train", "val", "test"],
        default="train",
        help="BN calibration split",
    )
    parser.add_argument(
        "--score-split",
        choices=["train", "val", "test"],
        default="val",
        help="subnet scoring split",
    )
    parser.add_argument(
        "--calib-subset-mode",
        choices=["full", "complete", "head", "skip_warmup"],
        default="full",
        help="subset mode for calib split",
    )
    parser.add_argument(
        "--score-subset-mode",
        choices=["full", "complete", "head", "skip_warmup"],
        default="full",
        help="subset mode for score split",
    )
    parser.add_argument(
        "--calib-frames-per-scene",
        type=int,
        default=10,
        help="frames per scene when calib subset mode is head/skip_warmup",
    )
    parser.add_argument(
        "--score-frames-per-scene",
        type=int,
        default=10,
        help="frames per scene when score subset mode is head/skip_warmup",
    )
    parser.add_argument(
        "--eval-mods",
        nargs="+",
        default=["det"],
        choices=["det", "track", "map", "motion"],
        help="dataset eval_mod for scoring evaluation",
    )
    parser.add_argument(
        "--train-use-test-pipeline",
        dest="train_use_test_pipeline",
        action="store_true",
        default=True,
        help="for train split loader, force cfg.data.test.pipeline (default: enabled)",
    )
    parser.add_argument(
        "--no-train-use-test-pipeline",
        dest="train_use_test_pipeline",
        action="store_false",
        help="disable forcing cfg.data.test.pipeline for train split",
    )
    parser.add_argument(
        "--token-align-check",
        action="store_true",
        help="run output token order alignment check on first score evaluation",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, key=value",
    )
    args = parser.parse_args()

    args.calib_subset_mode = "full" if args.calib_subset_mode == "complete" else args.calib_subset_mode
    args.score_subset_mode = "full" if args.score_subset_mode == "complete" else args.score_subset_mode
    args.params_percent = str(args.params_percent).strip().rstrip(",")

    if args.calib_frames_per_scene <= 0 or args.score_frames_per_scene <= 0:
        parser.error("--calib-frames-per-scene and --score-frames-per-scene must be > 0")
    if args.trace_max_images <= 0:
        parser.error("--trace-max-images must be > 0")
    return args


def apply_unfreeze_cfg(cfg: Config) -> None:
    """Remove known freeze settings in config before model build."""
    model_keys = [k for k in cfg.keys() if k.startswith("model_")]
    if "model_ego_agent" not in model_keys:
        model_keys.append("model_ego_agent")
    for mk in model_keys:
        if mk not in cfg:
            continue
        mcfg = cfg.get(mk)
        if not isinstance(mcfg, dict):
            continue
        if "load_from" in mcfg:
            mcfg["load_from"] = None
        if "pretrained" in mcfg:
            mcfg["pretrained"] = None
        for flag in ("freeze_img_backbone", "freeze_img_neck", "freeze_bn"):
            if flag in mcfg:
                mcfg[flag] = False
        if "img_backbone" in mcfg and isinstance(mcfg["img_backbone"], dict):
            ib = mcfg["img_backbone"]
            if "frozen_stages" in ib:
                ib["frozen_stages"] = 0
            if "norm_eval" in ib:
                ib["norm_eval"] = False
            if "norm_cfg" in ib and isinstance(ib["norm_cfg"], dict):
                if "requires_grad" in ib["norm_cfg"]:
                    ib["norm_cfg"]["requires_grad"] = True


def build_multi_agent_model(cfg: Config, checkpoint_path: str) -> nn.Module:
    other_agent_names = [k for k in cfg.keys() if "model_other_agent" in k]
    model_other_agents = {}

    for name in other_agent_names:
        cfg.get(name).train_cfg = None
        model = build_model(cfg.get(name), test_cfg=cfg.get("test_cfg"))
        load_from = cfg.get(name).get("load_from", None)
        if load_from:
            load_checkpoint(
                model,
                load_from,
                map_location="cpu",
                revise_keys=[(r"^model_ego_agent\.", "")],
            )
        model_other_agents[name] = model

    cfg.model_ego_agent.train_cfg = None
    model_ego_agent = build_model(cfg.model_ego_agent, test_cfg=cfg.get("test_cfg"))
    load_from = cfg.model_ego_agent.get("load_from", None)
    if load_from:
        load_checkpoint(
            model_ego_agent,
            load_from,
            map_location="cpu",
            revise_keys=[(r"^model_ego_agent\.", "")],
        )

    model_multi_agents = MultiAgent(model_ego_agent, model_other_agents)
    checkpoint = load_checkpoint(model_multi_agents, checkpoint_path, map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model_multi_agents.model_ego_agent.CLASSES = checkpoint["meta"]["CLASSES"]
    if "PALETTE" in checkpoint.get("meta", {}):
        model_multi_agents.model_ego_agent.PALETTE = checkpoint["meta"]["PALETTE"]
    return model_multi_agents


def build_loader_with_subset(
    cfg: Config,
    split: str,
    subset_mode: str,
    frames_per_scene: int,
    train_use_test_pipeline: bool,
):
    if split in ("val", "test"):
        ds_cfg = copy.deepcopy(cfg.data.val if split == "val" else cfg.data.test)
        ds_cfg.test_mode = True
        samples_per_gpu = ds_cfg.pop("samples_per_gpu", 1)
    else:
        ds_cfg = copy.deepcopy(cfg.data.train)
        ds_cfg.test_mode = True
        if train_use_test_pipeline and hasattr(cfg.data, "test"):
            ds_cfg.pipeline = copy.deepcopy(cfg.data.test.pipeline)
        samples_per_gpu = ds_cfg.pop("samples_per_gpu", 1)

    full_dataset = build_dataset(ds_cfg)
    if subset_mode == "full":
        dataset = full_dataset
        print(f"[INFO] split={split} subset_mode=full, size={len(dataset)}")
    else:
        indices = fixed_head_indices(
            full_dataset,
            frames_per_scene=frames_per_scene,
            sampling_mode=subset_mode,
        )
        dataset = SubsetDataset(full_dataset, indices)
        print(
            f"[INFO] split={split} subset_mode={subset_mode}, "
            f"frames_per_scene={frames_per_scene}, subset size={len(dataset)}"
        )

    loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.get("nonshuffler_sampler", None),
    )
    return dataset, loader


def safe_first_batch(loader, split: str, train_use_test_pipeline: bool):
    """Get first batch with clearer diagnostics for known pipeline mismatch."""
    try:
        return next(iter(loader))
    except Exception as exc:
        msg = str(exc)
        if (
            split == "train"
            and not train_use_test_pipeline
            and ("DataContainer" in msg or "img_metas" in msg)
        ):
            raise RuntimeError(
                "Failed to load first batch for train split with current pipeline. "
                "This dataset's test-mode path expects test pipeline output format. "
                "Retry with default behavior (use test pipeline for train split), "
                "or pass --train-use-test-pipeline."
            ) from exc
        raise


def _unwrap_container(x):
    while not isinstance(x, (torch.Tensor, np.ndarray)) and hasattr(x, "data"):
        x = x.data
    return x


def extract_backbone_input_from_batch(batch_on_dev: Dict[str, Any], max_images: int) -> torch.Tensor:
    """Extract pure Tensor input for backbone trace/calibration from one batch."""
    ego = batch_on_dev["ego_agent_data"]
    img = ego.get("img", None)
    if img is None:
        raise RuntimeError("Cannot find ego_agent_data['img'] in batch.")

    img = _unwrap_container(img)
    if isinstance(img, (list, tuple)):
        # Common case: list with one tensor.
        if len(img) == 0:
            raise RuntimeError("ego_agent_data['img'] is empty list/tuple.")
        img = _unwrap_container(img[0])

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    if not isinstance(img, torch.Tensor):
        raise RuntimeError(f"Unsupported img type for backbone input: {type(img)}")

    # Expected possibilities:
    #   6D: [B, T, N, C, H, W]
    #   5D: [B, N, C, H, W] or [B, T, C, H, W]
    #   4D: [B, C, H, W]
    # Convert to [M, C, H, W] and keep first max_images.
    if img.dim() == 6:
        # Use only first temporal step to avoid huge dummy input.
        x = img[:, 0]  # [B, N, C, H, W]
        x = x.reshape(-1, *x.shape[-3:])
    elif img.dim() == 5:
        x = img.reshape(-1, *img.shape[-3:])
    elif img.dim() == 4:
        x = img
    elif img.dim() == 3:
        x = img.unsqueeze(0)
    else:
        raise RuntimeError(f"Unsupported img tensor dim for backbone: {img.dim()}")

    if x.shape[0] > max_images:
        x = x[:max_images]
    return x.detach().contiguous()


def get_map_score(eval_res: Dict[str, Any]) -> float:
    """Global mAP objective over all classes."""
    if "pts_bbox_NuScenes/mAP" in eval_res:
        score = float(eval_res["pts_bbox_NuScenes/mAP"])
        if not math.isnan(score):
            return score

    # Fallback: average all AP-distance keys.
    ap_vals = []
    for k, v in eval_res.items():
        if "_AP_dist_" in k:
            val = float(v)
            if not math.isnan(val):
                ap_vals.append(val)
    if ap_vals:
        return float(sum(ap_vals) / len(ap_vals))
    raise KeyError("No valid mAP/AP keys found in evaluation result.")


def get_class_mean_ap(eval_res: Dict[str, Any], class_name: str) -> float:
    keys = [
        f"pts_bbox_NuScenes/{class_name}_AP_dist_0.5",
        f"pts_bbox_NuScenes/{class_name}_AP_dist_1.0",
        f"pts_bbox_NuScenes/{class_name}_AP_dist_2.0",
        f"pts_bbox_NuScenes/{class_name}_AP_dist_4.0",
    ]
    vals = []
    for k in keys:
        if k in eval_res:
            v = float(eval_res[k])
            if not math.isnan(v):
                vals.append(v)
    if not vals:
        raise KeyError(f"No valid AP keys found for class {class_name}")
    return float(sum(vals) / len(vals))


def print_focus_metrics(tag: str, eval_res: Dict[str, Any]) -> None:
    map_score = get_map_score(eval_res)
    print(f"[INFO] {tag} mAP: {map_score:.6f}")
    for cls in ("car", "pedestrian", "bicycle"):
        try:
            cls_ap = get_class_mean_ap(eval_res, cls)
            print(f"[INFO] {tag} {cls} AP: {cls_ap:.6f}")
        except Exception:
            print(f"[INFO] {tag} {cls} AP: N/A")


@torch.no_grad()
def eval_full_model_map(
    full_model: nn.Module,
    eval_dataset,
    eval_loader,
    eval_kwargs: Dict[str, Any],
    gpu_id: int,
    check_token_alignment: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    reset_temporal_states(full_model)
    full_model = full_model.cuda(gpu_id).eval()
    model_dp = MMDataParallel(full_model, device_ids=[gpu_id])
    outputs = single_gpu_test(model_dp, eval_loader, show=False)

    if check_token_alignment:
        assert_output_token_alignment(eval_dataset, outputs)

    for i in range(len(outputs)):
        det = outputs[i]
        if isinstance(det, dict):
            det.setdefault("ret_iou", {})
        elif isinstance(det, list) and len(det) > 0 and isinstance(det[0], dict):
            det[0].setdefault("ret_iou", {})

    res = eval_dataset.evaluate(outputs, **eval_kwargs)
    score = get_map_score(res)
    return float(score), res


def build_score_func(
    full_model: nn.Module,
    eval_dataset,
    eval_loader,
    eval_kwargs: Dict[str, Any],
    gpu_id: int,
    tracker: ScoreFuncTracker,
    token_align_check: bool,
):
    call_count = [0]

    def score_func(prunable_backbone_model: nn.Module) -> float:
        call_count[0] += 1
        call_id = call_count[0]
        print("\n" + "=" * 64)
        print(f"[score_func] call #{call_id}")
        print("=" * 64)
        t0 = time.time()

        # Refill candidate backbone into full model for real subnet eval.
        if hasattr(prunable_backbone_model, "backbone"):
            full_model.model_ego_agent.img_backbone = prunable_backbone_model.backbone

        score, _ = eval_full_model_map(
            full_model=full_model,
            eval_dataset=eval_dataset,
            eval_loader=eval_loader,
            eval_kwargs=eval_kwargs,
            gpu_id=gpu_id,
            check_token_alignment=(token_align_check and call_id == 1),
        )
        elapsed = time.time() - t0
        tracker.record(call_id=call_id, score=score, elapsed_sec=elapsed)
        print(f"[score_func] done call #{call_id}: mAP={score:.6f}, elapsed={elapsed:.2f}s")
        return float(score)

    return score_func


def _compute_layer_min_divisors(
    backbone: nn.Module,
    min_channel_ratio: float,
    base_divisor: int,
) -> Dict[str, int]:
    """Compute per-layer channel_divisor to enforce minimum channel retention.

    For each Conv2d layer, the effective channel_divisor is set to
    max(base_divisor, out_channels * min_channel_ratio) rounded up to
    multiple of base_divisor. This ensures the smallest searchable
    channel count >= out_channels * min_channel_ratio.
    """
    overrides = {}
    if min_channel_ratio <= 0:
        return overrides
    for name, m in backbone.named_modules():
        if isinstance(m, nn.Conv2d):
            oc = m.out_channels
            min_ch = int(math.ceil(oc * min_channel_ratio))
            # Round up to multiple of base_divisor
            min_ch = max(base_divisor, ((min_ch + base_divisor - 1) // base_divisor) * base_divisor)
            if min_ch > base_divisor:
                overrides[name] = min_ch
    return overrides


def _collect_stage_output_locks(
    backbone: nn.Module,
) -> Dict[str, int]:
    """Identify ResNet stage-output conv layers whose out_channels should be locked.

    These are the conv3 and downsample.0 layers in each stage's bottleneck blocks.
    Locking them prevents dimension mismatch with downstream FPN/neck.
    """
    locks = {}
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        stage = getattr(backbone, stage_name, None)
        if stage is None:
            continue
        for blk_idx, blk in enumerate(stage.children()):
            # Lock conv3 (bottleneck expansion output) -- this defines the stage output dim
            if hasattr(blk, "conv3") and isinstance(blk.conv3, nn.Conv2d):
                key = f"backbone.{stage_name}.{blk_idx}.conv3"
                locks[key] = blk.conv3.out_channels
            # Lock downsample projection to match
            if hasattr(blk, "downsample") and blk.downsample is not None:
                for ds_idx, ds_mod in enumerate(blk.downsample):
                    if isinstance(ds_mod, nn.Conv2d):
                        key = f"backbone.{stage_name}.{blk_idx}.downsample.{ds_idx}"
                        locks[key] = ds_mod.out_channels
    return locks


def build_backbone_fastnas_mode(
    channel_divisor: int,
    backbone: nn.Module = None,
    min_channel_ratio: float = 0.0,
    lock_stage_outputs: bool = True,
):
    """FastNAS mode config for backbone-only model.

    Args:
        channel_divisor: base channel divisor for search granularity.
        backbone: backbone module (needed for min_channel_ratio / lock_stage_outputs).
        min_channel_ratio: minimum per-layer channel retention ratio (0 = disabled).
        lock_stage_outputs: lock ResNet conv3/downsample output channels to original values.
    """
    conv2d_cfg = {"*": {"channel_divisor": channel_divisor}}
    bn2d_cfg = {"*": {"feature_divisor": channel_divisor}}

    # Fix 4a: per-layer min channel enforcement
    if backbone is not None and min_channel_ratio > 0:
        layer_mins = _compute_layer_min_divisors(backbone, min_channel_ratio, channel_divisor)
        for layer_name, min_div in layer_mins.items():
            # Use backbone.xxx prefix to match the wrapper's namespace
            prefixed = f"backbone.{layer_name}"
            conv2d_cfg[prefixed] = {"channel_divisor": min_div}
        if layer_mins:
            print(f"[INFO] min_channel_ratio={min_channel_ratio}: "
                  f"raised channel_divisor for {len(layer_mins)} conv layers")

    # Fix 4b: lock stage output channels (conv3 + downsample) to prevent FPN mismatch
    if backbone is not None and lock_stage_outputs:
        stage_locks = _collect_stage_output_locks(backbone)
        for layer_key, out_ch in stage_locks.items():
            # Set channel_divisor = out_channels so the only searchable value IS the original
            conv2d_cfg[layer_key] = {"channel_divisor": out_ch}
        if stage_locks:
            print(f"[INFO] lock_stage_outputs: locked {len(stage_locks)} stage-output conv layers")
            for k, v in sorted(stage_locks.items()):
                print(f"  {k}: locked at {v} channels")

    # Also lock corresponding BN layers
    if backbone is not None and lock_stage_outputs:
        for stage_name in ("layer1", "layer2", "layer3", "layer4"):
            stage = getattr(backbone, stage_name, None)
            if stage is None:
                continue
            for blk_idx, blk in enumerate(stage.children()):
                if hasattr(blk, "bn3") and isinstance(blk.bn3, (nn.BatchNorm2d,)):
                    key = f"backbone.{stage_name}.{blk_idx}.bn3"
                    bn2d_cfg[key] = {"feature_divisor": blk.bn3.num_features}
                if hasattr(blk, "downsample") and blk.downsample is not None:
                    for ds_idx, ds_mod in enumerate(blk.downsample):
                        if isinstance(ds_mod, (nn.BatchNorm2d,)):
                            key = f"backbone.{stage_name}.{blk_idx}.downsample.{ds_idx}"
                            bn2d_cfg[key] = {"feature_divisor": ds_mod.num_features}

    fastnas_cfg = {
        "nn.Conv2d": conv2d_cfg,
        "nn.BatchNorm2d": bn2d_cfg,
        "nn.Linear": None,
        "nn.Conv1d": None,
        "nn.Conv3d": None,
        "nn.ConvTranspose1d": None,
        "nn.ConvTranspose2d": None,
        "nn.ConvTranspose3d": None,
        "nn.BatchNorm1d": None,
        "nn.BatchNorm3d": None,
        "nn.SyncBatchNorm": None,
        "nn.InstanceNorm1d": None,
        "nn.InstanceNorm2d": None,
        "nn.InstanceNorm3d": None,
        "nn.LayerNorm": None,
        "nn.GroupNorm": None,
        "hf.BertAttention": None,
        "hf.GPTJAttention": None,
    }
    return [("fastnas", fastnas_cfg)]


def main():
    args = parse_args()
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_out_dir = osp.join(args.out_dir, run_ts)
    os.makedirs(run_out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = setup_logging(args.log_dir, run_ts)
    print(f"[INFO] run log: {log_path}")
    print(f"[INFO] run timestamp: {run_ts}")
    print(f"[INFO] output root dir: {args.out_dir}")
    print(f"[INFO] run output dir: {run_out_dir}")
    print(f"[INFO] config: {args.config}")
    print(f"[INFO] checkpoint: {args.checkpoint}")
    print(
        f"[INFO] subset modes: calib={args.calib_subset_mode}, "
        f"score={args.score_subset_mode}"
    )

    patch_torchprofile_dtype_setter()
    patch_nuscenes_eval_for_subset()

    if args.search_checkpoint:
        ckpt_name_raw = osp.basename(args.search_checkpoint)
        stem, ext = osp.splitext(ckpt_name_raw)
        ext = ext if ext else ".pth"
        ckpt_name = f"{stem}_{run_ts}{ext}"
    else:
        ckpt_name = f"fastnas_search_ckpt_{run_ts}.pth"
    search_ckpt = osp.join(run_out_dir, ckpt_name)
    os.makedirs(osp.dirname(search_ckpt) or run_out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    apply_unfreeze_cfg(cfg)

    if getattr(cfg, "plugin", False):
        import importlib

        plugin_dir = getattr(cfg, "plugin_dir", osp.dirname(args.config))
        module_path = ".".join(osp.dirname(plugin_dir).split("/"))
        print(f"[INFO] import plugin module: {module_path}")
        importlib.import_module(module_path)

    set_random_seed(args.seed, deterministic=args.deterministic)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pruning workflow.")
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    print(f"[INFO] device: {device}")

    print("[INFO] build full model + load checkpoint...")
    full_model = build_multi_agent_model(cfg, args.checkpoint).to(device).eval()
    for p in full_model.parameters():
        p.requires_grad = True
    total_params_full = sum(p.numel() for p in full_model.parameters())
    print(f"[INFO] full model params before pruning: {total_params_full / 1e6:.3f} M")

    if not hasattr(full_model, "model_ego_agent") or not hasattr(
        full_model.model_ego_agent, "img_backbone"
    ):
        raise RuntimeError("Cannot find model.model_ego_agent.img_backbone in full model.")
    original_backbone = full_model.model_ego_agent.img_backbone
    backbone_params_before = sum(p.numel() for p in original_backbone.parameters())
    print(f"[INFO] backbone params before pruning: {backbone_params_before / 1e6:.3f} M")

    # ===========================================================================
    # FIX 3 (CRITICAL): Deep copy backbone to prevent mtp.prune from modifying
    # full_model's backbone in-place via Python reference aliasing.
    #
    # Without this, backbone_wrapper.backbone IS the same object as
    # full_model.model_ego_agent.img_backbone. When mtp.prune converts the
    # wrapper to a supernet, it corrupts full_model's backbone. After export,
    # the exported backbone may have wrong weights/BN stats because the
    # original backbone object was destroyed.
    # ===========================================================================
    print("[INFO] deep copying backbone for pruning (prevents reference aliasing)...")
    backbone_for_prune = copy.deepcopy(original_backbone)
    backbone_wrapper = BackbonePruneWrapper(backbone_for_prune).to(device).eval()

    # Save original BN stats for post-export validation (Fix 5)
    original_bn_stats = _collect_bn_stats(original_backbone)
    print(f"[INFO] saved BN stats from {len(original_bn_stats)} original BN layers")

    print("[INFO] build calib loader ...")
    calib_dataset, calib_loader = build_loader_with_subset(
        cfg=cfg,
        split=args.calib_split,
        subset_mode=args.calib_subset_mode,
        frames_per_scene=args.calib_frames_per_scene,
        train_use_test_pipeline=args.train_use_test_pipeline,
    )
    print(f"[INFO] calib dataset size: {len(calib_dataset)}")

    print("[INFO] build score loader ...")
    score_dataset, score_loader = build_loader_with_subset(
        cfg=cfg,
        split=args.score_split,
        subset_mode=args.score_subset_mode,
        frames_per_scene=args.score_frames_per_scene,
        train_use_test_pipeline=args.train_use_test_pipeline,
    )
    score_dataset.eval_mod = list(args.eval_mods)
    if hasattr(score_dataset, "_dataset"):
        score_dataset._dataset.eval_mod = list(args.eval_mods)
    print(f"[INFO] score dataset size: {len(score_dataset)}, eval_mod={args.eval_mods}")

    eval_kwargs = cfg.get("evaluation", {}).copy()
    for k in ("interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"):
        eval_kwargs.pop(k, None)
    eval_kwargs["metric"] = ["bbox"]
    eval_kwargs["jsonfile_prefix"] = osp.join(
        run_out_dir,
        "score_eval_tmp",
        osp.basename(args.config).rsplit(".", 1)[0],
        run_ts,
    )

    # Baseline A: full val
    print("[INFO] build full-val baseline loader ...")
    full_val_dataset, full_val_loader = build_loader_with_subset(
        cfg=cfg,
        split="val",
        subset_mode="full",
        frames_per_scene=args.score_frames_per_scene,
        train_use_test_pipeline=args.train_use_test_pipeline,
    )
    full_val_dataset.eval_mod = list(args.eval_mods)
    if hasattr(full_val_dataset, "_dataset"):
        full_val_dataset._dataset.eval_mod = list(args.eval_mods)

    baseline_full_val_map, baseline_full_val_res = eval_full_model_map(
        full_model=full_model,
        eval_dataset=full_val_dataset,
        eval_loader=full_val_loader,
        eval_kwargs=eval_kwargs,
        gpu_id=args.gpu_id,
        check_token_alignment=False,
    )
    print(f"[INFO] baseline full-model mAP on full val: {baseline_full_val_map:.6f}")
    print_focus_metrics("baseline full-val", baseline_full_val_res)

    # Baseline B: score split (same subset as pruning score set)
    baseline_map, baseline_res = eval_full_model_map(
        full_model=full_model,
        eval_dataset=score_dataset,
        eval_loader=score_loader,
        eval_kwargs=eval_kwargs,
        gpu_id=args.gpu_id,
        check_token_alignment=args.token_align_check,
    )
    print(f"[INFO] baseline full-model mAP on score split: {baseline_map:.6f}")
    print_focus_metrics("baseline score-split", baseline_res)
    if "pts_bbox_NuScenes/mAP" in baseline_res:
        print(
            f"[INFO] baseline eval key pts_bbox_NuScenes/mAP={baseline_res['pts_bbox_NuScenes/mAP']}"
        )

    print("[INFO] build backbone dummy_input ...")
    raw_batch = safe_first_batch(
        calib_loader,
        split=args.calib_split,
        train_use_test_pipeline=args.train_use_test_pipeline,
    )
    batch_on_device = scatter(raw_batch, [args.gpu_id])[0]
    x0 = extract_backbone_input_from_batch(
        batch_on_device, max_images=args.trace_max_images
    ).to(device)
    dummy_input = (x0,)
    print(f"[INFO] dummy_input shape: {tuple(x0.shape)}")

    def collect_func(raw_batch):
        b = scatter(raw_batch, [args.gpu_id])[0]
        x = extract_backbone_input_from_batch(
            b, max_images=args.trace_max_images
        ).to(device)
        return (x,)

    score_tracker = ScoreFuncTracker()
    score_func = build_score_func(
        full_model=full_model,
        eval_dataset=score_dataset,
        eval_loader=score_loader,
        eval_kwargs=eval_kwargs,
        gpu_id=args.gpu_id,
        tracker=score_tracker,
        token_align_check=args.token_align_check,
    )

    prune_constraints = {"params": args.params_percent}
    # Fix 1 & 4: build fastnas mode with min_channel_ratio and lock_stage_outputs
    prune_mode = build_backbone_fastnas_mode(
        channel_divisor=args.channel_divisor,
        backbone=original_backbone,
        min_channel_ratio=args.min_channel_ratio,
        lock_stage_outputs=args.lock_stage_outputs,
    )
    max_iter_data_loader = (
        args.max_iter_data_loader if args.max_iter_data_loader > 0 else None
    )
    print(f"[INFO] constraints: {prune_constraints}")
    print("[INFO] prune target: backbone only")
    print(f"[INFO] max_iter_data_loader: {max_iter_data_loader}")
    print(f"[INFO] search checkpoint: {search_ckpt}")

    pruned_backbone_wrapper = None
    prune_res = None
    try:
        pruned_backbone_wrapper, prune_res = mtp.prune(
            model=backbone_wrapper,
            mode=prune_mode,
            constraints=prune_constraints,
            dummy_input=dummy_input,
            config={
                "data_loader": calib_loader,
                "score_func": score_func,
                "checkpoint": search_ckpt,
                "collect_func": collect_func,
                "max_iter_data_loader": max_iter_data_loader,
            },
        )
    finally:
        score_tracker.print_report()
        score_hist_path = osp.join(run_out_dir, f"score_calls_{run_ts}.pkl")
        mmcv.dump(score_tracker.call_history, score_hist_path)
        print(f"[INFO] saved score call history: {score_hist_path}")

    if pruned_backbone_wrapper is None or prune_res is None:
        raise RuntimeError("Pruning failed before producing pruned backbone model.")

    tracker_summary = score_tracker.get_summary()
    if tracker_summary.get("num_calls", 0) > 0:
        print(
            f"[INFO] search-phase best score_func mAP: "
            f"{tracker_summary['best_score']:.6f} "
            f"(call #{tracker_summary['best_call_id']:03d})"
        )

    # ===========================================================================
    # FIX 2: Pre-export diagnostic evaluations
    # Evaluate at three stages to pinpoint where accuracy is lost:
    #   A) full_model with whatever backbone state mtp.prune left (supernet?)
    #   B) full_model with original backbone restored (should match baseline)
    #   C) full_model with exported pruned backbone (the final result)
    # ===========================================================================
    diag_a_map = None
    diag_b_map = None

    if not args.skip_diagnostics:
        print("\n" + "=" * 72)
        print("[DIAG] === Pre-export diagnostic evaluations ===")
        print("=" * 72)

        # DIAG A: Evaluate full_model with current backbone state
        # After mtp.prune + score_func calls, full_model's backbone is the
        # supernet copy (from the last score_func assignment). The best subnet
        # should be active. This tells us if the supernet state is valid.
        print("\n[DIAG-A] eval full model with current backbone (supernet, best subnet active)...")
        diag_a_map, _ = eval_full_model_map(
            full_model=full_model,
            eval_dataset=score_dataset,
            eval_loader=score_loader,
            eval_kwargs=eval_kwargs,
            gpu_id=args.gpu_id,
        )
        print(f"[DIAG-A] supernet backbone mAP: {diag_a_map:.6f}")

        # DIAG B: Restore original backbone and verify it still works
        print("\n[DIAG-B] restoring original backbone and verifying (should match baseline)...")
        full_model.model_ego_agent.img_backbone = original_backbone
        diag_b_map, _ = eval_full_model_map(
            full_model=full_model,
            eval_dataset=score_dataset,
            eval_loader=score_loader,
            eval_kwargs=eval_kwargs,
            gpu_id=args.gpu_id,
        )
        print(f"[DIAG-B] original backbone mAP: {diag_b_map:.6f} (baseline was {baseline_map:.6f})")
        if abs(diag_b_map - baseline_map) > 0.005:
            print(
                f"[DIAG-B][WARN] original backbone mAP differs from baseline by "
                f"{abs(diag_b_map - baseline_map):.6f} -- possible state corruption!"
            )
        else:
            print("[DIAG-B][OK] original backbone integrity verified.")

    # ===========================================================================
    # FIX 5: Validate exported backbone before inserting into full model
    # ===========================================================================
    exported_backbone = None
    if hasattr(pruned_backbone_wrapper, "backbone"):
        exported_backbone = pruned_backbone_wrapper.backbone

    if exported_backbone is not None:
        print("\n" + "=" * 72)
        print("[VALIDATE] === Post-export backbone validation (Fix 5) ===")
        print("=" * 72)
        validation_ok = _validate_exported_backbone(
            original_backbone=original_backbone,
            exported_backbone=exported_backbone,
            original_bn_stats=original_bn_stats,
        )
    else:
        print("[WARN] pruned_backbone_wrapper has no .backbone attribute!")
        validation_ok = False

    # Insert exported backbone into full model
    if exported_backbone is not None:
        full_model.model_ego_agent.img_backbone = exported_backbone
    else:
        print("[WARN] no exported backbone to insert, keeping current backbone.")

    backbone_params_after = sum(
        p.numel() for p in full_model.model_ego_agent.img_backbone.parameters()
    )
    full_params_after = sum(p.numel() for p in full_model.parameters())
    backbone_kept_ratio = 100.0 * backbone_params_after / backbone_params_before if backbone_params_before > 0 else 0
    print(f"[INFO] backbone params after pruning: {backbone_params_after / 1e6:.3f} M")
    print(f"[INFO] full model params after pruning: {full_params_after / 1e6:.3f} M")
    if backbone_params_before > 0:
        print(f"[INFO] backbone kept ratio: {backbone_kept_ratio:.2f}%")
    if total_params_full > 0:
        print(
            f"[INFO] full-model kept ratio: {100.0 * full_params_after / total_params_full:.2f}%"
        )

    # ===========================================================================
    # FIX 1: Post-pruning min-params-ratio enforcement
    # ===========================================================================
    if args.min_params_ratio > 0 and backbone_params_before > 0:
        actual_ratio = backbone_params_after / backbone_params_before
        if actual_ratio < args.min_params_ratio:
            print(
                f"\n[ABORT] backbone retention ratio {actual_ratio:.4f} "
                f"< min_params_ratio {args.min_params_ratio:.4f}. "
                f"Pruning was too aggressive!"
            )
            print(
                f"[ABORT] backbone: {backbone_params_after / 1e6:.3f}M kept "
                f"vs minimum required: {backbone_params_before * args.min_params_ratio / 1e6:.3f}M"
            )
            print(
                "[ABORT] Suggestions:\n"
                "  1. Increase --min-channel-ratio (e.g., 0.5 or 0.75)\n"
                "  2. Use --lock-stage-outputs (default enabled)\n"
                "  3. Use tighter --params-percent (e.g., '95%' instead of '90%')\n"
                "  4. Increase --channel-divisor to reduce search granularity"
            )
            # Still save outputs for analysis, but mark as aborted
            print("[ABORT] saving outputs for analysis before exit...")

    # DIAG C: Final evaluation with exported backbone
    print("\n[DIAG-C] eval full model with exported pruned backbone...")
    final_map, final_res = eval_full_model_map(
        full_model=full_model,
        eval_dataset=score_dataset,
        eval_loader=score_loader,
        eval_kwargs=eval_kwargs,
        gpu_id=args.gpu_id,
        check_token_alignment=False,
    )
    print(f"[INFO] final selected backbone mAP on score split: {final_map:.6f}")
    print(f"[INFO] selected subnet evaluated score (mAP): {final_map:.6f}")

    # Print diagnostic comparison summary
    if not args.skip_diagnostics:
        print("\n" + "=" * 72)
        print("[DIAG] === Diagnostic Summary ===")
        print(f"  baseline mAP (original model):           {baseline_map:.6f}")
        if diag_b_map is not None:
            print(f"  DIAG-B (original backbone restored):     {diag_b_map:.6f}")
        if diag_a_map is not None:
            print(f"  DIAG-A (supernet, best subnet active):   {diag_a_map:.6f}")
        print(f"  DIAG-C (exported pruned backbone):       {final_map:.6f}")
        print(f"  backbone kept ratio:                     {backbone_kept_ratio:.2f}%")
        print(f"  validation passed:                       {validation_ok}")
        print("=" * 72)
        if diag_a_map is not None and diag_a_map > 0 and final_map == 0:
            print(
                "[DIAG][CONCLUSION] supernet works but export failed! "
                "The subnet export process corrupted weights or BN stats."
            )
        elif diag_a_map is not None and diag_a_map == 0:
            print(
                "[DIAG][CONCLUSION] supernet also gives 0 mAP. "
                "The selected subnet configuration itself is catastrophically bad."
            )
        if diag_b_map is not None and abs(diag_b_map - baseline_map) > 0.005:
            print(
                "[DIAG][CONCLUSION] original backbone was corrupted despite deep copy! "
                "Check if mtp.prune has side effects beyond the wrapper."
            )

    out_prefix = f"{args.out_name_prefix}_{run_ts}"
    backbone_modelopt_path = osp.join(run_out_dir, f"{out_prefix}_backbone_modelopt.pth")
    full_modelopt_path = osp.join(run_out_dir, f"{out_prefix}_full_modelopt.pth")
    prune_stats_path = osp.join(run_out_dir, f"{out_prefix}_prune_stats.pkl")
    summary_path = osp.join(run_out_dir, f"{out_prefix}_summary.pkl")

    print(f"[INFO] save pruned backbone modelopt: {backbone_modelopt_path}")
    mto.save(pruned_backbone_wrapper, backbone_modelopt_path)

    print(f"[INFO] save full model with pruned backbone (modelopt): {full_modelopt_path}")
    mto.save(FullModelWrapper(full_model), full_modelopt_path)

    mmcv.dump(prune_res, prune_stats_path)
    summary = {
        "run_ts": run_ts,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "calib_split": args.calib_split,
        "score_split": args.score_split,
        "calib_subset_mode": args.calib_subset_mode,
        "score_subset_mode": args.score_subset_mode,
        "calib_frames_per_scene": args.calib_frames_per_scene,
        "score_frames_per_scene": args.score_frames_per_scene,
        "baseline_full_val_map": baseline_full_val_map,
        "baseline_map": baseline_map,
        "final_map": final_map,
        "diag_a_supernet_map": diag_a_map,
        "diag_b_original_restored_map": diag_b_map,
        "validation_passed": validation_ok,
        "backbone_kept_ratio": backbone_kept_ratio,
        "min_params_ratio": args.min_params_ratio,
        "min_channel_ratio": args.min_channel_ratio,
        "lock_stage_outputs": args.lock_stage_outputs,
        "score_func_num_calls": tracker_summary.get("num_calls", 0),
        "score_func_score_min": tracker_summary.get("score_min", None),
        "score_func_score_max": tracker_summary.get("score_max", None),
        "score_func_score_mean": tracker_summary.get("score_mean", None),
        "score_func_best_call_id": tracker_summary.get("best_call_id", None),
        "score_func_best_score": tracker_summary.get("best_score", None),
        "backbone_params_before": backbone_params_before,
        "backbone_params_after": backbone_params_after,
        "full_params_before": total_params_full,
        "full_params_after": full_params_after,
        "backbone_modelopt_path": backbone_modelopt_path,
        "full_modelopt_path": full_modelopt_path,
        "prune_stats_path": prune_stats_path,
        "search_checkpoint": search_ckpt,
    }
    mmcv.dump(summary, summary_path)
    print(f"[INFO] saved prune stats: {prune_stats_path}")
    print(f"[INFO] saved summary: {summary_path}")

    print("\n" + "=" * 72)
    print("[INFO] Backbone-only pruning finished.")
    print(f"[INFO] baseline mAP={baseline_map:.6f} -> final mAP={final_map:.6f}")
    print(f"[INFO] backbone kept ratio: {backbone_kept_ratio:.2f}%")
    print(f"[INFO] outputs: {run_out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
