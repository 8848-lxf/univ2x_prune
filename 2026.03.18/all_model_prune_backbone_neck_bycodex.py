import os
import sys
import atexit
import signal

# ---- Ensure UniV2X repo root is on sys.path ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # UniV2X/univ2x_purned
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))  # UniV2X
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import copy
import math
import os.path as osp
import time
import warnings
import numpy as np
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, scatter
from mmcv.runner import load_checkpoint

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed

# NVIDIA Model Optimizer
import modelopt.torch.prune as mtp
import modelopt.torch.opt as mto
# from modelopt.torch.opt.dynamic import DynamicConv2dConfig, DynamicLinearConfig
from modelopt.torch.opt.utils import named_hparams

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent

warnings.filterwarnings("ignore")


# =============================================================================
# Patch: torchprofile Variable.dtype setter — scalarType() 对某些 tensor
#   返回 None，原始 setter 调用 dtype.lower() 会 AttributeError。
# =============================================================================
from torchprofile.utils.ir.variable import Variable as _TorchProfileVariable

@_TorchProfileVariable.dtype.setter          # type: ignore[attr-defined]
def _safe_dtype_setter(self, dtype):
    self._dtype = dtype.lower() if dtype is not None else 'unknown'

_TorchProfileVariable.dtype = _safe_dtype_setter
print("[PATCH] torchprofile Variable.dtype setter patched for None scalarType")


# =============================================================================
# Patch 0：NuScenesEval_custom 子集评估补丁
#   score_func 只推理 210 帧，pred_tokens ⊂ GT tokens（675 帧）；
#   原始 assert 会崩溃，此处捕获后把 GT 过滤到 pred_tokens 子集再继续。
# =============================================================================
def patch_nuscenes_eval_for_subset():
    from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval import NuScenesEval_custom
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
    from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval import filter_eval_boxes_by_overlap
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

            # 继续执行原始 __init__ 中 assert 之后的流程，保证评估行为与全量一致。
            self.pred_boxes = add_center_dist(self.nusc, self.pred_boxes)
            self.gt_boxes = add_center_dist(self.nusc, self.gt_boxes)
            self.pred_boxes = filter_eval_boxes(
                self.nusc, self.pred_boxes, self.class_range, verbose=self.verbose)
            self.gt_boxes = filter_eval_boxes(
                self.nusc, self.gt_boxes, self.class_range, verbose=self.verbose)

            if self.overlap_test:
                self.pred_boxes = filter_eval_boxes_by_overlap(self.nusc, self.pred_boxes)
                self.gt_boxes = filter_eval_boxes_by_overlap(self.nusc, self.gt_boxes, verbose=True)

            self.all_gt = _cp.deepcopy(self.gt_boxes)
            self.all_preds = _cp.deepcopy(self.pred_boxes)
            self.sample_tokens = self.gt_boxes.sample_tokens

            # index_map：scene 内帧序号（与原实现保持一致）
            self.index_map = {}
            for scene in self.nusc.scene:
                first_sample_token = scene['first_sample_token']
                sample = self.nusc.get('sample', first_sample_token)
                self.index_map[first_sample_token] = 1
                index = 2
                while sample['next'] != '':
                    sample = self.nusc.get('sample', sample['next'])
                    self.index_map[sample['token']] = index
                    index += 1

    NuScenesEval_custom.__init__ = _subset_safe_init
    print("[PATCH] NuScenesEval_custom patched for subset evaluation")

    # ---- 同理 patch TrackingEval_custom ----
    from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval import (
        TrackingEval_custom, load_gt as _local_load_gt,
    )
    from nuscenes.eval.tracking.data_classes import TrackingBox
    from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
    from nuscenes.eval.tracking.loaders import create_tracks
    from nuscenes import NuScenes

    _orig_track_init = TrackingEval_custom.__init__

    def _subset_safe_track_init(self, config, result_path, eval_set, output_dir,
                                nusc_version, nusc_dataroot, verbose=True,
                                render_classes=None, splits={},
                                category_to_type_name=None, class_range=None):
        try:
            _orig_track_init(self, config, result_path, eval_set, output_dir,
                             nusc_version, nusc_dataroot, verbose=verbose,
                             render_classes=render_classes, splits=splits,
                             category_to_type_name=category_to_type_name,
                             class_range=class_range)
        except AssertionError as exc:
            if "Samples in split" not in str(exc):
                raise
            # assert 在 __init__ 中间触发，局部变量丢失；
            # self.cfg/result_path/eval_set/splits/class_range 等已设好，
            # 重新加载 pred/gt 并过滤 GT 到 pred 子集。
            nusc = NuScenes(version=nusc_version, verbose=verbose, dataroot=nusc_dataroot)
            pred_boxes, self.meta = load_prediction(
                self.result_path, self.cfg.max_boxes_per_sample, TrackingBox, verbose=verbose)
            gt_boxes = _local_load_gt(
                nusc, self.eval_set, TrackingBox, verbose=verbose,
                splits=self.splits, category_to_type_name=self.category_to_type_name)

            # 过滤 GT 到 pred tokens 子集
            pred_tokens = set(pred_boxes.sample_tokens)
            filtered_gt = EvalBoxes()
            for token in pred_tokens:
                boxes = gt_boxes.boxes.get(token, [])
                filtered_gt.add_boxes(token, boxes)
            gt_boxes = filtered_gt
            print(f"[PATCH] TrackingEval subset mode: GT filtered to {len(pred_tokens)} pred tokens")

            pred_boxes = add_center_dist(nusc, pred_boxes)
            gt_boxes = add_center_dist(nusc, gt_boxes)
            pred_boxes = filter_eval_boxes(nusc, pred_boxes, self.class_range, verbose=verbose)
            gt_boxes = filter_eval_boxes(nusc, gt_boxes, self.class_range, verbose=verbose)
            self.sample_tokens = gt_boxes.sample_tokens
            self.tracks_gt = create_tracks(gt_boxes, nusc, self.eval_set, gt=True)
            self.tracks_pred = create_tracks(pred_boxes, nusc, self.eval_set, gt=False)

    TrackingEval_custom.__init__ = _subset_safe_track_init
    print("[PATCH] TrackingEval_custom patched for subset evaluation")

    # ---- 同理 patch MotionEval ----
    from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval_motion import MotionEval

    _orig_motion_init = MotionEval.__init__

    def _subset_safe_motion_init(self, *args, **kwargs):
        try:
            _orig_motion_init(self, *args, **kwargs)
        except AssertionError as exc:
            if "Samples in split" not in str(exc):
                raise
            # MotionEval 结构同 NuScenesEval_custom：
            # assert 之前 self.pred_boxes / self.gt_boxes / self.nusc 已赋值
            pred_tokens = set(self.pred_boxes.sample_tokens)
            filtered_gt = EvalBoxes()
            for token in pred_tokens:
                boxes = self.gt_boxes.boxes.get(token, [])
                filtered_gt.add_boxes(token, boxes)
            self.gt_boxes = filtered_gt
            print(f"[PATCH] MotionEval subset mode: GT filtered to {len(pred_tokens)} pred tokens")

            # 继续 assert 之后的步骤（对应原始 __init__ line 443-478）
            from nuscenes.eval.common.loaders import add_center_dist as _acd, filter_eval_boxes as _feb
            self.pred_boxes = _acd(self.nusc, self.pred_boxes)
            self.gt_boxes = _acd(self.nusc, self.gt_boxes)
            self.pred_boxes = _feb(self.nusc, self.pred_boxes, self.cfg.class_range, verbose=self.verbose)
            self.gt_boxes = _feb(self.nusc, self.gt_boxes, self.cfg.class_range, verbose=self.verbose)

            if self.overlap_test:
                from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval_motion import filter_eval_boxes_by_overlap
                self.pred_boxes = filter_eval_boxes_by_overlap(self.nusc, self.pred_boxes)
                self.gt_boxes = filter_eval_boxes_by_overlap(self.nusc, self.gt_boxes, verbose=True)

            import copy as _cp
            self.all_gt = _cp.deepcopy(self.gt_boxes)
            self.all_preds = _cp.deepcopy(self.pred_boxes)
            self.sample_tokens = self.gt_boxes.sample_tokens

            # index_map：scene 内帧序号
            nusc = self.nusc
            self.index_map = {}
            for scene in nusc.scene:
                first_sample_token = scene['first_sample_token']
                sample = nusc.get('sample', first_sample_token)
                self.index_map[first_sample_token] = 1
                index = 2
                while sample['next'] != '':
                    sample = nusc.get('sample', sample['next'])
                    self.index_map[sample['token']] = index
                    index += 1

    MotionEval.__init__ = _subset_safe_motion_init
    print("[PATCH] MotionEval patched for subset evaluation")


patch_nuscenes_eval_for_subset()


# =============================================================================
# 连续采样：与 verify_subset_eval.py 一致
#   - head: 从每个场景第 1 帧开始连续取 N 帧
#   - skip_warmup: 跳过每个场景前 4 帧，从第 5 帧开始连续取 N 帧
# =============================================================================
def fixed_head_indices(dataset, frames_per_scene=10, sampling_mode="skip_warmup"):
    """每个 scene 连续采样固定帧数（时间顺序）。"""
    scene_to_indices = defaultdict(list)
    for i, info in enumerate(dataset.data_infos):
        scene_to_indices[info["scene_token"]].append(i)

    scenes = sorted(scene_to_indices.keys())
    selected = []
    if sampling_mode == "head":
        start_idx = 0
    elif sampling_mode == "skip_warmup":
        start_idx = 4  # 第 5 帧（1-based）
    else:
        raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")

    for s in scenes:
        idxs = scene_to_indices[s]
        selected.extend(idxs[start_idx:start_idx + frames_per_scene])

    selected = sorted(selected)

    scene_counts = defaultdict(int)
    for i in selected:
        scene_counts[dataset.data_infos[i]["scene_token"]] += 1

    print(f"[INFO] 连续采样（mode={sampling_mode}, start_idx={start_idx}）："
          f"覆盖 {len(scene_counts)} / {len(scenes)} 个场景，总帧数={len(selected)}")
    print(f"       每场景帧数：min={min(scene_counts.values())} "
          f"max={max(scene_counts.values())} "
          f"avg={np.mean(list(scene_counts.values())):.1f}")
    return selected


# =============================================================================
# SubsetDataset：真正裁剪 data_infos，使评估器只看子集 token
#   同步裁剪 data_infos → _format_bbox 里 sample_token 正确
#   __len__ 返回子集大小 → format_results 里的 assert 通过
#   evaluate / format_results 委托给原始 dataset，临时替换 data_infos
# =============================================================================
class SubsetDataset(torch.utils.data.Dataset):
    _PASSTHROUGH = [
        "CLASSES", "eval_mod", "eval_detection_configs", "eval_version",
        "modality", "overlap_test", "version", "data_root", "nusc",
        "nusc_maps", "ErrNameMapping", "with_velocity",
        "split_datas_file", "class_range", "tmp_dataset_type", "planning_steps",
    ]

    def __init__(self, dataset, indices):
        self._dataset = dataset
        self._indices = list(indices)
        # 核心：同步裁剪 data_infos
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
    """硬校验：dataset 顺序与 model 输出顺序是否一一对应。"""
    ds_tokens = [info["token"] for info in eval_dataset.data_infos]
    assert len(outputs) == len(ds_tokens), (
        f"输出数量({len(outputs)}) != 数据集数量({len(ds_tokens)})"
    )

    out_tokens = []
    for out in outputs:
        token = None
        if isinstance(out, dict):
            token = out.get("token", None)
        elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            token = out[0].get("token", None)
        out_tokens.append(token)

    # 若模型输出不带 token，不做强制失败；后续由 evaluator token 对齐再检查。
    if all(t is None for t in out_tokens):
        print("[CHECK][WARN] 输出中未找到 token 字段，跳过逐样本 token 顺序校验。")
        return

    missing = [i for i, t in enumerate(out_tokens) if t is None]
    assert not missing, f"输出中存在缺失 token 的样本，示例索引: {missing[:10]}"

    mismatch = []
    for i, (pred_t, ds_t) in enumerate(zip(out_tokens, ds_tokens)):
        if pred_t != ds_t:
            mismatch.append((i, pred_t, ds_t))
            if len(mismatch) >= 10:
                break

    assert not mismatch, (
        "输出 token 与 dataset token 顺序不一致，示例: "
        + "; ".join([f"idx={i}, pred={p}, ds={d}" for i, p, d in mismatch])
    )
    print(f"[CHECK] 输出 token 与 dataset token 一一对应，共 {len(ds_tokens)} 条。")


# ===================== Tensor 提取 / 回填工具 =====================
# 将任意嵌套 dict/list/tuple 结构中的 Tensor 和 ndarray 全部提取为
# 扁平 tuple，同时保留"骨架"（skeleton）——其中非 tensor 值原样保留，
# tensor 位置用 _TensorSlot 占位。forward 时用 _fill_tensors 把
# tensor tuple 回填到骨架，重建模型需要的完整输入。
# =================================================================

class _TensorSlot:
    """标记骨架中一个 tensor 被提取的位置。"""
    __slots__ = ('idx', 'is_ndarray', 'np_dtype')
    def __init__(self, idx, is_ndarray=False, np_dtype=None):
        self.idx = idx
        self.is_ndarray = is_ndarray
        self.np_dtype = np_dtype


def _extract_tensors(obj):
    """递归提取 Tensor / ndarray，返回 (skeleton, tensor_list)。

    skeleton 保留所有非 tensor 值（str, bool, int, enum, type 等），
    tensor / ndarray 位置用 _TensorSlot 标记。ndarray 会被转为 Tensor。
    """
    tensors = []

    def _walk(x):
        # 解包 mmcv DataContainer
        if not isinstance(x, (torch.Tensor, np.ndarray)) and hasattr(x, 'data'):
            x = x.data
        if isinstance(x, torch.Tensor):
            slot = _TensorSlot(len(tensors))
            tensors.append(x.detach().contiguous())
            return slot
        if isinstance(x, np.ndarray):
            slot = _TensorSlot(len(tensors), is_ndarray=True, np_dtype=x.dtype)
            tensors.append(torch.from_numpy(x.copy()))
            return slot
        if isinstance(x, dict):
            return {k: _walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_walk(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_walk(v) for v in x)
        return x   # str, bool, int, float, enum, type …

    skeleton = _walk(obj)
    return skeleton, tensors


def _fill_tensors(skeleton, tensors):
    """用 tensor tuple 回填骨架，重建原始嵌套结构。

    ndarray 类型的数据始终转回 numpy（模型内部依赖 numpy 操作，如
    can_bus[:3] -= prev_pos）。这些数据在 trace 中会变为常量，但模型
    内部随后会调用 torch.tensor() 转为 tensor 参与计算，FLOPs 仍会
    被正确统计。
    """
    def _walk(x):
        if isinstance(x, _TensorSlot):
            t = tensors[x.idx]
            if x.is_ndarray:
                return t.cpu().numpy().astype(x.np_dtype)
            return t
        if isinstance(x, dict):
            return {k: _walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_walk(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_walk(v) for v in x)
        return x

    return _walk(skeleton)


def _flatten_output_tensors(obj):
    """递归收集模型输出中的所有 Tensor，返回 tuple。"""
    tensors = []
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
        raise RuntimeError("模型输出中未找到任何 Tensor，请检查 forward 逻辑")
    return tuple(tensors)


# ======================== ModelOptWrapper ========================
# 核心思路：把模型的全部输入拆分为
#   tensor 部分  → forward(*tensors) 的显式参数 → trace 的输入节点
#   非 tensor 部分 → self._skeleton 旁路（str/bool/enum/type 等）
#
# 这样 trace 能看到所有 tensor 的数据依赖链：
#   - img / other_img → backbone → neck → BEV → heads → 预测输出
#   - sdc_planning 等 GT tensor → 透传到输出（但它们也是 forward 参数，
#     有数据依赖，不会触发 "no observable data dependence" 错误）
#   - lidar2img, can_bus 等 → BEV encoder 投影 / 位置编码
# =================================================================
class ModelOptWrapper(nn.Module):
    def __init__(self, model: nn.Module, trace_prev_bev_mode: str = "warmup"):
        super().__init__()
        self._inner = model
        self._skeleton = None   # _extract_tensors 生成的骨架
        valid_modes = {"none", "cold", "warmup"}
        if trace_prev_bev_mode not in valid_modes:
            raise ValueError(
                f"Invalid trace_prev_bev_mode={trace_prev_bev_mode}, "
                f"expected one of {sorted(valid_modes)}"
            )
        self._trace_prev_bev_mode = trace_prev_bev_mode

    def set_skeleton(self, skeleton):
        """保存骨架（含非 tensor 元数据 + _TensorSlot 占位符）。"""
        self._skeleton = skeleton

    @staticmethod
    def _reset_temporal_states(module: nn.Module):
        """重置 UniV2X 里与时序相关的内部状态，保证 trace 可复现。"""
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

    def _build_full_input(self, tensors):
        if self._skeleton is None:
            raise RuntimeError("ModelOptWrapper._skeleton 尚未设置，请先调用 set_skeleton()")
        return _fill_tensors(self._skeleton, tensors)

    def _run_inner(self, full_input):
        with torch.no_grad():
            return self._inner(
                ego_agent_data=full_input["ego"],
                other_agent_data_dict=full_input["others"],
                img_metas=full_input["img_metas"],
                return_loss=False,
                w_label=True,
            )

    def forward(self, *tensors) -> Any:
        is_tracing = torch.jit.is_tracing()

        # trace 时控制 prev_bev 时序路径：
        #   - cold:   重置后单次 forward，不走 prev_bev 分支
        #   - warmup: 重置后先 warmup 一次，再第二次 forward，稳定走 prev_bev 分支
        #   - none:   不做任何控制
        if is_tracing and self._trace_prev_bev_mode in {"cold", "warmup"}:
            self._reset_temporal_states(self._inner)

        if is_tracing and self._trace_prev_bev_mode == "warmup":
            warmup_input = self._build_full_input(tensors)
            _ = self._run_inner(warmup_input)

        # 回填 tensor → 重建完整输入（ndarray 始终转回 numpy）
        full_input = self._build_full_input(tensors)
        result = self._run_inner(full_input)

        if is_tracing:
            flat = _flatten_output_tensors(result)
            # 注入对全部 input tensor 的微弱依赖（+0），
            # 防止个别输出（如 track_ids）因为不直接依赖某些输入而报错
            _dep = sum(t.float().sum() for t in tensors) * 0
            return tuple(t + _dep.to(device=t.device, dtype=t.dtype) for t in flat)
        return result


# --------------------------- 日志工具 ---------------------------
LOG_DIR = "/home/lixingfeng/UniAD_exmaine/UniV2X/univ2x_purned/modelopt/all_model_prune_backbone_neck_bycodex/logs_all_model_prune_backbone_neck_bycodex"

class _Tee:
    """Python 层 tee：终端和日志文件都完整输出。"""
    def __init__(self, stream, log_fh):
        self._stream = stream
        self._log_fh = log_fh

    def write(self, data):
        # 终端：原样输出
        self._stream.write(data)
        self._stream.flush()
        # 日志文件：完整记录所有内容（含完整进度条）
        if data:
            self._log_fh.write(data)
            self._log_fh.flush()

    def flush(self):
        self._stream.flush()
        self._log_fh.flush()

    def __getattr__(self, attr):
        return getattr(self._stream, attr)


def setup_logging() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"run_{timestamp}.log")

    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)

    # 只用 Python 层 _Tee 重定向，不做 os.dup2 fd 层重定向。
    # os.dup2 会把 C 层 write(fd=1) 也导入日志文件（含进度条），
    # 同时截断了终端的 fd 1，导致进度条在终端消失。
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    def _flush_and_close():
        try:
            log_fh.flush()
            log_fh.close()
        except Exception:
            pass

    atexit.register(_flush_and_close)

    _orig_sigint = signal.getsignal(signal.SIGINT)
    def _sigint_handler(sig, frame):
        print("\n[INFO] 收到 SIGINT，正在刷盘日志...", file=sys.__stderr__)
        _flush_and_close()
        signal.signal(signal.SIGINT, _orig_sigint)
        signal.raise_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, _sigint_handler)

    return log_path


# --------------------------- 参数解析 ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="FastNAS pruning for MultiAgent model")
    parser.add_argument("--gpu-id", type=int, default=1, help="使用的 GPU 编号")
    parser.add_argument("config", help="配置文件路径")
    parser.add_argument("checkpoint", help="待剪枝的完整模型权重路径")
    parser.add_argument(
        "--out-dir",
        default="/home/lixingfeng/UniAD_exmaine/UniV2X/univ2x_purned/modelopt/all_model_prune_backbone_neck_bycodex/output",
        help="剪枝输出目录，文件名自动带时间戳",
    )
    parser.add_argument(
        "--search-checkpoint",
        default=None,
        help="断点续搜：指定已有的搜索 checkpoint 路径。不指定则每次从头搜索",
    )
    # ---- 校准集参数（BN 统计量更新，用训练集子集） ----
    parser.add_argument(
        "--calib-split",
        choices=["train", "val", "test"],
        default="train",
        help="BN 校准数据集划分（推荐 train，BN 统计量应匹配训练分布）",
    )
    parser.add_argument(
        "--calib-subset-mode",
        choices=["full", "head", "skip_warmup"],
        default="head",
        help="校准集采样模式：full=全量，head=每场景头部连续采样，skip_warmup=每场景跳过前4帧后连续采样",
    )
    parser.add_argument(
        "--calib-frames-per-scene",
        type=int,
        default=5,
        help="校准集每场景取的帧数",
    )
    parser.add_argument(
        "--max-iter-data-loader",
        type=int,
        default=50,
        help="FastNAS 每轮 BN 校准最多迭代的 batch 数。>0 表示截断，<=0 表示使用完整 calib_loader。",
    )
    # ---- 评分集参数（候选子网排序，用 val 子集） ----
    parser.add_argument(
        "--score-split",
        choices=["train", "val", "test"],
        default="val",
        help="评分数据集划分（推荐 val，评估泛化性能）",
    )
    parser.add_argument(
        "--score-subset-mode",
        choices=["full", "head", "skip_warmup"],
        default="head",
        help="评分集采样模式：full=全量，head=每场景头部连续采样，skip_warmup=每场景跳过前4帧后连续采样",
    )
    parser.add_argument(
        "--score-frames-per-scene",
        type=int,
        default=10,
        help="评分集每场景取的帧数（head 模式下：21 场景 × 10 帧 = 210 帧）",
    )
    parser.add_argument(
        "--params-percent",
        type=str,
        default="99.5%",
        help="剪枝后参数量占原始模型的上限比例，例如 '95%%'",
    )
    parser.add_argument(
        "--trace-prev-bev-mode",
        choices=["none", "cold", "warmup"],
        default="warmup",
        help="控制 trace 时 prev_bev 时序路径：none=不控制，cold=强制首帧路径，warmup=先预热再trace第二帧路径",
    )
    parser.add_argument(
        "--eval-modes",
        nargs="+",
        default=["det"],
        choices=["det", "track", "motion", "map"],
        help="score_func 评估时启用的模式（默认只 det）。"
             "例如: --eval-modes det track motion",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="覆盖配置文件中的选项，格式为 key=value",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="兼容旧命令保留参数；当前脚本评分评估固定走单卡，忽略该参数。",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument(
        "--tmpdir",
        default=None,
        help="兼容旧命令保留参数；单卡评估模式下不使用。",
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="兼容旧命令保留参数；单卡评估模式下不使用。",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


# --------------------------- 模型构建 ---------------------------
def build_multi_agent_model(cfg, checkpoint_path):
    other_agent_names = [key for key in cfg.keys() if "model_other_agent" in key]
    model_other_agents = {}

    for name in other_agent_names:
        cfg.get(name).train_cfg = None
        model = build_model(cfg.get(name), test_cfg=cfg.get("test_cfg"))
        load_from = cfg.get(name).load_from
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
    load_from = cfg.model_ego_agent.load_from
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


# --------------------------- DataLoader：全量/子集 ---------------------------
def build_subset_loader(cfg, split="val", subset_mode="head",
                        frames_per_scene=10, dist_mode=False):
    """
    构建 full/head/skip_warmup 三种模式的 DataLoader。
    """
    if split in ("test", "val"):
        ds_cfg = copy.deepcopy(cfg.data.test if split == "test" else cfg.data.val)
        ds_cfg.test_mode = True
        samples_per_gpu = ds_cfg.pop("samples_per_gpu", 1)
    else:
        ds_cfg = copy.deepcopy(cfg.data.train)
        ds_cfg.test_mode = True    # 校准时用推理模式，不加数据增强
        # train config 使用 train pipeline，但 test_mode 需要 test pipeline
        ds_cfg.pipeline = copy.deepcopy(cfg.data.test.pipeline)
        samples_per_gpu = ds_cfg.pop("samples_per_gpu", 1)

    # 1. 构建完整 dataset
    full_dataset = build_dataset(ds_cfg)
    print(f"[INFO] 完整 {split} 集大小: {len(full_dataset)}")

    # 2. 按模式采样
    if subset_mode == "full":
        subset = full_dataset
        print(f"[INFO] 使用完整 {split} 集（mode=full），样本数: {len(subset)}")
    else:
        indices = fixed_head_indices(
            full_dataset,
            frames_per_scene=frames_per_scene,
            sampling_mode=subset_mode,
        )
        # 3. SubsetDataset：裁剪 data_infos，评估器只看子集 token
        subset = SubsetDataset(full_dataset, indices)
        print(f"[INFO] 子集大小: {len(subset)} 帧（mode={subset_mode}）")

    # 4. 构建 DataLoader
    loader = build_dataloader(
        subset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=dist_mode,
        shuffle=False,
        nonshuffler_sampler=cfg.data.get("nonshuffler_sampler", None),
    )
    return subset, loader


# --------------------------- score func ---------------------------
@torch.no_grad()
def _eval_car_mean_ap(model_dp: MMDataParallel, dataset, data_loader,
                      eval_kwargs: Dict[str, Any]) -> float:
    print(f"[score_func] >>>>>> ENTERING _eval_car_mean_ap, "
          f"dataset size={len(dataset)}, loader batches={len(data_loader)} <<<<<<",
          flush=True)
    model_dp.eval()
    outputs = single_gpu_test(model_dp, data_loader, show=False)

    if isinstance(outputs, dict):
        bbox_outputs = outputs.get("bbox_results", None)
    else:
        bbox_outputs = outputs
    if bbox_outputs is None:
        raise RuntimeError("single_gpu_test 返回结果为空，无法评估。")

    assert_output_token_alignment(dataset, bbox_outputs)

    for i in range(len(bbox_outputs)):
        det = bbox_outputs[i]
        if isinstance(det, dict):
            det.setdefault("ret_iou", {})
        elif isinstance(det, list) and len(det) > 0 and isinstance(det[0], dict):
            det[0].setdefault("ret_iou", {})

    if isinstance(outputs, dict):
        outputs["bbox_results"] = bbox_outputs
    else:
        outputs = bbox_outputs

    res = dataset.evaluate(outputs, **eval_kwargs)

    keys = [
        "pts_bbox_NuScenes/car_AP_dist_0.5",
        "pts_bbox_NuScenes/car_AP_dist_1.0",
        "pts_bbox_NuScenes/car_AP_dist_2.0",
        "pts_bbox_NuScenes/car_AP_dist_4.0",
    ]
    vals = [float(res[k]) for k in keys if k in res and not math.isnan(float(res[k]))]
    if not vals:
        raise ValueError("All car AP values are NaN; cannot compute score.")
    mean_ap = sum(vals) / len(vals)
    print(f"[score_func] car mean AP = {mean_ap:.4f}  (APs: {[f'{v:.4f}' for v in vals]})")

    # 打印所有评估指标（det/track/motion），方便观察剪枝对各任务的影响
    extra_keys = {
        "track": ["pts_bbox_NuScenes/amota", "pts_bbox_NuScenes/amotp", "pts_bbox_NuScenes/recall"],
        "motion": [k for k in res if "motion" in k.lower() or "minFDE" in k or "EPA" in k],
        "det_summary": ["pts_bbox_NuScenes/NDS", "pts_bbox_NuScenes/mAP"],
    }
    for group, ks in extra_keys.items():
        found = {k: f"{float(res[k]):.4f}" for k in ks if k in res}
        if found:
            print(f"[score_func] {group}: {found}")
    return mean_ap

def build_score_func(dataset, val_loader, eval_kwargs: Dict[str, Any], gpu_id: int):
    _call_count = [0]  # mutable counter in closure

    def score_func(model: nn.Module) -> float:
        _call_count[0] += 1
        print(f"\n{'='*60}", flush=True)
        print(f"[score_func] >>>>>> CALLED (#{_call_count[0]}) <<<<<<", flush=True)
        print(f"[score_func] model type = {type(model).__name__}, "
              f"has _inner = {hasattr(model, '_inner')}", flush=True)
        print(f"{'='*60}", flush=True)
        real_model = model._inner if hasattr(model, "_inner") else model
        real_model = real_model.cuda(gpu_id)
        model_dp = MMDataParallel(
            real_model,
            device_ids=[gpu_id],
        )
        score = _eval_car_mean_ap(model_dp, dataset, val_loader, eval_kwargs)
        print(f"[score_func] >>>>>> RETURNING score = {score:.6f} (call #{_call_count[0]}) <<<<<<\n",
              flush=True)
        return score
    return score_func

# ======================== BN 校准迭代追踪器 ========================
# 记录 FastNAS 使用 calib_loader 的轮次与 batch 消耗情况，用于验证
# max_iter_data_loader 是否生效以及每轮实际前向次数。
# ==================================================================

class CalibIterTracker:
    """追踪 BN 校准 data_loader 的迭代轮次和 collect_func 调用。"""

    def __init__(self):
        self.round_history = []  # [{round_id, timestamp, yielded_batches, collect_calls, ...}]
        self.total_collect_calls = 0
        self._next_round_id = 0
        self._active_round_id = None

    def reset(self):
        self.round_history.clear()
        self.total_collect_calls = 0
        self._next_round_id = 0
        self._active_round_id = None

    def _find_round(self, round_id: int):
        for rec in self.round_history:
            if rec["round_id"] == round_id:
                return rec
        raise KeyError(f"round_id={round_id} not found in calib tracker")

    def start_round(self, loader_batches: Optional[int] = None) -> int:
        import traceback as _tb

        self._next_round_id += 1
        round_id = self._next_round_id
        now = time.time()

        stack = _tb.extract_stack()
        caller_frames = []
        for frame in stack:
            if "modelopt" in frame.filename or "run_forward_loop" in frame.name:
                caller_frames.append(f"  {frame.filename}:{frame.lineno} in {frame.name}")
        caller_info = "\n".join(caller_frames[-5:]) if caller_frames else "(unknown caller)"

        rec = {
            "round_id": round_id,
            "start_time": now,
            "end_time": None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "end_timestamp": None,
            "loader_batches": loader_batches,
            "yielded_batches": 0,
            "collect_calls": 0,
            "end_reason": None,
            "caller_info": caller_info,
        }
        self.round_history.append(rec)
        self._active_round_id = round_id
        print(f"[CALIB] >>>>>> BN 校准迭代开始 (round #{round_id}, loader batches={loader_batches}) <<<<<<",
              flush=True)
        return round_id

    def mark_batch_yield(self, round_id: int):
        rec = self._find_round(round_id)
        rec["yielded_batches"] += 1
        self._active_round_id = round_id

    def mark_collect_call(self):
        self.total_collect_calls += 1
        round_id = self._active_round_id
        if round_id is None:
            print("[CALIB][WARN] collect_func 被调用时未检测到 active round。", flush=True)
            return
        rec = self._find_round(round_id)
        rec["collect_calls"] += 1

    def end_round(self, round_id: int, reason: str):
        rec = self._find_round(round_id)
        if rec["end_time"] is not None:
            return
        end_time = time.time()
        rec["end_time"] = end_time
        rec["end_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
        rec["end_reason"] = reason
        elapsed = end_time - rec["start_time"]
        print(
            f"[CALIB] round #{round_id} 结束: collect_calls={rec['collect_calls']}, "
            f"yielded={rec['yielded_batches']}, reason={reason}, elapsed={elapsed:.2f}s",
            flush=True,
        )
        if self._active_round_id == round_id:
            self._active_round_id = None

    def print_report(self, configured_max_iters: Optional[int]):
        print(f"\n{'='*70}")
        print("[CALIB_TRACKER] BN 校准 data_loader 调用报告")
        print(f"{'='*70}")
        print(f"[CALIB_TRACKER] 总轮次: {len(self.round_history)}")
        print(f"[CALIB_TRACKER] collect_func 总调用次数: {self.total_collect_calls}")
        if configured_max_iters is None:
            print("[CALIB_TRACKER] 配置 max_iter_data_loader = None（每轮遍历完整 calib_loader）")
        else:
            print(f"[CALIB_TRACKER] 配置 max_iter_data_loader = {configured_max_iters} (batch/轮)")
        print("[CALIB_TRACKER] 注：yielded_batches 可能比 collect_calls 大 1，"
              "因为 run_forward_loop 先取 batch 再检查 idx 上限。")

        if not self.round_history:
            print("[CALIB_TRACKER][WARN] 未检测到任何 BN 校准轮次。")
            print(f"{'='*70}\n")
            return

        # 收尾未正常结束的轮次（例如被 max_iter 截断导致迭代器提前释放）
        for rec in self.round_history:
            if rec["end_time"] is None:
                self.end_round(rec["round_id"], reason="stopped_early_or_max_iter")

        collect_vals = [rec["collect_calls"] for rec in self.round_history]
        print(
            f"[CALIB_TRACKER] 每轮 collect_func 调用范围: "
            f"[{min(collect_vals)}, {max(collect_vals)}], "
            f"avg={sum(collect_vals)/len(collect_vals):.2f}"
        )
        print("[CALIB_TRACKER] 轮次明细:")
        for rec in self.round_history:
            print(
                f"  #{rec['round_id']}  {rec['timestamp']} -> {rec['end_timestamp']}  "
                f"collect={rec['collect_calls']}  yielded={rec['yielded_batches']}  "
                f"reason={rec['end_reason']}"
            )
            if rec["caller_info"]:
                for line in rec["caller_info"].split("\n"):
                    print(f"    {line}")
        print(f"{'='*70}\n")


class _TrackedDataLoaderIter:
    """包装原始迭代器，记录每轮 BN 校准的 batch 消耗。"""

    def __init__(self, base_iter, tracker: CalibIterTracker, round_id: int):
        self._base_iter = base_iter
        self._tracker = tracker
        self._round_id = round_id
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self._base_iter)
        except StopIteration:
            self.close("exhausted")
            raise
        self._tracker.mark_batch_yield(self._round_id)
        return batch

    def close(self, reason: str):
        if self._closed:
            return
        self._closed = True
        self._tracker.end_round(self._round_id, reason)

    def __del__(self):
        try:
            self.close("stopped_early_or_max_iter")
        except Exception:
            # 进程退出阶段保持静默，避免析构时报错影响主流程
            pass


class TrackedDataLoader:
    """包装 DataLoader，记录每次 __iter__（即一轮校准 forward_loop）。"""

    def __init__(self, data_loader, tracker: CalibIterTracker):
        self._data_loader = data_loader
        self._tracker = tracker

    def __iter__(self):
        try:
            loader_batches = len(self._data_loader)
        except Exception:
            loader_batches = None
        round_id = self._tracker.start_round(loader_batches=loader_batches)
        return _TrackedDataLoaderIter(iter(self._data_loader), self._tracker, round_id)

    def __len__(self):
        return len(self._data_loader)

    def __getattr__(self, name):
        return getattr(self._data_loader, name)


_calib_iter_tracker = CalibIterTracker()


# --------------------------- 只剪 backbone/neck：构造 fastnas_mode_cfg ---------------------------
def _is_backbone_neck(name: str) -> bool:
    return (".img_backbone." in name) or (".img_neck." in name)

def assert_heads_not_searchable(model_after_convert: nn.Module):
    names = [n for n, _ in named_hparams(model_after_convert, configurable=True)]
    bad = [n for n in names if any(x in n for x in ("occ_head", "seg_head", "motion_head", "planning_head"))]
    print(f"[CHECK] configurable hparams = {len(names)}")
    print(f"[CHECK] head configurable hparams = {len(bad)}")
    if bad:
        print("[CHECK] bad examples:", bad[:30])
        raise RuntimeError("Heads entered FastNAS searchable hparams unexpectedly!")


# --------------------------- main ---------------------------
def main():
    log_path = setup_logging()
    print(f"[INFO] 本次运行日志保存至: {log_path}")

    args = parse_args()
    if args.launcher != "none":
        print(f"[WARN] 当前脚本评分评估固定使用单卡模式，忽略 --launcher={args.launcher}")

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib
        plugin_dir = getattr(cfg, "plugin_dir", os.path.dirname(args.config))
        _module_path = ".".join(os.path.dirname(plugin_dir).split("/"))
        print(f"[INFO] 加载插件模块: {_module_path}")
        importlib.import_module(_module_path)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # ✅ 解冻 backbone/neck（扩大搜索空间）；保持 heads 结构完整，不再置 None
    cfg.model_ego_agent.pretrained = None
    cfg.model_ego_agent.freeze_img_backbone = False
    cfg.model_ego_agent.freeze_img_neck = False
    cfg.model_ego_agent.img_backbone.frozen_stages = 0
    for name in [k for k in cfg.keys() if "model_other_agent" in k]:
        cfg.get(name).freeze_img_backbone = False
        cfg.get(name).freeze_img_neck = False
        cfg.get(name).img_backbone.frozen_stages = 0

    print("[INFO] 正在构建 MultiAgent 模型并加载权重...")
    model = build_multi_agent_model(cfg, args.checkpoint).to(device).eval()

    total_params_before = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 剪枝前参数量: {total_params_before / 1e6:.2f} M")

    wrapped_model = ModelOptWrapper(
        model,
        trace_prev_bev_mode=args.trace_prev_bev_mode,
    ).to(device).eval()
    print(f"[INFO] trace prev_bev 模式: {args.trace_prev_bev_mode}")

    # ================================================================
    # 校准集（BN 统计量更新）：训练集子集
    #   BN running_mean/running_var 应反映训练分布，用训练集子集更新
    # ================================================================
    print(f"[INFO] 构建 BN 校准集（split={args.calib_split}，"
          f"mode={args.calib_subset_mode}，{args.calib_frames_per_scene} 帧/场景）...")
    calib_dataset, calib_loader = build_subset_loader(
        cfg,
        split=args.calib_split,
        subset_mode=args.calib_subset_mode,
        frames_per_scene=args.calib_frames_per_scene,
        dist_mode=False,
    )
    print(f"[INFO] 校准集 size = {len(calib_dataset)}")

    # FastNAS 默认 max_iter_data_loader=50；这里允许显式放大或取消截断。
    max_iter_data_loader = args.max_iter_data_loader if args.max_iter_data_loader > 0 else None
    if max_iter_data_loader is None:
        print("[INFO] max_iter_data_loader = None（每轮 BN 校准遍历完整 calib_loader）")
    else:
        print(f"[INFO] max_iter_data_loader = {max_iter_data_loader} batches/轮")

    # 包装 calib_loader，追踪每轮 BN 校准的 batch 消耗与调用来源
    _calib_iter_tracker.reset()
    tracked_calib_loader = TrackedDataLoader(calib_loader, _calib_iter_tracker)

    # ================================================================
    # 评分集（候选子网排序）：val 子集（默认 head, 10 帧/场景）
    #   评估泛化性能，覆盖多个场景，避免搜索过拟合
    # ================================================================
    print(f"[INFO] 构建评分集（split={args.score_split}，"
          f"mode={args.score_subset_mode}，{args.score_frames_per_scene} 帧/场景）...")
    val_dataset, val_loader = build_subset_loader(
        cfg,
        split=args.score_split,
        subset_mode=args.score_subset_mode,
        frames_per_scene=args.score_frames_per_scene,
        dist_mode=False,
    )
    # eval_mod 只影响 score_func 中的 evaluate()
    val_dataset.eval_mod = args.eval_modes
    if hasattr(val_dataset, '_dataset'):
        val_dataset._dataset.eval_mod = args.eval_modes
    print(f"[INFO] 评分集 size = {len(val_dataset)}，eval_mod = {args.eval_modes}")

    eval_kwargs = cfg.get("evaluation", {}).copy()
    for drop_key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
        eval_kwargs.pop(drop_key, None)
    eval_kwargs["metric"] = ["bbox"]
    eval_kwargs["jsonfile_prefix"] = osp.join(
        "test_prune",
        args.config.split("/")[-1].rsplit(".", 1)[0],
        time.ctime().replace(" ", "_").replace(":", "_"),
    )

    score_func = build_score_func(
        val_dataset,
        val_loader,
        eval_kwargs,
        gpu_id=args.gpu_id,
    )

    # ✅ 辅助：从 batch 中构建统一输入结构 + 提取全部 tensor
    def _batch_to_full_input(batch_on_dev):
        """将 scatter 后的 batch 拆分为 {ego, others, img_metas} 三部分。"""
        ego = dict(batch_on_dev["ego_agent_data"])
        others_raw = batch_on_dev.get("other_agent_data_dict", {})
        others = {k: dict(v) for k, v in others_raw.items()}
        img_metas = batch_on_dev.get("img_metas", ego.get("img_metas", None))
        # img_metas 只保留顶层，子 dict 内的删掉（避免重复）
        ego.pop("img_metas", None)
        for _, d in others.items():
            d.pop("img_metas", None)
        return {'ego': ego, 'others': others, 'img_metas': img_metas}

    # ✅ dummy_input：从 calib_loader 取一个 batch，提取全部 tensor
    print("[INFO] 构造 dummy_input...")
    raw_batch = next(iter(calib_loader))
    batch_on_device = scatter(raw_batch, [args.gpu_id])[0]

    full_input = _batch_to_full_input(batch_on_device)
    skeleton, tensor_list = _extract_tensors(full_input)
    wrapped_model.set_skeleton(skeleton)
    dummy_input = tuple(t.to(device) for t in tensor_list)

    # -------- 打印调试信息 --------
    def _inspect(name, obj, depth=0, max_depth=3):
        prefix = "  " * depth
        if isinstance(obj, _TensorSlot):
            t = tensor_list[obj.idx]
            print(f"{prefix}{name}: -> tensor_list[{obj.idx}] shape={t.shape} dtype={t.dtype}")
        elif isinstance(obj, torch.Tensor):
            print(f"{prefix}{name}: Tensor shape={obj.shape} dtype={obj.dtype} device={obj.device}")
        elif isinstance(obj, np.ndarray):
            print(f"{prefix}{name}: ndarray shape={obj.shape} dtype={obj.dtype}")
        elif isinstance(obj, dict):
            print(f"{prefix}{name}: dict keys={list(obj.keys())}")
            if depth < max_depth:
                for k, v in obj.items():
                    _inspect(f"{name}['{k}']", v, depth+1, max_depth)
        elif isinstance(obj, (list, tuple)):
            print(f"{prefix}{name}: {type(obj).__name__} len={len(obj)}")
            if depth < max_depth and len(obj) > 0:
                _inspect(f"{name}[0]", obj[0], depth+1, max_depth)
                if len(obj) > 1:
                    _inspect(f"{name}[-1]", obj[-1], depth+1, max_depth)
        else:
            print(f"{prefix}{name}: {type(obj).__name__} = {repr(obj)[:120]}")

    print("=" * 60)
    print(f">>> 提取到 {len(tensor_list)} 个 tensor，骨架保留非 tensor 元数据")
    _inspect("skeleton", skeleton)
    print("=" * 60)
    # -----------------------------------------------------------------------------------

    del raw_batch, batch_on_device
    torch.cuda.empty_cache()
    print(f"[INFO] dummy_input ok, {len(dummy_input)} tensors, "
          f"allocated={torch.cuda.memory_allocated()/1e9:.2f} GiB")

    # ✅ collect_func：每个 batch 提取全部 tensor，更新骨架
    def collect_func(raw_batch):
        _calib_iter_tracker.mark_collect_call()
        batch = scatter(raw_batch, [args.gpu_id])[0]
        fi = _batch_to_full_input(batch)
        skel, tlist = _extract_tensors(fi)
        wrapped_model.set_skeleton(skel)
        return tuple(t.to(device) for t in tlist)

    prune_constraints = {"params": args.params_percent}
    print(f"[INFO] 剪枝约束: {prune_constraints}")

    # ---- 带时间戳的输出路径 ----
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)

    out_model_path = osp.join(
        args.out_dir, f"all_model_prune_bb_neck_pruned_model_{run_ts}.pth")
    # 搜索 checkpoint：未指定则自动生成新路径（不会加载旧状态）
    if args.search_checkpoint is not None:
        search_ckpt_path = args.search_checkpoint
        print(f"[INFO] 断点续搜，加载: {search_ckpt_path}")
    else:
        search_ckpt_path = osp.join(
            args.out_dir, f"all_model_prune_bb_neck_fastnas_search_checkpoint_{run_ts}.pth")
        print(f"[INFO] 全新搜索，checkpoint 将保存到: {search_ckpt_path}")


    # ---- Monkey-patch: fix Named Tensor + TorchScript tracer incompatibility ----
    import copy as _copy_module
    import torch as _torch

    _original_deepcopy = _copy_module.deepcopy

    
    def _tracer_safe_deepcopy(obj, memo=None):
        if isinstance(obj, _torch.Tensor) and _torch.jit.is_tracing():
            if obj.names and any(n is not None for n in obj.names):
                return obj.rename(None).clone()
            return obj.clone()

        if isinstance(obj, _torch.storage.UntypedStorage) and _torch.jit.is_tracing():
            return obj

        return _original_deepcopy(obj, memo)

    _copy_module.deepcopy = _tracer_safe_deepcopy
    print("[PATCH] copy.deepcopy patched: Named Tensor will be rename(None).clone() under tracer")
    # ---- end patch ----

    print("[INFO] 开始 FastNAS 剪枝搜索（完整网络结构，only backbone/neck searchable）...")
    fastnas_cfg = {
        # ---- 只对 backbone/neck 的 Conv2d 和 BN 启用搜索 ----
        "nn.Conv2d": {
            "*_inner.model_ego_agent.img_backbone*": {"channel_divisor": 16},
            "*_inner.model_ego_agent.img_neck*": {"channel_divisor": 16},
            "*_inner.model_other_agent_inf.img_backbone*": {"channel_divisor": 16},
            "*_inner.model_other_agent_inf.img_neck*": {"channel_divisor": 16},
        },
        "nn.BatchNorm2d": {
            "*_inner.model_ego_agent.img_backbone*": {"feature_divisor": 16},
            "*_inner.model_ego_agent.img_neck*": {"feature_divisor": 16},
            "*_inner.model_other_agent_inf.img_backbone*": {"feature_divisor": 16},
            "*_inner.model_other_agent_inf.img_neck*": {"feature_divisor": 16},
        },
        # ---- 显式禁用所有其他默认模块类型（防止 heads 被搜索） ----
        # ModelOpt 默认对以下 16 种类型启用搜索规则 {"*": {...}}，
        # 未在 config 中显式覆盖的类型会保留默认规则，导致 heads 中的
        # Linear/LayerNorm 等层被意外搜索。设为 None 使其全部 freeze。
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
    }
    pruned_model, prune_res = mtp.prune(
        model=wrapped_model,
        mode=[("fastnas", fastnas_cfg)],
        constraints=prune_constraints,
        dummy_input=dummy_input,
        config={
            "data_loader": tracked_calib_loader, # 训练集子集（BN 校准）
            "score_func": score_func,           # val 子集 210 帧（候选网络排序）
            "checkpoint": search_ckpt_path,
            "collect_func": collect_func,
            "max_iter_data_loader": max_iter_data_loader,
        },
    )

    # ---- 恢复原始 deepcopy ----
    _copy_module.deepcopy = _original_deepcopy
    print("[PATCH] copy.deepcopy restored")
    # --------------------------
    # ✅ 验收：heads 不应进入 searchable hparams
    assert_heads_not_searchable(pruned_model)

    inner = pruned_model._inner if hasattr(pruned_model, "_inner") else pruned_model
    total_params_after = sum(p.numel() for p in inner.parameters())
    actual_ratio = total_params_after / total_params_before * 100
    print(f"[INFO] 剪枝后参数量: {total_params_after / 1e6:.2f} M")
    print(f"[INFO] 实际保留比例: {actual_ratio:.2f}%（目标: {args.params_percent}）")

    print(f"[INFO] 正在保存剪枝后的模型到: {out_model_path}")
    mto.save(pruned_model, out_model_path)

    stats_path = out_model_path.replace(".pth", "_prune_stats.pkl")
    mmcv.dump(prune_res, stats_path)
    print(f"[INFO] 搜索统计信息保存至: {stats_path}")
    _calib_iter_tracker.print_report(configured_max_iters=max_iter_data_loader)

    print("\n" + "=" * 60)
    print("剪枝完成！后续步骤：")
    print("  1. 微调/推理时，build 完整模型结构（heads 全在），再 mto.restore：")
    print(f"       wrapped = mto.restore(ModelOptWrapper(full_model), '{out_model_path}')")
    print("  2. wrapped._inner 即为剪枝后的完整 MultiAgent（仅 bb/neck 被剪）")
    print("=" * 60)


if __name__ == "__main__":
    main()
