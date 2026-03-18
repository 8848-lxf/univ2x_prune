"""
verify_subset_eval.py
---------------------
用与剪枝 score_func 完全相同的 val 子集（每场景前 10 帧）
评估原始未剪枝模型，验证子集评估流程是否正确。

如果原始模型在该子集上 car AP 也接近 0，说明评估流程有 bug；
如果 car AP 正常（>0.1），说明评估逻辑没问题，问题在剪枝后模型本身。

用法：
    cd /home/lixingfeng/UniAD_exmaine/UniV2X
    python ./univ2x_purned/verify_subset_eval.py \
        --gpu-id 7
"""

import os
import sys
import math
import time
import json
import warnings
from collections import defaultdict
from typing import Dict, Any

# ---- Ensure UniV2X repo root is on sys.path ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent

warnings.filterwarnings("ignore")


# ---- 从剪枝脚本中复制的子集采样 + SubsetDataset 逻辑 ----

def fixed_head_indices(dataset, frames_per_scene=10, sampling_mode="skip_warmup"):
    """每个 scene 连续采样固定帧数（时间顺序）。"""
    scene_to_indices = defaultdict(list)
    for i, info in enumerate(dataset.data_infos):
        scene_to_indices[info['scene_token']].append(i)

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
        scene_counts[dataset.data_infos[i]['scene_token']] += 1

    print(f"[INFO] 连续采样（mode={sampling_mode}, start_idx={start_idx}）："
          f"覆盖 {len(scene_counts)} / {len(scenes)} 个场景，总帧数={len(selected)}")
    print(f"       每场景帧数：min={min(scene_counts.values())} "
          f"max={max(scene_counts.values())} "
          f"avg={np.mean(list(scene_counts.values())):.1f}")
    return selected


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


def _preview_tokens(tokens, limit=6):
    items = sorted(tokens)
    if len(items) <= limit * 2:
        return items
    return items[:limit] + ["..."] + items[-limit:]


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

    # 若模型输出不带 token，不做强制失败；后续由 results_nusc_det.json 再做集合对齐诊断。
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


def diagnose_eval_alignment(eval_dataset, jsonfile_prefix, use_full_val=False):
    """诊断 sample_token 对齐与 det AP 分母来源（npos）。"""
    print("\n" + "=" * 70)
    print("[DIAG] 评估对齐诊断")
    print("=" * 70)

    dataset_tokens = [info["token"] for info in eval_dataset.data_infos]
    dataset_token_set = set(dataset_tokens)
    dup_count = len(dataset_tokens) - len(dataset_token_set)
    print(f"[DIAG] dataset tokens: total={len(dataset_tokens)}, unique={len(dataset_token_set)}, duplicates={dup_count}")

    result_det_path = os.path.join(jsonfile_prefix, "results_nusc_det.json")
    if not os.path.exists(result_det_path):
        print(f"[DIAG][WARN] 未找到结果文件: {result_det_path}")
        print("=" * 70)
        return

    with open(result_det_path, "r", encoding="utf-8") as f:
        result_det = json.load(f)
    pred_tokens = set(result_det.get("results", {}).keys())
    print(f"[DIAG] prediction tokens: {len(pred_tokens)}")

    ds_minus_pred = dataset_token_set - pred_tokens
    pred_minus_ds = pred_tokens - dataset_token_set
    print(f"[DIAG] dataset - pred = {len(ds_minus_pred)}")
    print(f"[DIAG] pred - dataset = {len(pred_minus_ds)}")
    if ds_minus_pred:
        print(f"[DIAG] dataset-only token preview: {_preview_tokens(ds_minus_pred)}")
    if pred_minus_ds:
        print(f"[DIAG] pred-only token preview: {_preview_tokens(pred_minus_ds)}")

    base_dataset = getattr(eval_dataset, "_dataset", eval_dataset)

    try:
        from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval import NuScenesEval_custom

        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        eval_set = eval_set_map.get(getattr(base_dataset, "version", "v1.0-trainval"), "val")

        # 优先复用 evaluate() 中已经构建好的 evaluator，确保和真实评估完全同路径。
        nusc_eval = getattr(base_dataset, "nusc_eval", None)
        if nusc_eval is not None and hasattr(nusc_eval, "gt_boxes") and hasattr(nusc_eval, "pred_boxes"):
            print("[DIAG] 复用 eval_dataset.nusc_eval（真实评估路径）")
        else:
            print("[DIAG] 未找到 eval_dataset.nusc_eval，回退为手动构建 det evaluator")
            if hasattr(base_dataset, "create_splits_spd") and callable(base_dataset.create_splits_spd):
                splits = base_dataset.create_splits_spd()
            else:
                from nuscenes.utils.splits import create_splits_scenes
                splits = create_splits_scenes()

            if hasattr(base_dataset, "category_to_detection_name") and callable(base_dataset.category_to_detection_name):
                category_to_type_name = base_dataset.category_to_detection_name
            else:
                from nuscenes.eval.detection.utils import category_to_detection_name
                category_to_type_name = category_to_detection_name

            nusc_eval = NuScenesEval_custom(
                base_dataset.nusc,
                config=base_dataset.eval_detection_configs,
                result_path=result_det_path,
                eval_set=eval_set,
                output_dir=os.path.join(jsonfile_prefix, "diag_det"),
                verbose=False,
                overlap_test=getattr(base_dataset, "overlap_test", False),
                data_infos=eval_dataset.data_infos,
                splits=splits,
                category_to_type_name=category_to_type_name,
                class_range=getattr(base_dataset, "class_range", None),
            )

        gt_tokens_eval = set(nusc_eval.gt_boxes.sample_tokens)
        pred_tokens_eval = set(nusc_eval.pred_boxes.sample_tokens)
        print(f"[DIAG] evaluator GT tokens: {len(gt_tokens_eval)}")
        print(f"[DIAG] evaluator PRED tokens: {len(pred_tokens_eval)}")

        gt_minus_pred = gt_tokens_eval - pred_tokens_eval
        pred_minus_gt = pred_tokens_eval - gt_tokens_eval
        print(f"[DIAG] evaluator GT - PRED = {len(gt_minus_pred)}")
        print(f"[DIAG] evaluator PRED - GT = {len(pred_minus_gt)}")
        if gt_minus_pred:
            print(f"[DIAG] evaluator GT-only token preview: {_preview_tokens(gt_minus_pred)}")
        if pred_minus_gt:
            print(f"[DIAG] evaluator PRED-only token preview: {_preview_tokens(pred_minus_gt)}")

        # npos: det AP 的正样本分母（按类别统计）。
        npos = {name: 0 for name in base_dataset.CLASSES}
        for box in nusc_eval.gt_boxes.all:
            if box.detection_name in npos:
                npos[box.detection_name] += 1
        print("[DIAG] det AP npos by class:")
        for name in base_dataset.CLASSES:
            print(f"  - {name}: {npos[name]}")
        print(f"[DIAG] total GT boxes used for det AP: {len(nusc_eval.gt_boxes.all)}")
        print("[DIAG] 结论：det AP 的分母来自上面的 npos，不是全量样本数。")
    except Exception as exc:
        print(f"[DIAG][WARN] 诊断过程中出现异常: {exc}")

    print("=" * 70)


# ---- NuScenesEval subset patch (same as pruning script) ----

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

    # ---- Tracking patch ----
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
            nusc = NuScenes(version=nusc_version, verbose=verbose, dataroot=nusc_dataroot)
            pred_boxes, self.meta = load_prediction(
                self.result_path, self.cfg.max_boxes_per_sample, TrackingBox, verbose=verbose)
            gt_boxes = _local_load_gt(
                nusc, self.eval_set, TrackingBox, verbose=verbose,
                splits=self.splits, category_to_type_name=self.category_to_type_name)

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


patch_nuscenes_eval_for_subset()


# ---- 模型构建 ----

def build_multi_agent_model(cfg, checkpoint_path):
    other_agent_names = [key for key in cfg.keys() if "model_other_agent" in key]
    model_other_agents = {}

    for name in other_agent_names:
        cfg.get(name).train_cfg = None
        model = build_model(cfg.get(name), test_cfg=cfg.get("test_cfg"))
        load_from = cfg.get(name).load_from
        if load_from:
            load_checkpoint(model, load_from, map_location="cpu",
                            revise_keys=[(r"^model_ego_agent\.", "")])
        model_other_agents[name] = model

    cfg.model_ego_agent.train_cfg = None
    model_ego_agent = build_model(cfg.model_ego_agent, test_cfg=cfg.get("test_cfg"))
    load_from = cfg.model_ego_agent.load_from
    if load_from:
        load_checkpoint(model_ego_agent, load_from, map_location="cpu",
                        revise_keys=[(r"^model_ego_agent\.", "")])

    model_multi_agents = MultiAgent(model_ego_agent, model_other_agents)
    checkpoint = load_checkpoint(model_multi_agents, checkpoint_path, map_location="cpu")

    if "CLASSES" in checkpoint.get("meta", {}):
        model_multi_agents.model_ego_agent.CLASSES = checkpoint["meta"]["CLASSES"]
    if "PALETTE" in checkpoint.get("meta", {}):
        model_multi_agents.model_ego_agent.PALETTE = checkpoint["meta"]["PALETTE"]

    return model_multi_agents


# ---- main ----

def main():
    parser = argparse.ArgumentParser(
        description="用 val 全量/子集评估原始模型，验证评估流程")
    parser.add_argument("--config", default="./projects/configs_e2e_univ2x/univ2x_coop_e2e.py")
    parser.add_argument("--checkpoint", default="./ckpts/univ2x_coop_e2e_stg2.pth")
    parser.add_argument("--gpu-id", type=int, default=7)
    parser.add_argument("--frames-per-scene", type=int, default=10)
    parser.add_argument(
        "--subset-sampling-mode",
        choices=["head", "skip_warmup"],
        default="skip_warmup",
        help="子集连续采样模式：head=从第1帧取，skip_warmup=从第5帧取。",
    )
    parser.add_argument(
        "--use-full-val",
        action="store_true",
        help="启用后不切分子集，直接在完整 val 集评估。",
    )
    parser.add_argument(
        "--diagnose-eval-alignment",
        action="store_true",
        help="评估后打印 token 对齐与 det AP 分母（npos）诊断信息。",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib
        plugin_dir = getattr(cfg, "plugin_dir", os.path.dirname(args.config))
        _module_path = ".".join(os.path.dirname(plugin_dir).split("/"))
        print(f"[INFO] 加载插件模块: {_module_path}")
        importlib.import_module(_module_path)

    set_random_seed(args.seed, deterministic=True)
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")

    print(f"[INFO] 使用设备: {device}")
    print(f"[INFO] 配置文件: {args.config}")
    print(f"[INFO] 原始模型: {args.checkpoint}")

    # ---- 构建模型 ----
    print("[INFO] 正在构建 MultiAgent 模型...")
    model = build_multi_agent_model(cfg, args.checkpoint).to(device).eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 模型参数量: {total_params / 1e6:.2f} M")

    # ---- 构建 val 数据集 ----
    if args.use_full_val:
        print("[INFO] 使用完整 val 集评估（不切分子集）...")
    else:
        print(f"[INFO] 构建 val 子集（per_scene, {args.frames_per_scene} 帧/场景, "
              f"mode={args.subset_sampling_mode}）...")

    # 使用 cfg.data.test 配置来构建 val 数据集（与评估对齐）
    test_cfg = cfg.data.test.copy()
    test_cfg.test_mode = True
    samples_per_gpu = test_cfg.pop("samples_per_gpu", 1)
    full_dataset = build_dataset(test_cfg)
    print(f"[INFO] 完整 val 集大小: {len(full_dataset)}")

    if args.use_full_val:
        eval_dataset = full_dataset
        print(f"[INFO] 使用完整 val，样本数: {len(eval_dataset)}")
    else:
        indices = fixed_head_indices(
            full_dataset,
            frames_per_scene=args.frames_per_scene,
            sampling_mode=args.subset_sampling_mode,
        )
        eval_dataset = SubsetDataset(full_dataset, indices)
        print(f"[INFO] 子集大小: {len(eval_dataset)}")

    eval_dataset.eval_mod = ["det"]
    if hasattr(eval_dataset, '_dataset'):
        eval_dataset._dataset.eval_mod = ["det"]

    loader = build_dataloader(
        eval_dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.get("nonshuffler_sampler", None),
    )

    # ---- 推理 ----
    print("[INFO] 开始推理...")
    model_dp = MMDataParallel(model, device_ids=[args.gpu_id])
    model_dp.eval()
    outputs = single_gpu_test(model_dp, loader, show=False)
    print(f"[INFO] 推理完成，得到 {len(outputs)} 个样本结果")
    assert_output_token_alignment(eval_dataset, outputs)

    # ---- ret_iou 补丁（与剪枝 score_func 一致） ----
    for i in range(len(outputs)):
        det = outputs[i]
        if isinstance(det, dict):
            det.setdefault("ret_iou", {})
        elif isinstance(det, list) and len(det) > 0 and isinstance(det[0], dict):
            det[0].setdefault("ret_iou", {})

    # ---- 评估 ----
    print("[INFO] 开始评估...")
    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
        eval_kwargs.pop(key, None)
    eval_kwargs["metric"] = ["bbox"]
    prefix_root = "test_verify_fullval" if args.use_full_val else "test_verify_subset"
    eval_kwargs["jsonfile_prefix"] = os.path.join(
        prefix_root,
        time.ctime().replace(" ", "_").replace(":", "_"),
    )

    res = eval_dataset.evaluate(outputs, **eval_kwargs)
    if args.diagnose_eval_alignment:
        diagnose_eval_alignment(
            eval_dataset=eval_dataset,
            jsonfile_prefix=eval_kwargs["jsonfile_prefix"],
            use_full_val=args.use_full_val,
        )

    # ---- 输出结果 ----
    print("\n" + "=" * 70)
    print("子集评估结果（原始未剪枝模型）")
    print("=" * 70)

    # car AP
    car_keys = [
        "pts_bbox_NuScenes/car_AP_dist_0.5",
        "pts_bbox_NuScenes/car_AP_dist_1.0",
        "pts_bbox_NuScenes/car_AP_dist_2.0",
        "pts_bbox_NuScenes/car_AP_dist_4.0",
    ]
    car_vals = [float(res[k]) for k in car_keys if k in res and not math.isnan(float(res[k]))]
    if car_vals:
        mean_ap = sum(car_vals) / len(car_vals)
        print(f"car mean AP = {mean_ap:.4f}  (APs: {[f'{v:.4f}' for v in car_vals]})")
    else:
        print("car AP: 全部 NaN 或缺失！")

    # 所有类别 AP
    for k, v in sorted(res.items()):
        if "AP" in k or "NDS" in k or "mAP" in k:
            print(f"  {k}: {float(v):.4f}")

    print("=" * 70)

    if car_vals and mean_ap > 0.05:
        print("\n[CONCLUSION] 原始模型在 val 子集上 car AP 正常。")
        print("             子集评估流程没有问题，问题出在剪枝后的模型。")
    elif car_vals and mean_ap <= 0.05:
        print("\n[CONCLUSION] 原始模型在 val 子集上 car AP 也很低！")
        print("             说明子集评估流程可能存在 GT 匹配问题，需要排查。")
    else:
        print("\n[CONCLUSION] 无法计算 car AP，评估流程存在严重问题。")


if __name__ == "__main__":
    main()
