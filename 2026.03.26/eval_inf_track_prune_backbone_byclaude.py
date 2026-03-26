"""
eval_inf_track_prune_backbone_byclaude.py
-------------------------------------
对剪枝产出的模型进行恢复（mto.restore）并在完整 val 数据集上运行
全任务评估（det / track / map 等）。

评估逻辑对齐原始 tools/test.py，支持 eval_mod 控制任务头。

恢复流程：
  1. 用原始 checkpoint 构建完整模型（neck/head/tracking 等拥有正确权重）
  2. 用 BackbonePruneWrapper 包装 backbone，mto.restore 恢复剪枝结构+权重
  3. 将剪枝 backbone 回填到 full_model 中
  4. 在完整 val 集上推理和评估

用法（全量 val 全任务评估）:
    cd /home/lixingfeng/UniAD_exmaine/UniV2X
    python ./univ2x_purned/eval_inf_track_prune_backbone_byclaude.py \
        --gpu-id 2 \
        --eval-modes det track map

用法（skip_warmup 子集评估）:
    python ./univ2x_purned/eval_inf_track_prune_backbone_byclaude.py \
        --gpu-id 2 \
        --subset-mode skip_warmup --frames-per-scene 10 \
        --eval-modes det track map
"""

import os
import sys

# ---- Ensure UniV2X repo root is on sys.path ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import os.path as osp
import time
import warnings

import torch
import torch.nn as nn
import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed

import numpy as np
import copy as _cp
import modelopt.torch.opt as mto
from collections import defaultdict

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent

warnings.filterwarnings("ignore")

# 默认剪枝 backbone modelopt 路径（由 mtp.prune 产出，包含正确的 modelopt 元数据）
DEFAULT_BACKBONE_MODELOPT = (
    "./univ2x_purned/modelopt/univ2x_inf_track_prune_backbone_byclaude/out/"
    "20260326_150910_452188/"
    "univ2x_inf_track_backbone_pruned_20260326_150910_452188_backbone_modelopt.pth"
)
# 原始完整模型 checkpoint（用于恢复非 backbone 部分的权重）
DEFAULT_ORIGINAL_CHECKPOINT = "./ckpts/univ2x_sub_inf_stg1.pth"


# ====================== 日志 ======================
LOG_DIR = "/home/lixingfeng/UniAD_exmaine/UniV2X/univ2x_purned/modelopt/logs_eval_inf_track_prune_backbone_byclaude.py"


class _Tee:
    """Python 层 tee：终端原样输出，日志文件完整记录全部输出。"""
    def __init__(self, stream, log_fh):
        self._stream = stream
        self._log_fh = log_fh

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        if data:
            # tqdm/mmcv 进度条通常用 '\r' 刷新同一行，直接写文件会“看起来像没记录”。
            # 将 '\r' 规范成换行，便于在日志中保留完整进度输出轨迹。
            log_data = data.replace("\r\n", "\n").replace("\r", "\n")
            self._log_fh.write(log_data)
            self._log_fh.flush()

    def flush(self):
        self._stream.flush()
        self._log_fh.flush()

    def __getattr__(self, attr):
        return getattr(self._stream, attr)


def setup_logging() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"eval_{timestamp}.log")

    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    return log_path


# ====================== BackbonePruneWrapper ======================
# 必须与剪枝脚本中 mto.save(pruned_backbone_wrapper, ...) 保存时
# 使用的 wrapper 结构完全一致。剪枝脚本中只对 backbone 做了包装和
# mtp.prune，所以 backbone_modelopt.pth 的 key 前缀是 "backbone.*"。
# ==================================================================
class BackbonePruneWrapper(nn.Module):
    """与剪枝脚本中的 BackbonePruneWrapper 完全一致。"""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, img: torch.Tensor):
        return self.backbone(img)


def fixed_head_indices(dataset, frames_per_scene=10, sampling_mode="skip_warmup"):
    """Scene-wise contiguous head sampling (from pruning script)."""
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
    """Subset wrapper preserving dataset evaluate/format behavior (from pruning script)."""

    _PASSTHROUGH = [
        "CLASSES", "eval_mod", "eval_detection_configs", "eval_version",
        "modality", "overlap_test", "version", "data_root", "nusc",
        "nusc_maps", "ErrNameMapping", "with_velocity", "split_datas_file",
        "class_range", "tmp_dataset_type", "planning_steps",
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


def patch_nuscenes_eval_for_subset():
    """Patch evaluator to handle token mismatch (safety net for full-val too)."""
    from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval import NuScenesEval_custom
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes
    from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval import (
        filter_eval_boxes_by_overlap,
    )

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
    print("[PATCH] NuScenesEval_custom patched")

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
        self, config, result_path, eval_set, output_dir,
        nusc_version, nusc_dataroot, verbose=True, render_classes=None,
        splits={}, category_to_type_name=None, class_range=None,
    ):
        try:
            _orig_track_init(
                self, config, result_path, eval_set, output_dir,
                nusc_version, nusc_dataroot, verbose=verbose,
                render_classes=render_classes, splits=splits,
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
                nusc, self.eval_set, TrackingBox, verbose=verbose,
                splits=self.splits, category_to_type_name=self.category_to_type_name,
            )
            pred_tokens = set(pred_boxes.sample_tokens)
            filtered_gt = EvalBoxes()
            for token in pred_tokens:
                boxes = gt_boxes.boxes.get(token, [])
                filtered_gt.add_boxes(token, boxes)
            gt_boxes = filtered_gt
            print(f"[PATCH] TrackingEval subset: GT filtered to {len(pred_tokens)} pred tokens")
            pred_boxes = _add_center_dist(nusc, pred_boxes)
            gt_boxes = _add_center_dist(nusc, gt_boxes)
            pred_boxes = _filter_eval_boxes(nusc, pred_boxes, self.class_range, verbose=verbose)
            gt_boxes = _filter_eval_boxes(nusc, gt_boxes, self.class_range, verbose=verbose)
            self.sample_tokens = gt_boxes.sample_tokens
            self.tracks_gt = create_tracks(gt_boxes, nusc, self.eval_set, gt=True)
            self.tracks_pred = create_tracks(pred_boxes, nusc, self.eval_set, gt=False)

    TrackingEval_custom.__init__ = _subset_safe_track_init
    print("[PATCH] TrackingEval_custom patched")


def reset_temporal_states(module: nn.Module) -> None:
    """Reset temporal/cache states to match pruning-time evaluation behavior."""
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


def _count_boxes_recursive(obj) -> int:
    """Count predicted boxes from nested prediction objects."""
    if isinstance(obj, dict):
        if "boxes_3d" in obj:
            boxes = obj["boxes_3d"]
            if hasattr(boxes, "tensor"):
                return int(boxes.tensor.shape[0])
            if hasattr(boxes, "shape") and len(getattr(boxes, "shape")) > 0:
                return int(boxes.shape[0])
        return sum(_count_boxes_recursive(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_count_boxes_recursive(v) for v in obj)
    return 0


def summarize_outputs(outputs) -> None:
    """Print lightweight sanity checks for inference outputs."""
    total = len(outputs)
    none_cnt = sum(o is None for o in outputs)
    empty_cnt = sum(
        (isinstance(o, dict) and len(o) == 0) or (isinstance(o, list) and len(o) == 0)
        for o in outputs
    )
    box_counts = [_count_boxes_recursive(o) for o in outputs]
    nonzero_box_samples = sum(c > 0 for c in box_counts)
    total_boxes = int(sum(box_counts))
    print(
        "[CHECK] outputs summary: "
        f"total={total}, none={none_cnt}, empty={empty_cnt}, "
        f"nonzero_box_samples={nonzero_box_samples}, total_boxes={total_boxes}"
    )
    first_non_none = next((o for o in outputs if o is not None), None)
    if isinstance(first_non_none, dict):
        print(f"[CHECK] first non-empty output keys: {list(first_non_none.keys())}")


# ====================== 模型构建 ======================
def build_multi_agent_model(cfg, checkpoint_path: str):
    """构建 MultiAgent 模型并加载原始完整 checkpoint。

    这样非 backbone 部分（neck/head/tracking 等）拥有正确的预训练权重，
    backbone 部分之后通过 mto.restore 替换为剪枝版本。
    """
    other_agent_names = [key for key in cfg.keys() if "model_other_agent" in key]
    model_other_agents = {}

    for name in other_agent_names:
        cfg.get(name).train_cfg = None
        model = build_model(cfg.get(name), test_cfg=cfg.get("test_cfg"))
        load_from = cfg.get(name).get("load_from", None)
        if load_from:
            load_checkpoint(
                model, load_from, map_location="cpu",
                revise_keys=[(r"^model_ego_agent\.", "")],
            )
        model_other_agents[name] = model

    cfg.model_ego_agent.train_cfg = None
    model_ego_agent = build_model(cfg.model_ego_agent, test_cfg=cfg.get("test_cfg"))
    load_from = cfg.model_ego_agent.get("load_from", None)
    if load_from:
        load_checkpoint(
            model_ego_agent, load_from, map_location="cpu",
            revise_keys=[(r"^model_ego_agent\.", "")],
        )

    model_multi_agents = MultiAgent(model_ego_agent, model_other_agents)
    ckpt = load_checkpoint(model_multi_agents, checkpoint_path, map_location="cpu")
    if "CLASSES" in ckpt.get("meta", {}):
        model_multi_agents.model_ego_agent.CLASSES = ckpt["meta"]["CLASSES"]
    if "PALETTE" in ckpt.get("meta", {}):
        model_multi_agents.model_ego_agent.PALETTE = ckpt["meta"]["PALETTE"]
    return model_multi_agents


# ====================== 参数解析 ======================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate pruned MultiAgent model on full val set (all tasks)"
    )
    parser.add_argument(
        "--config",
        default="./projects/configs_e2e_univ2x/univ2x_sub_inf_e2e_track.py",
        help="配置文件路径",
    )
    parser.add_argument(
        "--original-checkpoint",
        default=DEFAULT_ORIGINAL_CHECKPOINT,
        help="原始完整模型 checkpoint（用于恢复非 backbone 部分权重）",
    )
    parser.add_argument(
        "--backbone-modelopt",
        default=DEFAULT_BACKBONE_MODELOPT,
        help="mto.save 保存的剪枝 backbone modelopt 路径（backbone_modelopt.pth）",
    )
    parser.add_argument(
        "--finetune-ckpt",
        type=str,
        default=None,
        help=(
            "可选：微调阶段保存的 mmcv checkpoint（如 epoch_3.pth）。"
            "若提供，将在 backbone 替换后继续加载其 state_dict 再评估。"
        ),
    )
    parser.add_argument("--gpu-id", type=int, default=2, help="使用的 GPU 编号")
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        default=["bbox"],
        help="evaluation metrics (与 tools/test.py 一致，默认 bbox)",
    )
    parser.add_argument(
        "--eval-modes",
        nargs="+",
        default=["det", "track", "map"],
        choices=["det", "track", "motion", "map"],
        help="推理时启用的模型任务头。默认 det track map 全任务评估。",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="保存推理结果的 pkl 文件路径",
    )
    parser.add_argument(
        "--subset-mode",
        choices=["full", "head", "skip_warmup"],
        default="full",
        help="数据集切分模式: full=全量val, head=固定头部采样, skip_warmup=跳过冷启动后采样",
    )
    parser.add_argument(
        "--frames-per-scene",
        type=int,
        default=10,
        help="head/skip_warmup 模式下每个 scene 采样的帧数（默认 10）",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="覆盖配置文件中的选项",
    )
    return parser.parse_args()


# ====================== main ======================
def main():
    log_path = setup_logging()
    print(f"[INFO] 评估日志保存至: {log_path}")

    args = parse_args()
    print(f"[INFO] 配置文件: {args.config}")
    print(f"[INFO] 原始 checkpoint: {args.original_checkpoint}")
    print(f"[INFO] backbone modelopt: {args.backbone_modelopt}")
    print(f"[INFO] 微调模型: {args.finetune_ckpt if args.finetune_ckpt else '(none)'}")
    print(f"[INFO] 评估任务: {args.eval_modes}")
    print(f"[INFO] 评估指标: {args.eval}")
    print(f"[INFO] 数据集模式: {args.subset_mode}"
          + (f" (frames_per_scene={args.frames_per_scene})" if args.subset_mode != "full" else ""))

    # 仅在子集模式下才需要 nuscenes eval patch（处理预测token与GT token不匹配）
    if args.subset_mode != "full":
        patch_nuscenes_eval_for_subset()

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

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # ---- 解冻 backbone/neck（与剪枝脚本一致） ----
    cfg.model_ego_agent.pretrained = None
    cfg.model_ego_agent.load_from = None
    cfg.model_ego_agent.freeze_img_backbone = False
    cfg.model_ego_agent.freeze_img_neck = False
    cfg.model_ego_agent.img_backbone.frozen_stages = 0
    for name in [k for k in cfg.keys() if "model_other_agent" in k]:
        cfg.get(name).load_from = None
        cfg.get(name).freeze_img_backbone = False
        cfg.get(name).freeze_img_neck = False
        cfg.get(name).img_backbone.frozen_stages = 0

    # ================================================================
    # 步骤 1: 构建完整模型 + 加载原始 checkpoint
    #   这样 neck/head/tracking 等非 backbone 部分拥有正确权重
    # ================================================================
    print("[INFO] 正在构建 MultiAgent 模型 + 加载原始 checkpoint...")
    full_model = build_multi_agent_model(cfg, args.original_checkpoint)

    total_params_original = sum(p.numel() for p in full_model.parameters())
    backbone_params_original = sum(
        p.numel() for p in full_model.model_ego_agent.img_backbone.parameters()
    )
    print(f"[INFO] 原始模型参数量: {total_params_original / 1e6:.2f} M")
    print(f"[INFO] 原始 backbone 参数量: {backbone_params_original / 1e6:.2f} M")

    # ================================================================
    # 步骤 2: 用 BackbonePruneWrapper 包装 backbone，mto.restore 恢复剪枝结构
    #
    # 关键：剪枝脚本中对 BackbonePruneWrapper(backbone) 执行 mtp.prune，
    # 然后 mto.save(pruned_backbone_wrapper, backbone_modelopt_path)。
    # 因此 backbone_modelopt.pth 的 key 前缀是 "backbone.*"，
    # mto.restore 需要传入相同结构的 BackbonePruneWrapper。
    # ================================================================
    print(f"[INFO] 正在恢复剪枝 backbone: {args.backbone_modelopt}")
    original_backbone = full_model.model_ego_agent.img_backbone
    backbone_wrapper = BackbonePruneWrapper(original_backbone)
    restored_wrapper = mto.restore(backbone_wrapper, args.backbone_modelopt)
    print("[INFO] mto.restore backbone 完成")

    # 将恢复后的剪枝 backbone 回填到 full_model
    pruned_backbone = restored_wrapper.backbone
    full_model.model_ego_agent.img_backbone = pruned_backbone
    print("[INFO] 剪枝 backbone 已回填到 full_model")

    backbone_params_pruned = sum(p.numel() for p in pruned_backbone.parameters())
    print(f"[INFO] 剪枝后 backbone 参数量: {backbone_params_pruned / 1e6:.2f} M "
          f"(保留 {backbone_params_pruned / backbone_params_original * 100:.2f}%)")

    # ---- 可选：叠加加载微调 checkpoint（例如 epoch_x.pth） ----
    if args.finetune_ckpt:
        if not osp.isfile(args.finetune_ckpt):
            raise FileNotFoundError(f"微调 checkpoint 不存在: {args.finetune_ckpt}")
        print(f"[INFO] 正在加载微调 checkpoint: {args.finetune_ckpt}")
        _ = load_checkpoint(
            full_model,
            args.finetune_ckpt,
            map_location="cpu",
            strict=False,
            revise_keys=[(r"^module\.", "")],
        )
        print("[INFO] 微调 checkpoint 加载完成（strict=False）")

    inner_model = full_model.to(device).eval()

    total_params_pruned = sum(p.numel() for p in inner_model.parameters())
    print(f"[INFO] 剪枝后全模型参数量: {total_params_pruned / 1e6:.2f} M")
    print(f"[INFO] 全模型参数保留比例: {total_params_pruned / total_params_original * 100:.2f}%")

    # ================================================================
    # 构建完整 val 数据集 —— 参照 verify_subset_eval.py --use-full-val
    #   使用 cfg.data.val 的副本（与剪枝脚本 score_split=val 对齐）
    # ================================================================
    print(f"[INFO] 构建 val 数据集 (subset_mode={args.subset_mode})...")

    eval_ds_cfg = cfg.data.val.copy()
    eval_ds_cfg.test_mode = True
    samples_per_gpu = eval_ds_cfg.pop("samples_per_gpu", 1)

    full_dataset = build_dataset(eval_ds_cfg)

    if args.subset_mode == "full":
        dataset = full_dataset
    else:
        indices = fixed_head_indices(
            full_dataset,
            frames_per_scene=args.frames_per_scene,
            sampling_mode=args.subset_mode,
        )
        dataset = SubsetDataset(full_dataset, indices)

    # 显式设置 eval_mod（控制推理时启用的任务头）
    dataset.eval_mod = args.eval_modes
    if hasattr(dataset, "_dataset"):
        dataset._dataset.eval_mod = args.eval_modes
    print(f"[INFO] 数据集大小: {len(dataset)}")
    print(f"[INFO] eval_mod = {dataset.eval_mod}")

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.get("nonshuffler_sampler", None),
    )

    # ================================================================
    # 推理 —— single_gpu_test
    # ================================================================
    print("[INFO] 开始在完整数据集上推理...")
    reset_temporal_states(inner_model)
    model_dp = MMDataParallel(inner_model, device_ids=[args.gpu_id])
    model_dp.eval()
    raw_outputs = single_gpu_test(model_dp, data_loader, show=False)

    bbox_outputs = raw_outputs.get("bbox_results", None) if isinstance(raw_outputs, dict) else raw_outputs
    if bbox_outputs is None:
        raise RuntimeError("single_gpu_test returns empty bbox outputs.")
    print(f"[INFO] 推理完成，得到 {len(bbox_outputs)} 个样本结果")
    if len(bbox_outputs) != len(dataset):
        print(
            f"[WARN] output sample count {len(bbox_outputs)} != dataset size {len(dataset)}"
        )
    summarize_outputs(bbox_outputs)

    if isinstance(raw_outputs, dict):
        raw_outputs["bbox_results"] = bbox_outputs
        outputs_for_eval = raw_outputs
    else:
        outputs_for_eval = bbox_outputs

    # ---- 保存推理结果（可选） ----
    if args.out:
        print(f"[INFO] 保存推理结果到: {args.out}")
        mmcv.dump(outputs_for_eval, args.out)

    # ================================================================
    # 评估 —— 参照 verify_subset_eval.py 的 evaluate 调用
    # ================================================================
    print("[INFO] 开始评估...")

    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
        eval_kwargs.pop(key, None)
    eval_kwargs["metric"] = args.eval

    kwargs = {} if args.eval_options is None else args.eval_options
    eval_kwargs.update(kwargs)

    eval_kwargs["jsonfile_prefix"] = osp.join(
        "test_eval_pruned",
        args.config.split("/")[-1].split(".")[-2],
        time.ctime().replace(" ", "_").replace(":", "_"),
    )

    results = dataset.evaluate(outputs_for_eval, **eval_kwargs)

    # ---- 打印评估结果 ----
    print("\n" + "=" * 70)
    print("评估结果 (全部指标)")
    print("=" * 70)
    if isinstance(results, dict):
        # 按任务分组打印
        det_keys = [k for k in sorted(results.keys()) if "NuScenes" in k or "AP" in k or "mAP" in k]
        track_keys = [k for k in sorted(results.keys()) if "track" in k.lower() or "AMOTA" in k or "AMOTP" in k]
        map_keys = [k for k in sorted(results.keys()) if "map" in k.lower() and "mAP" not in k]
        other_keys = [k for k in sorted(results.keys()) if k not in det_keys + track_keys + map_keys]

        if det_keys:
            print("\n--- Detection ---")
            for k in det_keys:
                print(f"  {k}: {results[k]}")
        if track_keys:
            print("\n--- Tracking ---")
            for k in track_keys:
                print(f"  {k}: {results[k]}")
        if map_keys:
            print("\n--- Map ---")
            for k in map_keys:
                print(f"  {k}: {results[k]}")
        if other_keys:
            print("\n--- Other ---")
            for k in other_keys:
                print(f"  {k}: {results[k]}")

        # 摘要关键指标
        print("\n" + "-" * 70)
        print("关键指标摘要:")
        for key_name in [
            "pts_bbox_NuScenes/mAP", "pts_bbox_NuScenes/NDS",
            "pts_bbox_NuScenes/mATE", "pts_bbox_NuScenes/mASE",
            "pts_bbox_NuScenes/mAOE", "pts_bbox_NuScenes/mAVE", "pts_bbox_NuScenes/mAAE",
        ]:
            if key_name in results:
                print(f"  {key_name}: {results[key_name]:.6f}")
        for key_name in ["AMOTA", "AMOTP"]:
            if key_name in results:
                print(f"  {key_name}: {results[key_name]:.6f}")
    else:
        print(results)

    print("\n" + "=" * 70)
    print(f"[INFO] 评估完成。日志: {log_path}")
    print(f"[INFO] 评估任务: {args.eval_modes}")
    print(f"[INFO] 剪枝后参数量: {total_params_pruned / 1e6:.2f} M "
          f"（原始 {total_params_original / 1e6:.2f} M，"
          f"保留 {total_params_pruned / total_params_original * 100:.2f}%）")
    print("=" * 70)


if __name__ == "__main__":
    main()
