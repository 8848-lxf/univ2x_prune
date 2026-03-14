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
from typing import Any, Dict

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

# DCN 剪枝支持：注册 ModulatedDeformConv2dPack 到 ModelOpt DMRegistry
from univ2x_purned.pruning_tools import register_mdconv, verify_registration

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
            if hasattr(self, 'sample_tokens'):
                self.sample_tokens = sorted(pred_tokens)
            print(f"[PATCH] NuScenesEval subset mode: GT filtered to {len(pred_tokens)} pred tokens")

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
# 固定头部采样：每个 val 场景从第一帧开始取前 frames_per_scene 帧
#   完全确定性，无随机性，保证 FastNAS 每轮 score_func 评估同一批样本。
#   21 个场景 × 10 帧 = 210 帧。
# =============================================================================
def fixed_head_indices(dataset, frames_per_scene=10):
    """
    对 val 集的每个 scene_token，取前 frames_per_scene 帧（时间顺序）。
    若某场景帧数 < frames_per_scene，则取全部帧。
    返回排序后的全局 index 列表（可直接传给 SubsetDataset）。
    """
    scene_to_indices = defaultdict(list)
    for i, info in enumerate(dataset.data_infos):
        # data_infos 已按 timestamp 排序，append 顺序即时间顺序
        scene_to_indices[info['scene_token']].append(i)

    scenes = sorted(scene_to_indices.keys())
    selected = []
    for s in scenes:
        idxs = scene_to_indices[s]
        selected.extend(idxs[:frames_per_scene])

    selected = sorted(selected)

    # ---- 打印采样统计 ----
    scene_counts = defaultdict(int)
    for i in selected:
        scene_counts[dataset.data_infos[i]['scene_token']] += 1

    print(f"[INFO] 固定头部采样：覆盖 {len(scene_counts)} / {len(scenes)} 个场景，"
          f"总帧数={len(selected)}")
    print(f"       每场景帧数：min={min(scene_counts.values())} "
          f"max={max(scene_counts.values())} "
          f"avg={np.mean(list(scene_counts.values())):.1f}")
    for s in scenes:
        print(f"       scene {s}: {scene_counts.get(s,0)} / "
              f"{len(scene_to_indices[s])} 帧")
    return selected


def single_scene_indices(dataset, scene_id="0003", num_frames=10):
    """
    只选取 scene_token 包含 scene_id 的场景，取前 num_frames 帧。
    返回排序后的全局 index 列表。
    """
    scene_to_indices = defaultdict(list)
    for i, info in enumerate(dataset.data_infos):
        scene_to_indices[info['scene_token']].append(i)

    # 模糊匹配：scene_token 中包含 scene_id
    matched = [s for s in scene_to_indices if scene_id in s]
    if not matched:
        all_scenes = sorted(scene_to_indices.keys())
        raise ValueError(
            f"scene_id='{scene_id}' 未匹配到任何场景。可用场景:\n"
            + "\n".join(f"  {s} ({len(scene_to_indices[s])} 帧)" for s in all_scenes)
        )
    if len(matched) > 1:
        print(f"[WARN] scene_id='{scene_id}' 匹配到 {len(matched)} 个场景，取第一个: {matched[0]}")
    scene_token = matched[0]
    idxs = scene_to_indices[scene_token][:num_frames]

    print(f"[INFO] 单场景采样：scene_token={scene_token}，"
          f"取前 {num_frames} 帧，实际 {len(idxs)} 帧 / 共 {len(scene_to_indices[scene_token])} 帧")
    return sorted(idxs)


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
    def __init__(self, model: nn.Module):
        super().__init__()
        self._inner = model
        self._skeleton = None   # _extract_tensors 生成的骨架

    def set_skeleton(self, skeleton):
        """保存骨架（含非 tensor 元数据 + _TensorSlot 占位符）。"""
        self._skeleton = skeleton

    def forward(self, *tensors) -> Any:
        is_tracing = torch.jit.is_tracing()

        # 回填 tensor → 重建完整输入（ndarray 始终转回 numpy）
        full_input = _fill_tensors(self._skeleton, tensors)

        with torch.no_grad():
            result = self._inner(
                ego_agent_data=full_input['ego'],
                other_agent_data_dict=full_input['others'],
                img_metas=full_input['img_metas'],
                return_loss=False,
                w_label=True,   # 所有 tensor 都是 forward 参数，GT 透传也有依赖
            )

        if is_tracing:
            flat = _flatten_output_tensors(result)
            # 注入对全部 input tensor 的微弱依赖（+0），
            # 防止个别输出（如 track_ids）因为不直接依赖某些输入而报错
            _dep = sum(t.float().sum() for t in tensors) * 0
            return tuple(t + _dep.to(device=t.device, dtype=t.dtype) for t in flat)
        return result


# --------------------------- 日志工具 ---------------------------
LOG_DIR = "/home/lixingfeng/UniAD_exmaine/UniV2X/univ2x_purned/modelopt/logs_all_model_prune_backbone_neck_DCN"

import re as _re

# 进度条特征：含 \r、或匹配 mmcv/tqdm 进度条格式
_PROGRESS_RE = _re.compile(
    r'\r|'                       # \r 覆写行
    r'\[\s*\d+/\d+.*elapsed|'    # mmcv ProgressBar: [  1/675, ... elapsed
    r'\d+%\|[▏▎▍▌▋▊▉█ ]*\|'     # tqdm 风格: 50%|████      |
)


class _Tee:
    """Python 层 tee：终端原样输出，日志文件过滤掉进度条。"""
    def __init__(self, stream, log_fh):
        self._stream = stream
        self._log_fh = log_fh

    def write(self, data):
        # 终端：原样输出（保留进度条动态效果）
        self._stream.write(data)
        self._stream.flush()
        # 日志文件：跳过进度条行，只记录有意义的输出
        if data and not _PROGRESS_RE.search(data):
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
    parser.add_argument("--gpu-id", type=int, default=3, help="使用的 GPU 编号")
    parser.add_argument("config", help="配置文件路径")
    parser.add_argument("checkpoint", help="待剪枝的完整模型权重路径")
    parser.add_argument(
        "--out-dir",
        default="/home/lixingfeng/UniAD_exmaine/UniV2X/univ2x_purned/modelopt/all_model_prune_backbone_neck_DCN/output",
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
        choices=["per_scene", "single_scene"],
        default="per_scene",
        help="校准集子集切分模式",
    )
    parser.add_argument(
        "--calib-frames-per-scene",
        type=int,
        default=5,
        help="校准集每场景取的帧数",
    )
    parser.add_argument(
        "--calib-scene-id",
        type=str,
        default="0003",
        help="校准集 single_scene 模式下的场景 ID",
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
        choices=["per_scene", "single_scene"],
        default="per_scene",
        help="评分集子集切分模式（推荐 per_scene 覆盖更多场景）",
    )
    parser.add_argument(
        "--score-frames-per-scene",
        type=int,
        default=10,
        help="评分集每场景取的帧数（per_scene 模式：21 场景 × 10 帧 = 210 帧）",
    )
    parser.add_argument(
        "--score-scene-id",
        type=str,
        default="0003",
        help="评分集 single_scene 模式下的场景 ID",
    )
    parser.add_argument(
        "--params-percent",
        type=str,
        default="95%",
        help="剪枝后参数量占原始模型的上限比例，例如 '95%%'",
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
    return parser.parse_args()


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


# --------------------------- DataLoader：固定子集 ---------------------------
def build_subset_loader(cfg, split="val", subset_mode="single_scene",
                        frames_per_scene=10, scene_id="0003"):
    """
    构建固定子集的 DataLoader。
      - per_scene:     每个场景取前 frames_per_scene 帧（21 场景 × N 帧）
      - single_scene:  只取 scene_id 对应场景的前 frames_per_scene 帧
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
    if subset_mode == "single_scene":
        indices = single_scene_indices(full_dataset, scene_id=scene_id, num_frames=frames_per_scene)
    else:
        indices = fixed_head_indices(full_dataset, frames_per_scene=frames_per_scene)

    # 3. SubsetDataset：裁剪 data_infos，评估器只看子集 token
    subset = SubsetDataset(full_dataset, indices)
    print(f"[INFO] 子集大小: {len(subset)} 帧（mode={subset_mode}）")

    # 4. 构建 DataLoader
    loader = build_dataloader(
        subset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.get("nonshuffler_sampler", None),
    )
    return subset, loader


# --------------------------- score func ---------------------------
@torch.no_grad()
def _eval_car_mean_ap(model_dp: MMDataParallel, dataset, data_loader, eval_kwargs: Dict[str, Any]) -> float:
    model_dp.eval()
    outputs = single_gpu_test(model_dp, data_loader, show=False)

    for i in range(len(outputs)):
        det = outputs[i]
        if isinstance(det, dict):
            det.setdefault("ret_iou", {})
        elif isinstance(det, list) and len(det) > 0 and isinstance(det[0], dict):
            det[0].setdefault("ret_iou", {})

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

def build_score_func(dataset, val_loader, eval_kwargs: Dict[str, Any], gpu_id: int = 0):
    def score_func(model: nn.Module) -> float:
        real_model = model._inner if hasattr(model, "_inner") else model
        real_model = real_model.to(f"cuda:{gpu_id}")  # ✅ 稳：确保 device 一致
        model_dp = MMDataParallel(real_model, device_ids=[gpu_id])
        return _eval_car_mean_ap(model_dp, dataset, val_loader, eval_kwargs)
    return score_func


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

    # ---- 注册 DCN 动态模块到 ModelOpt，必须在 mtp.prune() 之前 ----
    register_mdconv()
    verify_registration()

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

    wrapped_model = ModelOptWrapper(model).to(device).eval()

    # ================================================================
    # 校准集（BN 统计量更新）：训练集子集
    #   BN running_mean/running_var 应反映训练分布，用训练集子集更新
    # ================================================================
    print(f"[INFO] 构建 BN 校准集（split={args.calib_split}，"
          f"mode={args.calib_subset_mode}，{args.calib_frames_per_scene} 帧/场景"
          f"{f'，scene_id={args.calib_scene_id}' if args.calib_subset_mode == 'single_scene' else ''}）...")
    calib_dataset, calib_loader = build_subset_loader(
        cfg,
        split=args.calib_split,
        subset_mode=args.calib_subset_mode,
        frames_per_scene=args.calib_frames_per_scene,
        scene_id=args.calib_scene_id,
    )
    print(f"[INFO] 校准集 size = {len(calib_dataset)}")

    # ================================================================
    # 评分集（候选子网排序）：val 子集 per_scene 210 帧
    #   评估泛化性能，覆盖多个场景，避免搜索过拟合
    # ================================================================
    print(f"[INFO] 构建评分集（split={args.score_split}，"
          f"mode={args.score_subset_mode}，{args.score_frames_per_scene} 帧/场景"
          f"{f'，scene_id={args.score_scene_id}' if args.score_subset_mode == 'single_scene' else ''}）...")
    val_dataset, val_loader = build_subset_loader(
        cfg,
        split=args.score_split,
        subset_mode=args.score_subset_mode,
        frames_per_scene=args.score_frames_per_scene,
        scene_id=args.score_scene_id,
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

    score_func = build_score_func(val_dataset, val_loader, eval_kwargs, gpu_id=args.gpu_id)

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
        args.out_dir, f"all_model_prune_bb_neck_DCN_pruned_model_{run_ts}.pth")
    # 搜索 checkpoint：未指定则自动生成新路径（不会加载旧状态）
    if args.search_checkpoint is not None:
        search_ckpt_path = args.search_checkpoint
        print(f"[INFO] 断点续搜，加载: {search_ckpt_path}")
    else:
        search_ckpt_path = osp.join(
            args.out_dir, f"all_model_prune_bb_neck_DCN_fastnas_search_checkpoint_{run_ts}.pth")
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
        # ---- DCN 层（ResNet layer3/layer4 的 conv2）剪枝 ----
        # ModulatedDeformConv2dPack 结构：
        #   - weight(out_channels, in_channels/groups, kH, kW): 主卷积权重 → 可剪 out/in
        #   - bias(out_channels): 主卷积偏置 → 随 out_channels 联动
        #   - conv_offset: Conv2d(in_channels, 27, 3x3) → out=27 固定不剪，in 与主 conv 联动
        # SymMap 会自动将 conv_offset.in_channels 与父模块 in_channels 绑定为同一 TracedHp，
        # conv_offset.out_channels(=27) 被识别为常量 Symbol，不会被搜索。
        "mmcv.ops.ModulatedDeformConv2dPack": {
            "*_inner.model_ego_agent.img_backbone.layer3*": {"channel_divisor": 16},
            "*_inner.model_ego_agent.img_backbone.layer4*": {"channel_divisor": 16},
            "*_inner.model_other_agent_inf.img_backbone.layer3*": {"channel_divisor": 16},
            "*_inner.model_other_agent_inf.img_backbone.layer4*": {"channel_divisor": 16},
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
            "data_loader": calib_loader,        # 训练集子集（BN 校准）
            "score_func": score_func,           # val 子集 210 帧（候选网络排序）
            "checkpoint": search_ckpt_path,
            "collect_func": collect_func,
        },
    )

    # ---- 恢复原始 deepcopy ----
    _copy_module.deepcopy = _original_deepcopy
    print("[PATCH] copy.deepcopy restored")
    # --------------------------
    # ✅ 验收：heads 不应进入 searchable hparams
    assert_heads_not_searchable(pruned_model)

    # ✅ 验收：DCN conv_offset 联动一致性
    from univ2x_purned.pruning_tools.dcn_fastnas_config import _verify_conv_offset_coupling
    _verify_conv_offset_coupling(pruned_model)

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

    print("\n" + "=" * 60)
    print("剪枝完成！后续步骤：")
    print("  1. 微调/推理时，build 完整模型结构（heads 全在），再 mto.restore：")
    print(f"       wrapped = mto.restore(ModelOptWrapper(full_model), '{out_model_path}')")
    print("  2. wrapped._inner 即为剪枝后的完整 MultiAgent（仅 bb/neck 被剪）")
    print("=" * 60)


if __name__ == "__main__":
    main()