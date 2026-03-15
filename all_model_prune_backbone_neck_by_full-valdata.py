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


# NOTE: 本脚本使用全量 val 集评估，不使用子集采样和 GT 过滤补丁。



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
LOG_DIR = "/home/lixingfeng/UniAD_exmaine/UniV2X/univ2x_purned/modelopt/logs_all_model_prune_backbone_neck"

import re as _re

# 进度条特征：含 \r、或匹配 mmcv/tqdm 进度条格式
_PROGRESS_RE = _re.compile(
    r'\r|'                       # \r 覆写行
    r'\[\s*\d+/\d+.*elapsed|'    # mmcv ProgressBar: [  1/675, ... elapsed
    r'\d+%\|[▏▎▍▌▋▊▉█ ]*\|'     # tqdm 风格: 50%|████      |
)

# mmcv ProgressBar 格式：[  123/675, 0.5 s/iter, 4.5s elapsed]
_MMCV_PROGRESS_RE = _re.compile(r'\[\s*(\d+)/(\d+)')
# tqdm 格式：50%|████| 123/456
_TQDM_PROGRESS_RE = _re.compile(r'(\d+)%\|')

# 需要记录的里程碑百分比
_MILESTONE_PERCENTS = {25, 50, 75, 100}


class _Tee:
    """Python 层 tee：终端原样输出，日志文件按里程碑记录进度。"""
    def __init__(self, stream, log_fh):
        self._stream = stream
        self._log_fh = log_fh
        self._logged_milestones = set()  # 已记录的里程碑百分比
        self._last_task_id = None        # 用于区分不同进度条任务

    def _check_milestone(self, data):
        """检查进度条是否到达 25%/50%/75%/100% 里程碑，是则写入日志。"""
        # 尝试解析 mmcv 格式: [  123/675, ...]
        m = _MMCV_PROGRESS_RE.search(data)
        if m:
            current, total = int(m.group(1)), int(m.group(2))
            if total > 0:
                pct = current * 100 // total
                task_id = f"mmcv_{total}"
                # 新任务时重置里程碑
                if task_id != self._last_task_id:
                    self._logged_milestones.clear()
                    self._last_task_id = task_id
                for milestone in sorted(_MILESTONE_PERCENTS):
                    if pct >= milestone and milestone not in self._logged_milestones:
                        self._logged_milestones.add(milestone)
                        self._log_fh.write(
                            f"[PROGRESS] {current}/{total} ({milestone}%)\n")
                        self._log_fh.flush()
            return

        # 尝试解析 tqdm 格式: 50%|████|
        m2 = _TQDM_PROGRESS_RE.search(data)
        if m2:
            pct = int(m2.group(1))
            task_id = "tqdm"
            if task_id != self._last_task_id:
                self._logged_milestones.clear()
                self._last_task_id = task_id
            for milestone in sorted(_MILESTONE_PERCENTS):
                if pct >= milestone and milestone not in self._logged_milestones:
                    self._logged_milestones.add(milestone)
                    self._log_fh.write(f"[PROGRESS] tqdm {milestone}%\n")
                    self._log_fh.flush()

    def write(self, data):
        # 终端：原样输出（保留进度条动态效果）
        self._stream.write(data)
        self._stream.flush()
        # 日志文件：对进度条检查里程碑，非进度条正常记录
        if data:
            if _PROGRESS_RE.search(data):
                self._check_milestone(data)
            else:
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
    parser.add_argument("--gpu-id", type=int, default=7, help="使用的 GPU 编号")
    parser.add_argument("config", help="配置文件路径")
    parser.add_argument("checkpoint", help="待剪枝的完整模型权重路径")
    parser.add_argument(
        "--out-dir",
        default="/home/lixingfeng/UniAD_exmaine/UniV2X/univ2x_purned/modelopt/all_model_prune_backbone_neck/output",
        help="剪枝输出目录，文件名自动带时间戳",
    )
    parser.add_argument(
        "--search-checkpoint",
        default=None,
        help="断点续搜：指定已有的搜索 checkpoint 路径。不指定则每次从头搜索",
    )
    # ---- BN 校准集划分（默认 val，全量） ----
    parser.add_argument(
        "--calib-split",
        choices=["train", "val", "test"],
        default="val",
        help="BN 校准数据集划分（本脚本使用全量数据，不做子集切分）",
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


# --------------------------- DataLoader：全量数据集 ---------------------------
def build_full_loader(cfg, split="val"):
    """
    构建全量数据集的 DataLoader（不做任何子集切分）。
    """
    if split in ("test", "val"):
        ds_cfg = copy.deepcopy(cfg.data.test if split == "test" else cfg.data.val)
        ds_cfg.test_mode = True
        samples_per_gpu = ds_cfg.pop("samples_per_gpu", 1)
    else:
        ds_cfg = copy.deepcopy(cfg.data.train)
        ds_cfg.test_mode = True    # 校准时用推理模式，不加数据增强
        ds_cfg.pipeline = copy.deepcopy(cfg.data.test.pipeline)
        samples_per_gpu = ds_cfg.pop("samples_per_gpu", 1)

    dataset = build_dataset(ds_cfg)
    print(f"[INFO] 全量 {split} 集大小: {len(dataset)}")

    loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.get("nonshuffler_sampler", None),
    )
    return dataset, loader


# --------------------------- score func ---------------------------
@torch.no_grad()
def _eval_car_mean_ap(model_dp: MMDataParallel, dataset, data_loader, eval_kwargs: Dict[str, Any]) -> float:
    print(f"[score_func] >>>>>> ENTERING _eval_car_mean_ap, dataset size={len(dataset)}, "
          f"loader batches={len(data_loader)} <<<<<<", flush=True)
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
    _call_count = [0]  # mutable counter in closure

    def score_func(model: nn.Module) -> float:
        _call_count[0] += 1
        print(f"\n{'='*60}", flush=True)
        print(f"[score_func] >>>>>> CALLED (#{_call_count[0]}) <<<<<<", flush=True)
        print(f"[score_func] model type = {type(model).__name__}, "
              f"has _inner = {hasattr(model, '_inner')}", flush=True)
        print(f"{'='*60}", flush=True)
        real_model = model._inner if hasattr(model, "_inner") else model
        real_model = real_model.to(f"cuda:{gpu_id}")  # ✅ 稳：确保 device 一致
        model_dp = MMDataParallel(real_model, device_ids=[gpu_id])
        score = _eval_car_mean_ap(model_dp, dataset, val_loader, eval_kwargs)
        print(f"[score_func] >>>>>> RETURNING score = {score:.6f} (call #{_call_count[0]}) <<<<<<\n",
              flush=True)
        return score
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
    # 校准集（BN 统计量更新）：全量数据集
    # ================================================================
    print(f"[INFO] 构建 BN 校准集（split={args.calib_split}，全量）...")
    calib_dataset, calib_loader = build_full_loader(cfg, split=args.calib_split)
    print(f"[INFO] 校准集 size = {len(calib_dataset)}")

    # ================================================================
    # 评分集（候选子网排序）：全量 val 集
    # ================================================================
    print("[INFO] 构建评分集（split=val，全量）...")
    val_dataset, val_loader = build_full_loader(cfg, split="val")
    val_dataset.eval_mod = args.eval_modes
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