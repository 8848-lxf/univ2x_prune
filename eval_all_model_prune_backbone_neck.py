"""
eval_all_model_prune_backbone_neck.py
-------------------------------------
对 all_model_prune_backbone_neck.py 剪枝产出的模型进行恢复（mto.restore）
并在完整 val 数据集上运行完整任务评估（det/track/map/motion 等）。

评估逻辑完全对齐原始 tools/test.py，使用配置文件中定义的 eval_mod。

用法：
    cd /home/lixingfeng/UniAD_exmaine/UniV2X
    python ./univ2x_purned/eval_all_model_prune_backbone_neck.py \
        ./projects/configs_e2e_univ2x/univ2x_coop_e2e.py \
        ./univ2x_purned/modelopt/all_model_prune_backbone_neck/output/all_model_prune_bb_neck_pruned_model_20260313_154715.pth \
        --gpu-id 7 \
        --eval bbox
"""

import os
import sys

# ---- Ensure UniV2X repo root is on sys.path ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import copy
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

import modelopt.torch.opt as mto

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent

warnings.filterwarnings("ignore")


# ====================== 日志 ======================
LOG_DIR = "/home/lixingfeng/UniAD_exmaine/UniV2X/univ2x_purned/modelopt/logs_eval_all_model_prune_backbone_neck"

import re as _re

_PROGRESS_RE = _re.compile(
    r'\r|'
    r'\[\s*\d+/\d+.*elapsed|'
    r'\d+%\|[▏▎▍▌▋▊▉█ ]*\|'
)


class _Tee:
    """Python 层 tee：终端原样输出，日志文件过滤掉进度条。"""
    def __init__(self, stream, log_fh):
        self._stream = stream
        self._log_fh = log_fh

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
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
    log_path = os.path.join(LOG_DIR, f"eval_{timestamp}.log")

    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    return log_path


# ====================== ModelOptWrapper ======================
# 必须与剪枝脚本中保存时使用的 wrapper 结构完全一致，
# 这样 mto.restore 才能正确匹配 state_dict 的 key 前缀。
# =============================================================
class ModelOptWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self._inner = model

    def forward(self, *args, **kwargs):
        return self._inner(*args, **kwargs)


# ====================== 模型构建 ======================
def build_multi_agent_model(cfg, original_checkpoint_path=None):
    """构建 MultiAgent 模型结构并加载原始权重。"""
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

    if original_checkpoint_path and os.path.exists(original_checkpoint_path):
        checkpoint = load_checkpoint(
            model_multi_agents, original_checkpoint_path, map_location="cpu"
        )
        if "CLASSES" in checkpoint.get("meta", {}):
            model_multi_agents.model_ego_agent.CLASSES = checkpoint["meta"]["CLASSES"]
        if "PALETTE" in checkpoint.get("meta", {}):
            model_multi_agents.model_ego_agent.PALETTE = checkpoint["meta"]["PALETTE"]

    return model_multi_agents


# ====================== 参数解析 ======================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate pruned MultiAgent model on full val set (all tasks)"
    )
    parser.add_argument("config", help="配置文件路径")
    parser.add_argument("pruned_model", help="mto.save 保存的剪枝模型路径 (.pth)")
    parser.add_argument(
        "--original-checkpoint",
        default="./ckpts/univ2x_coop_e2e_stg2.pth",
        help="原始未剪枝模型权重（用于构建模型骨架时加载 CLASSES/PALETTE 等 meta）",
    )
    parser.add_argument("--gpu-id", type=int, default=3, help="使用的 GPU 编号")
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        default=["bbox"],
        help="evaluation metrics (与 tools/test.py 一致，默认 bbox)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="保存推理结果的 pkl 文件路径",
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
    print(f"[INFO] 剪枝模型: {args.pruned_model}")
    print(f"[INFO] 评估指标: {args.eval}")

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
    cfg.model_ego_agent.freeze_img_backbone = False
    cfg.model_ego_agent.freeze_img_neck = False
    cfg.model_ego_agent.img_backbone.frozen_stages = 0
    for name in [k for k in cfg.keys() if "model_other_agent" in k]:
        cfg.get(name).freeze_img_backbone = False
        cfg.get(name).freeze_img_neck = False
        cfg.get(name).img_backbone.frozen_stages = 0

    # ---- 构建模型骨架并加载原始权重 ----
    print("[INFO] 正在构建 MultiAgent 模型骨架...")
    model = build_multi_agent_model(cfg, args.original_checkpoint)

    total_params_original = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 原始模型参数量: {total_params_original / 1e6:.2f} M")

    # ---- 用 ModelOptWrapper 包装 + mto.restore 恢复剪枝结构和权重 ----
    wrapped_model = ModelOptWrapper(model)
    print(f"[INFO] 正在恢复剪枝模型: {args.pruned_model}")
    restored_model = mto.restore(wrapped_model, args.pruned_model)
    print("[INFO] mto.restore 完成")

    inner_model = restored_model._inner if hasattr(restored_model, "_inner") else restored_model
    inner_model = inner_model.to(device).eval()

    total_params_pruned = sum(p.numel() for p in inner_model.parameters())
    print(f"[INFO] 剪枝后参数量: {total_params_pruned / 1e6:.2f} M")
    print(f"[INFO] 参数保留比例: {total_params_pruned / total_params_original * 100:.2f}%")

    # ================================================================
    # 构建完整 val 数据集 —— 完全对齐 tools/test.py 的逻辑
    #   使用 cfg.data.test（已在 config 中设置了 test_mode=True）
    #   eval_mod 直接来自配置文件（如 ['det', 'map', 'track', 'motion']）
    # ================================================================
    print("[INFO] 构建完整 val/test 数据集...")

    # 与 tools/test.py 完全一致的数据集构建方式
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )

    dataset = build_dataset(cfg.data.test)
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
    model_dp = MMDataParallel(inner_model, device_ids=[args.gpu_id])
    model_dp.eval()
    outputs = single_gpu_test(model_dp, data_loader, show=False)

    # ---- 保存推理结果（可选） ----
    if args.out:
        print(f"[INFO] 保存推理结果到: {args.out}")
        mmcv.dump(outputs, args.out)

    # ================================================================
    # 评估 —— 完全对齐 tools/test.py 的 evaluate 调用
    # ================================================================
    print("[INFO] 开始评估...")

    kwargs = {} if args.eval_options is None else args.eval_options
    kwargs["jsonfile_prefix"] = osp.join(
        "test_eval_pruned",
        args.config.split("/")[-1].split(".")[-2],
        time.ctime().replace(" ", "_").replace(":", "_"),
    )

    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=args.eval, **kwargs))

    results = dataset.evaluate(outputs, **eval_kwargs)

    # ---- 打印评估结果 ----
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    print(results)

    print("\n" + "=" * 70)
    print(f"[INFO] 评估完成。日志: {log_path}")
    print(f"[INFO] 剪枝后参数量: {total_params_pruned / 1e6:.2f} M "
          f"（原始 {total_params_original / 1e6:.2f} M，"
          f"保留 {total_params_pruned / total_params_original * 100:.2f}%）")
    print("=" * 70)


if __name__ == "__main__":
    main()
