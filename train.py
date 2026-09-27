"""实验1 · 统一训练入口：一次训练「全部输入策略 × 全部模型」，保存权重与训练参数。

策略（统一使用原始数据的后 DATA_TAIL=20000 个点，同一套 seq_len / pred_len / 70-15-15 切分 /
逐变量标准化 / 早停 / 超参设置）：
    univariate          单变量（主蒸汽流量）
    univariate_lag      主蒸汽流量滞后增强（K=4 通道）
    paper               主蒸汽流量 + 6 个状态变量
    cluster_A           主蒸汽流量 + 筛选保留变量（读实验2 retained_features.csv）
    cluster_A_state     cluster_A + 6 个状态变量（去重）
    paper_plus_primary  paper + 6 个一次风变量（可选，不在默认列表）

模型：10 个中先训练 8 个（默认排除 Channel-Independent-LSTM / LSTM-iTransformer；RTF 因
    Autoformer FFT 在 fp16 下受限，自动改 fp32 训练）。每个 (模型, 配置) 在多种子下各训一次。

用法：
    python train.py                          # 默认 8 个模型 × 5 个策略 × 6 个种子
    python train.py --models PatchTST,LSTM_mv
    python train.py --configs univariate,paper
    python train.py --seeds 10,20,30,42,50,60
    python train.py --smoke                  # 只做一次前向+反向的形状自检，不真正训练

输出：
    checkpoints/<model>__<config>.pth        每个 (模型, 配置) 的权重 + 元信息
    train_summary.csv                        每组的训练参数（参数量/最优轮次/验证损失等）
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from extract_column import get_column                    # noqa: E402
from models import build_model, MODEL_NAMES, MODEL_DISPLAY, MODEL_OUTPUT_MODE, ARCH_DEFAULTS  # noqa: E402

# Windows GBK 控制台：尽量用 UTF-8 输出，避免中文乱码（不影响数据正确性）
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# 输入配置（四组）
# ═══════════════════════════════════════════════════════════════════════════════

STEAM = '主蒸汽流量'

# 论文多元变量（状态变量）：主蒸汽流量 + 6 个强相关过程状态变量
# 注：以下中文变量名为按论文英文描述与 data_array.npy 实际列名匹配得到，若需调整请修改此处。
STATE_VARS = [
    STEAM,                              # 1. Main steam flow（主蒸汽流量）
    '锅炉汽包右侧压力',                  # 2. Drum pressure（汽包压力 / steam drum pressure 2）
    '蒸发器I进口左侧烟气温度1',           # 3. Flue gas temp, left of evaporator inlet
    '过热蒸汽集箱压力1',                 # 4. Outlet pressure of superheated steam
    '中温过热器进口右侧烟气温度1',         # 5. Flue gas temp, right of medium-temp SH inlet
    '第1烟道上部前侧烟气温度T11',         # 6. Flue gas temp, front-middle of first flue (T11)
    '一级省煤器出口烟气温度',             # 7. Flue gas temp, left of economizer outlet
]

# 一次风驱动变量（6 个）
PRIMARY_AIR_VARS = [
    '锅炉燃烧段一次风机电机转速',          # primary-air fan motor speed, combustion section
    '锅炉燃烧段一次风机电流',             # primary-air fan motor current, combustion section
    '燃烧段一次风机出口压力',             # primary-air outlet pressure, combustion section
    '干燥段一次风量计算',                # calculated primary-air flow, drying section
    '燃烧段一次风量计算',                # calculated primary-air flow, combustion section
    '燃烬段一次风量计算',                # calculated primary-air flow, burnout section
]

# ---- 实验2 的筛选保留变量（cluster_A / cluster_A_state 用）----
EXP2_RESULTS = ROOT / '实验2-多变量效果检验(特征筛选所提方法)' / 'results'
RETAINED_CSV = EXP2_RESULTS / 'retained_features.csv'


def _load_retained_vars() -> list[str]:
    """读取实验2脚本2的保留变量名（retained_features.csv 的 variable_name 列）。"""
    if not RETAINED_CSV.exists():
        print(f'[警告] 未找到 {RETAINED_CSV}，cluster_A / cluster_A_state 配置将缺失。'
              f'请先运行 实验2 的 02_lag_correlation_screening.py。')
        return []
    with open(RETAINED_CSV, encoding='utf-8-sig', newline='') as f:
        return [r['variable_name'] for r in csv.DictReader(f)]


RETAINED_VARS = _load_retained_vars()
# cluster_A_state 追加的状态变量（实验1 paper 的 6 个状态变量，去除已在保留变量中的）
STATE_VARS_EXTRA = [s for s in STATE_VARS[1:] if s not in set(RETAINED_VARS)]

# 滞后增强单变量配置的超参数：构造 K 个延迟通道，通道 k 后退 step*k 步
LAG_N_CHANNELS = 4
LAG_STEP = 5

CONFIGS = {
    'univariate':           {'vars': [STEAM], 'lag': None},
    'univariate_lag':       {'vars': [STEAM], 'lag': {'n_channels': LAG_N_CHANNELS, 'step': LAG_STEP}},
    'paper':                {'vars': STATE_VARS, 'lag': None},
    'cluster_A':            {'vars': [STEAM] + RETAINED_VARS, 'lag': None},
    'cluster_A_state':      {'vars': [STEAM] + RETAINED_VARS + STATE_VARS_EXTRA, 'lag': None},
    'paper_plus_primary':   {'vars': STATE_VARS + PRIMARY_AIR_VARS, 'lag': None},
}

CONFIG_DISPLAY = {
    'univariate': 'Univariate (Steam-only)',
    'univariate_lag': 'Univariate (Lag-enhanced)',
    'paper': 'Paper multivariate (State)',
    'cluster_A': 'Cluster A (retained)',
    'cluster_A_state': 'Cluster A + state',
    'paper_plus_primary': 'Paper + Primary-air',
}

# 默认训练「上面提到的」5 个策略（paper_plus_primary 保留可用，但不在默认列表）
DEFAULT_CONFIGS = ['univariate', 'univariate_lag', 'paper', 'cluster_A', 'cluster_A_state']

# 统一训练设置（跨模型、跨配置保持一致）
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# 只使用原始数据的最后 DATA_TAIL 个时间点（None = 全量），再按 70/15/15 时序切分
DATA_TAIL = 20000
SEED = 42            # 固定随机种子，保证训练可复现

# 多种子训练：除被排除模型外，其余模型在这批种子下各训一次（默认即全量）
SEEDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 暂不训练的模型（Channel-Independent-LSTM / LSTM-iTransformer）
EXCLUDE_MODELS = {'LSTM_ci', 'LSTM_iTransformer'}

# 通道独立类模型的逐通道损失权重：主蒸汽流量（channel 0）权重最高，其余辅助通道低权重监督
MAIN_WEIGHT = 1.0
OTHER_WEIGHT = 0.2

# RTF 因 Autoformer 的 FFT 在 CUDA half 精度下受限（cuFFT 要求 2 的幂次长度），
# 故训练时对 RTF 强制 fp32（use_amp=False），其余模型保持 AMP。
RTF_MODEL = 'RTF'


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch（CPU+GPU）随机种子，保证训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════════
# 数据管线
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(var_names: list[str]) -> np.ndarray:
    """按变量名抽取列并拼成 (N, D) 特征矩阵（主蒸汽流量恒在第 0 列）。"""
    cols = []
    for v in var_names:
        col = get_column(v, as_float=True).astype(np.float32)
        cols.append(col)
    matrix = np.column_stack(cols)
    if DATA_TAIL is not None:
        matrix = matrix[-DATA_TAIL:]
    return matrix


def split_standardize(matrix: np.ndarray, seq_len: int):
    """70/15/15 时序切分 + 逐列 z-score 标准化（仅用训练集统计量）。

    返回 (train, val, test, mean, std)，其中 mean/std 形状为 (1, D)。
    """
    n = len(matrix)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_raw = matrix[:train_end]
    val_raw = matrix[train_end - seq_len: val_end]
    test_raw = matrix[val_end - seq_len:]

    mean = train_raw.mean(axis=0, keepdims=True)
    std = train_raw.std(axis=0, keepdims=True) + 1e-6

    return ((train_raw - mean) / std,
            (val_raw - mean) / std,
            (test_raw - mean) / std,
            mean, std)


class SteamDataset(Dataset):
    """滑动窗口数据集：x=(seq_len, D)，y=(pred_len, D) 为所有通道的未来值。"""

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n = len(data) - seq_len - pred_len + 1
        if self.n <= 0:
            raise ValueError(f'数据长度不足以构造窗口: len={len(data)}, seq_len={seq_len}, pred_len={pred_len}')

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        # y = 所有通道的未来 (pred_len, n_vars)；单目标模型在损失里取第 0 列
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return x, y


def build_config_data(config_name: str, seq_len: int, pred_len: int, batch_size: int) -> dict:
    """为某一输入配置构建 train/val/test DataLoader 及反标准化统计量。

    返回 dict: train_loader, val_loader, test_loader, n_vars, var_names,
    steam_mean, steam_std（主蒸汽流量反标准化用标量）。
    """
    cfg = CONFIGS[config_name]

    def make_loaders(train, val, test):
        def mk(data, shuffle):
            ds = SteamDataset(data, seq_len, pred_len)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                              pin_memory=torch.cuda.is_available())
        return mk(train, True), mk(val, False), mk(test, False)

    if cfg['lag'] is None:
        matrix = build_feature_matrix(cfg['vars'])
        train, val, test, mean, std = split_standardize(matrix, seq_len)
        steam_mean, steam_std = float(mean[0, 0]), float(std[0, 0])
        var_names = cfg['vars']
        n_vars = len(var_names)
        matrix_std = (matrix - mean) / std   # 全量时序标准化矩阵（供 RTF 在线回看历史）
        val_end = int(len(matrix) * (TRAIN_RATIO + VAL_RATIO))
    else:
        # 滞后增强：先对原始主蒸汽流量标准化，再构造 K 个滞后通道
        n_ch = cfg['lag']['n_channels']
        step = cfg['lag']['step']
        s = get_column(STEAM, as_float=True).astype(np.float32)
        if DATA_TAIL is not None:
            s = s[-DATA_TAIL:]
        N = len(s)
        train_end_raw = int(N * TRAIN_RATIO)
        steam_mean = float(s[:train_end_raw].mean())
        steam_std = float(s[:train_end_raw].std()) + 1e-6
        s_std = (s - steam_mean) / steam_std

        offset = step * (n_ch - 1)
        # 通道 k = 原序列滞后 step*k 步；第 0 列 = 无滞后主蒸汽流量
        C = np.column_stack([s_std[offset - step * k: N - step * k] for k in range(n_ch)])
        n = len(C)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        train = C[:train_end]
        val = C[train_end - seq_len: val_end]
        test = C[val_end - seq_len:]
        var_names = [f'{STEAM}@lag{step * k}' for k in range(n_ch)]
        n_vars = n_ch
        matrix_std = C                          # 滞后通道矩阵（已标准化），供 RTF 回看历史

    train_loader, val_loader, test_loader = make_loaders(train, val, test)
    return {
        'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
        'n_vars': n_vars, 'var_names': var_names,
        'steam_mean': steam_mean, 'steam_std': steam_std,
        'matrix_std': matrix_std, 'val_end': val_end,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════════════════════════

def train_epochs(model, train_loader, val_loader, epochs, lr, device, patience, output_mode, use_amp=None):
    """统一训练循环（Adam + 梯度裁剪 + AMP + 早停），返回 (model, best_epoch, best_val_loss)。

    output_mode='multi'  时用逐通道加权 MSE（主蒸汽流量=channel 0 权重最高，其余辅助通道低权重）；
    output_mode='single' 时直接对主蒸汽流量（channel 0）算 MSE。
    use_amp=None 时按设备自动（CUDA 开启）；RTF 因 Autoformer 的 FFT 在 half 精度下受限，须传 False。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if use_amp is None:
        use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    if output_mode == 'multi':
        n_vars = getattr(model, 'n_vars')
        channel_weights = torch.full((n_vars,), OTHER_WEIGHT, device=device)
        channel_weights[0] = MAIN_WEIGHT

        def loss_fn(pred, y):
            mse = ((pred - y) ** 2).mean(dim=(0, 1))     # (V,)
            return (mse * channel_weights).sum()
    else:
        def loss_fn(pred, y):
            return ((pred - y[:, :, 0]) ** 2).mean()

    best_val = float('inf')
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_total += loss.item() * x.size(0)
        train_loss = train_total / max(1, len(train_loader.dataset))

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    val_total += loss_fn(model(x), y).item() * x.size(0)
        val_loss = val_total / max(1, len(val_loader.dataset))

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or no_improve >= patience:
            print(f'      epoch {epoch:3d}/{epochs} | train_loss={train_loss:.6f} '
                  f'| val_loss={val_loss:.6f} | best={best_val:.6f} | no_improve={no_improve}/{patience}')

        if no_improve >= patience:
            print(f'      -> 早停于 epoch {epoch}（best_epoch={best_epoch}）')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_val


def train_one(model_name: str, config_name: str, device: torch.device, args, seed: int) -> dict:
    """训练单个 (模型, 配置, 种子) 组合，保存权重并返回训练参数记录。"""
    tag = f'{model_name}__{config_name}__seed{seed}'
    ckpt_path = args.checkpoints_dir / f'{tag}.pth'
    # 已训练过的跳过，避免重复训练
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        print(f'[跳过] {tag}.pth 已存在，跳过训练（best_epoch={ckpt.get("best_epoch")}）。')
        return {
            'model': model_name,
            'config': config_name,
            'seed': seed,
            'n_vars': ckpt.get('n_vars'),
            'n_params': ckpt.get('n_params'),
            'best_epoch': ckpt.get('best_epoch'),
            'best_val_loss': ckpt.get('best_val_loss'),
        }

    data = build_config_data(config_name, args.seq_len, args.pred_len, args.batch_size)
    n_vars = data['n_vars']
    output_mode = MODEL_OUTPUT_MODE[model_name]

    model = build_model(model_name, n_vars=n_vars, seq_len=args.seq_len, pred_len=args.pred_len).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'\n{"=" * 74}\n'
          f'[训练] 模型={MODEL_DISPLAY[model_name]} ({model_name}) | '
          f'配置={CONFIG_DISPLAY[config_name]} | seed={seed} | n_vars={n_vars} | 输出模式={output_mode}\n'
          f'  seq_len={args.seq_len} pred_len={args.pred_len} batch={args.batch_size} '
          f'lr={args.lr} epochs={args.epochs} patience={args.patience} params={n_params:,}\n'
          f'  变量: {data["var_names"]}\n'
          f'{"=" * 74}')

    use_amp = False if model_name == RTF_MODEL else None   # RTF 需 fp32（Autoformer FFT 限制）
    model, best_epoch, best_val = train_epochs(
        model, data['train_loader'], data['val_loader'],
        args.epochs, args.lr, device, args.patience, output_mode, use_amp=use_amp,
    )

    # 保存权重 + 元信息（供 test.py 复现模型与反标准化）
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.checkpoints_dir / f'{tag}.pth'
    torch.save({
        'model_name': model_name,
        'config_name': config_name,
        'seed': seed,
        'n_vars': n_vars,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'var_names': data['var_names'],
        'steam_mean': data['steam_mean'],
        'steam_std': data['steam_std'],
        'arch_kwargs': dict(ARCH_DEFAULTS[model_name]),
        'model_state_dict': model.state_dict(),
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val),
        'n_params': n_params,
    }, ckpt_path)
    print(f'  [保存] {ckpt_path.name}')

    return {
        'model': model_name,
        'config': config_name,
        'seed': seed,
        'n_vars': n_vars,
        'n_params': n_params,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val),
        'checkpoint': ckpt_path.name,
    }


def smoke_check(device: torch.device, seq_len: int, pred_len: int):
    """对所有 (模型, 配置) 做一次前向+反向，验证形状无误，不训练、不保存。"""
    print('== 冒烟自检（仅一次前向/反向，不训练）==')
    for config_name in CONFIGS:
        data = build_config_data(config_name, seq_len, pred_len, batch_size=8)
        n_vars = data['n_vars']
        for model_name in MODEL_NAMES:
            model = build_model(model_name, n_vars=n_vars, seq_len=seq_len, pred_len=pred_len).to(device)
            x, y = next(iter(data['train_loader']))
            x, y = x.to(device), y.to(device)
            out = model(x)
            # 依据输出模式校验形状：multi -> (B, pred_len, n_vars)，single -> (B, pred_len)
            if MODEL_OUTPUT_MODE[model_name] == 'multi':
                expected = y.shape
            else:
                expected = y[:, :, 0].shape
            assert out.shape == expected, f'{model_name}/{config_name}: {tuple(out.shape)} vs {tuple(expected)}'
            target = y if MODEL_OUTPUT_MODE[model_name] == 'multi' else y[:, :, 0]
            (out - target).pow(2).mean().backward()
            print(f'  OK {model_name:16s} {config_name:20s} n_vars={n_vars} '
                  f'mode={MODEL_OUTPUT_MODE[model_name]} out={tuple(out.shape)}')
    print('[OK] 冒烟自检通过')


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_seeds(s: str) -> list[int]:
    """把 '10,20,30' 解析为 int 列表。"""
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description='实验1：多模型 × 多变量配置 × 多种子训练')
    p.add_argument('--models', type=str, default=','.join(MODEL_NAMES),
                   help='逗号分隔的模型名，默认全部（但会排除 EXCLUDE_MODELS 里的 LSTM_ci/LSTM_iTransformer）')
    p.add_argument('--configs', type=str, default=','.join(DEFAULT_CONFIGS),
                   help='逗号分隔的配置名，默认 5 个策略')
    p.add_argument('--seeds', type=_parse_seeds, default=SEEDS,
                   help='逗号分隔的随机种子列表，默认 10,20,30,40,50,60,70,80,90,100')
    p.add_argument('--seq-len', type=int, default=60)
    p.add_argument('--pred-len', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--checkpoints-dir', type=Path, default=HERE / 'checkpoints')
    p.add_argument('--smoke', action='store_true', help='只做形状自检，不训练')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f'Device: {device}')

    model_names = [m.strip() for m in args.models.split(',') if m.strip()]
    config_names = [c.strip() for c in args.configs.split(',') if c.strip()]
    model_names = [m for m in model_names if m in MODEL_NAMES and m not in EXCLUDE_MODELS]
    config_names = [c for c in config_names if c in CONFIGS]
    seeds = args.seeds

    if args.smoke:
        set_seed(SEED)
        smoke_check(device, args.seq_len, args.pred_len)
        return

    print(f'训练计划：{len(model_names)} 个模型 × {len(config_names)} 个配置 × {len(seeds)} 个种子 = '
          f'{len(model_names) * len(config_names) * len(seeds)} 组')

    records = []
    for seed in seeds:
        set_seed(seed)
        print(f'\n{"#" * 74}\n# 种子 seed={seed}\n{"#" * 74}')
        for config_name in config_names:
            for model_name in model_names:
                rec = train_one(model_name, config_name, device, args, seed=seed)
                records.append(rec)

    # 汇总训练参数
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.checkpoints_dir.parent / 'train_summary.csv'
    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'config', 'seed', 'n_vars', 'n_params', 'best_epoch', 'best_val_loss'],
                           extrasaction='ignore')
        w.writeheader()
        w.writerows(records)
    print(f'\n训练参数汇总已保存: {summary_csv}')

    print('\n' + '=' * 74)
    print(f'{"模型":<22}{"配置":<28}{"种子":>5}{"参数量":>10}{"最优轮次":>9}{"验证损失":>12}')
    for r in records:
        print(f'{MODEL_DISPLAY[r["model"]]:<22}{CONFIG_DISPLAY[r["config"]]:<28}'
              f'{r["seed"]:>5}{r["n_params"]:>10,}{r["best_epoch"]:>9}{r["best_val_loss"]:>12.6f}')
    print('=' * 74)
    print('训练完成。下一步运行 test.py 评估指标（MAE/RMSE/R2/TCR）。')


if __name__ == '__main__':
    main()
