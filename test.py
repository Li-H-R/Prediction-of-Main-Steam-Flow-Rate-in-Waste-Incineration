"""实验1 · 测试入口：加载已训练权重，用提出的四项指标评估。

指标（在标准化 z-score 后的值上计算）：
    MAE   平均绝对误差
    RMSE  均方根误差
    R2    决定系数（解释方差比例）
    TCR   Trend Consistency Rate（趋势一致率，相邻步涨跌方向一致的比例）

用法：
    python test.py                          # 评估 checkpoints/ 下所有已训练模型
    python test.py --checkpoints-dir checkpoints

输出：
    test_summary.csv   每组的 MAE / RMSE / R2 / TCR 汇总
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import build_model, MODEL_DISPLAY, MODEL_OUTPUT_MODE  # noqa: E402
from train import build_config_data, CONFIG_DISPLAY, EXCLUDE_MODELS  # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# 评估时排除的配置（不评估、不写入 test_summary.csv）
EXCLUDE_CONFIGS = {'paper_plus_primary'}


# ═══════════════════════════════════════════════════════════════════════════════
# 指标
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算 MAE / RMSE / R2 / TCR（在传入的量纲上计算，此处为标准化后的值）。

    :param y_true: (N, pred_len) 真值
    :param y_pred: (N, pred_len) 预测值
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ft = y_true.ravel()
    fp = y_pred.ravel()

    mae = float(np.mean(np.abs(ft - fp)))
    mse = float(np.mean((ft - fp) ** 2))
    rmse = float(np.sqrt(mse))

    ss_tot = float(np.sum((ft - np.mean(ft)) ** 2))
    r2 = float(1.0 - np.sum((ft - fp) ** 2) / ss_tot) if ss_tot > 0 else float('nan')

    # TCR：预测窗口内相邻步涨跌方向一致的比例
    dt = np.diff(y_true, axis=1)
    dp = np.diff(y_pred, axis=1)
    tcr = float(np.mean(np.sign(dt) == np.sign(dp))) if dt.size > 0 else float('nan')

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'TCR': tcr}


# ═══════════════════════════════════════════════════════════════════════════════
# RTF 在线自适应匹配（Adaptive Time Matching）
# ═══════════════════════════════════════════════════════════════════════════════
# 论文 RTF 的核心：预测期分段，逐段在 {离线 AS, 在线 Ridge} 中选验证 MAE 最小者。
# 按用户要求：回看历史 180 个样本，前 90 个用于在线 Ridge 拟合、后 90 个用于验证选模型；
# 历史不足 180 时回退离线 AS。所有参数固定，不做调参。

RTF_HISTORY = 180        # 自适应匹配回看的历史总样本数
RTF_FIT_SAMPLES = 90     # 前 90 个样本用于在线 Ridge 拟合；后 90 个用于验证（AS/Ridge 比 MAE）
RTF_SEGMENT = 180        # 预测期分段大小（每段选一次模型）
RTF_RIDGE_ALPHA = 1.0    # 在线 Ridge 正则系数


def _rtf_predict_batched(model, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    """AS 模型批量预测：X (n, seq_len, n_vars) -> (n, pred_len)。fp32（RTF 的 FFT 禁用 AMP）。"""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            out = model(xb)
            outs.append(out.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def _rtf_adaptive_predict(model, matrix_std: np.ndarray, val_end: int,
                          seq_len: int, pred_len: int, device: torch.device,
                          batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """RTF 自适应匹配预测，返回 (y_true, y_pred)，形状均为 (n_windows, pred_len)。

    matrix_std: 全量时序标准化矩阵 (N, n_vars)，主蒸汽流量在第 0 列。
    val_end:    测试集首个预测原点（全局行索引）。预测原点 tau 的输入为
                [tau-seq_len, tau)，目标为 [tau, tau+pred_len) 的第 0 列。
    """
    from sklearn.linear_model import Ridge   # 延迟导入，避免影响其它模型评估

    N = matrix_std.shape[0]
    origins = np.arange(val_end, N - pred_len + 1)
    n_w = origins.shape[0]

    def win_x(o):
        return matrix_std[o - seq_len: o]

    def win_y(o):
        return matrix_std[o: o + pred_len, 0]

    y_true = np.zeros((n_w, pred_len), dtype=np.float32)
    y_pred = np.zeros((n_w, pred_len), dtype=np.float32)

    for s in range(0, n_w, RTF_SEGMENT):
        e = min(s + RTF_SEGMENT, n_w)
        hist_origins = origins[max(0, s - RTF_HISTORY): s]   # 前 180 个历史原点
        seg_origins = origins[s:e]

        # 历史不足 RTF_HISTORY 个样本 → 回退离线 AS（历史训练模型）
        if hist_origins.shape[0] < RTF_HISTORY:
            chosen = 'AS'
            ridge = None
        else:
            # 前 90 拟合 Ridge，后 90 验证（AS 与 Ridge 在验证段比 MAE）
            fit_origins = hist_origins[:RTF_FIT_SAMPLES]
            val_origins = hist_origins[RTF_FIT_SAMPLES:]
            Xf = np.stack([win_x(o) for o in fit_origins], axis=0)   # (90, seq_len, n_vars)
            yf = np.stack([win_y(o) for o in fit_origins], axis=0)   # (90, pred_len)
            ridge = Ridge(alpha=RTF_RIDGE_ALPHA).fit(Xf.reshape(Xf.shape[0], -1), yf)

            Xv = np.stack([win_x(o) for o in val_origins], axis=0)   # (90, seq_len, n_vars)
            yv = np.stack([win_y(o) for o in val_origins], axis=0)   # (90, pred_len)
            as_mae = float(np.mean(np.abs(_rtf_predict_batched(model, Xv, device, batch_size) - yv)))
            ridge_mae = float(np.mean(np.abs(ridge.predict(Xv.reshape(Xv.shape[0], -1)) - yv)))
            chosen = 'AS' if as_mae <= ridge_mae else 'Ridge'

        Xseg = np.stack([win_x(o) for o in seg_origins], axis=0)
        yseg = np.stack([win_y(o) for o in seg_origins], axis=0)
        if chosen == 'Ridge' and ridge is not None:
            seg_pred = ridge.predict(Xseg.reshape(Xseg.shape[0], -1))
        else:
            seg_pred = _rtf_predict_batched(model, Xseg, device, batch_size)

        y_true[s:e] = yseg
        y_pred[s:e] = seg_pred

    return y_true, y_pred


# ═══════════════════════════════════════════════════════════════════════════════
# 评估单个 checkpoint
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_checkpoint(ckpt_path: Path, device: torch.device, seq_len: int, pred_len: int,
                        batch_size: int) -> dict | None:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_name = ckpt['model_name']
    config_name = ckpt['config_name']
    if config_name in EXCLUDE_CONFIGS or model_name in EXCLUDE_MODELS:
        return None
    if ckpt.get('seed') is None:
        return None   # 旧格式（单种子、无 seed 字段）checkpoint，跳过以避免与多种子结果混算
    n_vars = ckpt['n_vars']
    output_mode = MODEL_OUTPUT_MODE[model_name]

    # 重建模型（与训练时相同架构）
    model = build_model(model_name, n_vars=n_vars, seq_len=seq_len, pred_len=pred_len)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 重建测试集（与训练时同一套 70/15/15 切分，保证测试集一致）
    data = build_config_data(config_name, seq_len, pred_len, batch_size)

    if model_name == 'RTF':
        # RTF：走 Adaptive Time Matching（离线 AS + 在线 Ridge 分段选最优）
        y_true, y_pred = _rtf_adaptive_predict(
            model, data['matrix_std'], data['val_end'], seq_len, pred_len, device, batch_size)
    else:
        test_loader = data['test_loader']
        all_true, all_pred = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    pred = model(x)
                # 目标恒为主蒸汽流量（channel 0）；通道独立模型预测所有通道，需再取第 0 列
                y_slice = y[:, :, 0]
                p_slice = pred[:, :, 0] if output_mode == 'multi' else pred
                all_true.append(y_slice.cpu().numpy())
                all_pred.append(p_slice.float().cpu().numpy())
        y_true = np.concatenate(all_true, axis=0)
        y_pred = np.concatenate(all_pred, axis=0)

    # 指标直接在标准化（z-score）后的值上计算
    metrics = compute_metrics(y_true, y_pred)
    metrics.update({
        'model': model_name,
        'config': config_name,
        'seed': ckpt.get('seed'),
        'best_epoch': ckpt.get('best_epoch', None),
        'n_params': ckpt.get('n_params', None),
    })
    return metrics


def aggregate(rows: list[dict]) -> list[dict]:
    """按 (model, config) 分组，对 MAE/RMSE/R2/TCR 计算跨种子均值±标准差，返回汇总行。"""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r['model'], r['config']), []).append(r)

    metrics = ['MAE', 'RMSE', 'R2', 'TCR']
    summary = []
    for (model, config) in sorted(groups):
        g = groups[(model, config)]
        row = {'model': model, 'config': config, 'n_seeds': len(g)}
        row['n_params'] = g[0]['n_params']
        for m in metrics:
            vals = np.asarray([r[m] for r in g], dtype=float)
            row[m] = float(vals.mean())
            row[f'{m}_std'] = float(vals.std())
        be = [r['best_epoch'] for r in g if r.get('best_epoch') is not None]
        row['best_epoch'] = round(float(np.mean(be))) if be else None
        summary.append(row)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='实验1：测试已训练模型（MAE/RMSE/R2/TCR）')
    p.add_argument('--checkpoints-dir', type=Path, default=HERE / 'checkpoints')
    p.add_argument('--seq-len', type=int, default=60)
    p.add_argument('--pred-len', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt_files = sorted(args.checkpoints_dir.glob('*.pth'))
    if not ckpt_files:
        print(f'未在 {args.checkpoints_dir} 找到任何 .pth 文件。请先运行 train.py。')
        return

    print(f'Device: {device} | 待评估 checkpoint: {len(ckpt_files)} 个')

    rows = []
    n_skipped = 0
    for ckpt_path in ckpt_files:
        r = evaluate_checkpoint(ckpt_path, device, args.seq_len, args.pred_len, args.batch_size)
        if r is None:
            n_skipped += 1
            continue
        rows.append(r)
        print(f'  [{MODEL_DISPLAY[r["model"]]:<22} {CONFIG_DISPLAY[r["config"]]:<28} '
              f'seed={r["seed"]}] MAE={r["MAE"]:.4f} RMSE={r["RMSE"]:.4f} '
              f'R2={r["R2"]:.4f} TCR={r["TCR"]:.4f}')
    if n_skipped:
        print(f'[跳过] {n_skipped} 个 checkpoint（被排除的模型/配置，或旧格式无 seed 字段）。')

    if not rows:
        print('没有可评估的 checkpoint。')
        return

    summary = aggregate(rows)
    summary_csv = args.checkpoints_dir.parent / 'test_summary.csv'
    fieldnames = ['model', 'config', 'n_seeds',
                  'MAE', 'MAE_std', 'RMSE', 'RMSE_std', 'R2', 'R2_std', 'TCR', 'TCR_std',
                  'best_epoch', 'n_params']
    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(summary)

    print('\n' + '=' * 108)
    print(f'{"模型":<22}{"配置":<26}{"n":>3}   '
          f'{"MAE":>16}{"RMSE":>16}{"R2":>16}{"TCR":>16}')
    print('-' * 108)
    for r in summary:
        print(f'{MODEL_DISPLAY[r["model"]]:<22}{CONFIG_DISPLAY[r["config"]]:<26}{r["n_seeds"]:>3}   '
              f'{r["MAE"]:.4f}±{r["MAE_std"]:.4f}  '
              f'{r["RMSE"]:.4f}±{r["RMSE_std"]:.4f}  '
              f'{r["R2"]:.4f}±{r["R2_std"]:.4f}  '
              f'{r["TCR"]:.4f}±{r["TCR_std"]:.4f}')
    print('=' * 108)
    print(f'测试指标汇总（每个 (模型, 配置) 跨种子均值±标准差）已保存: {summary_csv}')


if __name__ == '__main__':
    main()
