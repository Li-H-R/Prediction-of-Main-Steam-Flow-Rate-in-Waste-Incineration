"""实验1 · 全部预测模型定义（统一单目标接口）。

本文件包含实验1（多模型 × 多变量配置比较）所需的 10 个预测模型：

    PatchTST       通道独立 PatchTST（channel independence，共享编码器）
    CTPatchTST     通道注意力 + PatchTST（单层跨变量交互）
    LSTM_mv        Multivariate-input Single-target LSTM（多变量输入单目标）
    LSTM_ci        Channel-Independent LSTM（通道独立共享 LSTM）
    iTransformer   倒置 Transformer（每个变量整条序列作为一个 token）
    Transformer    经典 Transformer（时间维注意力，均值池化后预测）
    Crossformer    维度-段嵌入 + 跨时间/跨维度两阶段注意力（多变量联合预测）
    Crossformer_full  完整版 Crossformer（cross_models 原版：DSW 嵌入 + 多尺度编码器 + 路由器两阶段注意力 + 分层解码器）
    RTF            RTF 的 AS 核心 = Scaleformer(多尺度) + Autoformer
    MLP            全连接神经网络（最近 10 步拼接后直接映射到预测）
    LSTM_iTransformer  共享 LSTM 逐变量编码 + 变量维自注意力（iTransformer 的改进）

统一接口（train.py / test.py 无差别调用）：

    forward(x) -> (B, pred_len, n_vars)   多变量输出模型（PatchTST / CTPatchTST / LSTM_ci / Crossformer）：
                                           预测所有通道未来，输出维度 = 输入变量数；
    forward(x) -> (B, pred_len)           单目标类模型（LSTM_mv / LSTM_iTransformer / iTransformer / Transformer / RTF / MLP）：
                                           直接输出主蒸汽流量未来 pred_len 步。
        x   : (B, seq_len, n_vars) 标准化后的历史窗口，第 0 列恒为主蒸汽流量
        输出: 最终测试只取第 0 列（主蒸汽流量）计算指标（见 MODEL_OUTPUT_MODE）。

说明：
  * 除 RTF 外，其余模型都做「直接预测」（不使用残差/增量），保证跨模型公平可比；
  * RTF 复用 2026最新论文方法/src 中的 Autoformer / Scaleformer（此处内联，避免跨目录
    相对导入问题），作为 RTF 框架的深度预测核心（AS = Scaleformer + Autoformer）。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 使 cross_models（完整版 Crossformer 依赖的原版实现）在本文件被直接 import / 运行时也可导入
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# 模型注册表
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_NAMES = [
    'PatchTST', 'CTPatchTST', 'LSTM_mv', 'LSTM_ci',
    'iTransformer', 'Transformer', 'Crossformer', 'Crossformer_full',
    'RTF', 'MLP', 'LSTM_iTransformer',
]

MODEL_DISPLAY = {
    'PatchTST': 'PatchTST',
    'CTPatchTST': 'CT-PatchTST',
    'LSTM_mv': 'Multivariate-LSTM',
    'LSTM_ci': 'Channel-Independent-LSTM',
    'iTransformer': 'iTransformer',
    'Transformer': 'Transformer',
    'Crossformer': 'Crossformer',
    'Crossformer_full': 'Crossformer (full)',
    'RTF': 'RTF',
    'MLP': 'MLP',
    'LSTM_iTransformer': 'LSTM-iTransformer',
}

# 输出模式：
#   'multi'  = 通道独立类模型，每通道各预测自身未来，输出 (B, pred_len, n_vars)；
#   'single' = 单目标类模型，直接输出主蒸汽流量 (B, pred_len)。
# 仅 PatchTST / CTPatchTST / Channel-Independent-LSTM / Crossformer 为 multi，最终测试只取第 0 通道（主蒸汽流量）。
MODEL_OUTPUT_MODE = {
    'PatchTST': 'multi',
    'CTPatchTST': 'multi',
    'LSTM_mv': 'single',
    'LSTM_ci': 'multi',
    'Crossformer': 'multi',
    'Crossformer_full': 'multi',
    'iTransformer': 'single',
    'Transformer': 'single',
    'RTF': 'single',
    'MLP': 'single',
    'LSTM_iTransformer': 'single',
}

# 每个模型的架构超参数默认值（训练设置——seq_len/pred_len/切分/lr/epochs/patience——统一）
ARCH_DEFAULTS = {
    'PatchTST': dict(patch_len=6, stride=6, d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1),
    'CTPatchTST': dict(patch_len=6, stride=6, d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1, channel_heads=1),
    'LSTM_mv': dict(hidden_size=64, num_layers=2, dropout=0.1),
    'LSTM_ci': dict(hidden_size=64, num_layers=2, dropout=0.1),
    'iTransformer': dict(d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1),
    'Transformer': dict(d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1),
    'Crossformer': dict(seg_len=6, d_model=64, n_heads=4, e_layers=2, d_ff=256, dropout=0.1),
    'Crossformer_full': dict(seg_len=6, win_size=2, factor=10, d_model=64, n_heads=4,
                             e_layers=2, d_ff=256, dropout=0.1),
    'RTF': dict(d_model=64, n_heads=8, d_ff=256, encoder_layers=2, decoder_layers=1,
                dropout=0.1, moving_avg_kernel=25, factor=1, scales=(1, 2)),
    'MLP': dict(n_steps=10, hidden_dims=(256, 128), dropout=0.1),
    'LSTM_iTransformer': dict(hidden_size=64, lstm_layers=2, d_model=128, n_heads=4,
                              n_layers=3, d_ff=256, dropout=0.1),
}


def build_model(name: str, n_vars: int, seq_len: int, pred_len: int, **overrides) -> nn.Module:
    """按名称构建模型，统一注入 n_vars / seq_len / pred_len。

    :param name: 模型名（见 MODEL_NAMES）
    :param n_vars: 输入变量（通道）数
    :param seq_len: 历史窗口长度
    :param pred_len: 预测步长
    :param overrides: 覆盖对应模型的架构超参数
    """
    kw = dict(ARCH_DEFAULTS[name])
    kw.update(overrides)

    if name == 'PatchTST':
        return PatchTST(seq_len=seq_len, pred_len=pred_len, n_vars=n_vars, **kw)
    if name == 'CTPatchTST':
        return CTPatchTST(seq_len=seq_len, pred_len=pred_len, n_vars=n_vars, **kw)
    if name == 'LSTM_mv':
        return SingleTargetLSTM(n_vars=n_vars, pred_len=pred_len, **kw)
    if name == 'LSTM_ci':
        return ChannelIndependentLSTM(n_vars=n_vars, pred_len=pred_len, **kw)
    if name == 'iTransformer':
        return iTransformer(n_vars=n_vars, seq_len=seq_len, pred_len=pred_len, **kw)
    if name == 'Transformer':
        return TransformerModel(n_vars=n_vars, seq_len=seq_len, pred_len=pred_len, **kw)
    if name == 'Crossformer':
        return Crossformer(n_vars=n_vars, seq_len=seq_len, pred_len=pred_len, **kw)
    if name == 'Crossformer_full':
        return CrossformerFull(n_vars=n_vars, seq_len=seq_len, pred_len=pred_len, **kw)
    if name == 'RTF':
        kw.setdefault('label_len', seq_len // 2)
        return RTF(n_vars=n_vars, seq_len=seq_len, pred_len=pred_len, **kw)
    if name == 'MLP':
        return MLP(n_vars=n_vars, seq_len=seq_len, pred_len=pred_len, **kw)
    if name == 'LSTM_iTransformer':
        return LSTM_iTransformer(n_vars=n_vars, seq_len=seq_len, pred_len=pred_len, **kw)
    raise ValueError(f'未知模型名: {name}，可选 {MODEL_NAMES}')


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PatchTST（通道独立，只输出主蒸汽流量通道）
# ═══════════════════════════════════════════════════════════════════════════════

class PatchTST(nn.Module):
    """PatchTST (Channel Independence)：每个变量独立切 patch，共享同一 Transformer 与预测头。

    Input (B, T, V) -> 逐通道 unfold 切 patch -> (B*V, P, patch_len) -> 共享投影/编码/头
        -> (B, V, pred_len) -> 取第 0 列（主蒸汽流量）-> (B, pred_len)
    """

    def __init__(self, seq_len=60, pred_len=5, n_vars=1, patch_len=6, stride=6,
                 d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        if self.num_patches <= 0:
            raise ValueError(f'Invalid patch config: seq_len={seq_len}, patch_len={patch_len}')

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)

        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, 'gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(self.num_patches * d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x):
        B, T, V = x.shape
        x = x.permute(0, 2, 1).reshape(B * V, T)              # (B*V, T)
        x = x.unfold(-1, self.patch_len, self.stride)         # (B*V, P, patch_len)
        x = self.patch_proj(x)                                # (B*V, P, d)
        x = self.drop(x + self.pos)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.reshape(B * V, -1)                              # (B*V, P*d)
        out = self.head(x)                                    # (B*V, pred_len)
        out = out.reshape(B, V, -1)                           # (B, V, pred_len)
        return out.permute(0, 2, 1)                           # (B, pred_len, V)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CT-PatchTST（单层通道注意力 + PatchTST）
# ═══════════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """单层通道注意力：在每个 patch 位置对变量维度做多头注意力（Pre-LN + 残差）。

    输入 (B, V, P, D) -> 输出 (B, V, P, D)
    """

    def __init__(self, d_model, n_heads=1, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, f'd_model({d_model}) 需能被 n_heads({n_heads}) 整除'
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, V, P, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * P, V, D)        # (B*P, V, D)
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed)       # (B*P, V, D)
        x = x + self.drop(attn_out)
        return x.reshape(B, P, V, D).permute(0, 2, 1, 3)      # (B, V, P, D)


class CTPatchTST(nn.Module):
    """CT-PatchTST：先切 patch，再做一层跨变量通道注意力，最后经典 PatchTST 时间编码。

    Input (B, T, V) -> 逐通道 patch -> (B, V, P, patch_len) -> 共享投影 + 位置编码
        -> 1 层 ChannelAttention（跨变量）-> reshape(B*V, P, d) -> 时间 Transformer
        -> 共享头 -> (B, V, pred_len) -> 取第 0 列 -> (B, pred_len)
    """

    def __init__(self, seq_len=60, pred_len=5, n_vars=1, patch_len=6, stride=6,
                 d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1, channel_heads=1):
        super().__init__()
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.num_patches = (seq_len - patch_len) // stride + 1
        if self.num_patches <= 0:
            raise ValueError(f'Invalid patch config: seq_len={seq_len}, patch_len={patch_len}')

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, 1, self.num_patches, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.channel_attn = ChannelAttention(d_model, channel_heads, dropout)

        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, 'gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(self.num_patches * d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x):
        B, T, V = x.shape
        x = x.permute(0, 2, 1)                                # (B, V, T)
        patches = x.unfold(-1, self.patch_len, self.stride)   # (B, V, P, patch_len)
        B, V, P, pl = patches.shape
        patches = patches.reshape(B * V, P, pl)
        tokens = self.patch_proj(patches)                     # (B*V, P, d)
        tokens = tokens.reshape(B, V, P, self.d_model)
        tokens = self.drop(tokens + self.pos)                 # (B, V, P, d)
        tokens = self.channel_attn(tokens)                    # 跨变量交互
        tokens = tokens.reshape(B * V, P, self.d_model)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        flat = tokens.reshape(B * V, -1)
        out = self.head(flat)                                 # (B*V, pred_len)
        out = out.reshape(B, V, -1)                           # (B, V, pred_len)
        return out.permute(0, 2, 1)                           # (B, pred_len, V)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Multivariate-input Single-target LSTM
# ═══════════════════════════════════════════════════════════════════════════════

class SingleTargetLSTM(nn.Module):
    """多变量输入 -> 单目标 LSTM：(B, T, V) -> LSTM(input_size=V) -> last hidden -> 头 -> (B, pred_len)."""

    def __init__(self, n_vars=1, pred_len=5, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_vars, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len),
        )

    def forward(self, x):
        out, _ = self.lstm(x)           # (B, T, hidden)
        return self.head(out[:, -1, :])  # (B, pred_len)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Channel-Independent LSTM
# ═══════════════════════════════════════════════════════════════════════════════

class ChannelIndependentLSTM(nn.Module):
    """通道独立 LSTM：每个变量独立进共享 LSTM(input_size=1)，各自预测自身未来，取第 0 列。"""

    def __init__(self, n_vars=1, pred_len=5, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len),
        )

    def forward(self, x):
        B, T, V = x.shape
        x = x.permute(0, 2, 1).reshape(B * V, T, 1)   # (B*V, T, 1)
        out, _ = self.lstm(x)                         # (B*V, T, hidden)
        h = out[:, -1, :]                             # (B*V, hidden)
        out = self.head(h)                            # (B*V, pred_len)
        out = out.reshape(B, V, -1)                   # (B, V, pred_len)
        return out.permute(0, 2, 1)                   # (B, pred_len, V)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. iTransformer（倒置 Transformer：每个变量整条序列作为一个 token）
# ═══════════════════════════════════════════════════════════════════════════════

class iTransformer(nn.Module):
    """iTransformer：把 (B, T, V) 转置为 (B, V, T)，每个变量整条时间序列经线性嵌入为一个 token，
    在变量维度上做自注意力，最后每个 token 线性投影到 pred_len，取第 0 列（主蒸汽流量）。"""

    def __init__(self, n_vars=1, seq_len=60, pred_len=5, d_model=128, n_heads=4,
                 n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.embed = nn.Linear(seq_len, d_model)      # 每条变量序列 -> token
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, 'gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len)      # 每个 token -> pred_len

    def forward(self, x):
        B, T, V = x.shape
        x = x.permute(0, 2, 1)                        # (B, V, T)
        h = self.embed(x)                             # (B, V, d)
        h = self.encoder(h)
        h = self.norm(h)
        out = self.head(h)                            # (B, V, pred_len)
        return out[:, 0, :]                           # (B, pred_len)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Transformer（经典：时间维注意力，均值池化后预测单目标）
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerModel(nn.Module):
    """经典 Transformer 编码器：线性嵌入 (T,V)->(T,d)，加位置编码，时间维自注意力，均值池化 -> 头。"""

    def __init__(self, n_vars=1, seq_len=60, pred_len=5, d_model=128, n_heads=4,
                 n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(n_vars, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, 'gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x):
        h = self.embed(x) + self.pos                    # (B, T, d)
        h = self.drop(h)
        h = self.encoder(h)
        h = self.norm(h)
        h = h.mean(dim=1)                               # (B, d) 时间维均值池化
        return self.head(h)                             # (B, pred_len)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RTF（Scaleformer + Autoformer）—— 内联自 2026最新论文方法/src
# ═══════════════════════════════════════════════════════════════════════════════
#
# 以下 Autoformer / Scaleformer 实现与 2026最新论文方法/src/{autoformer,scaleformer}.py
# 完全一致（此处内联以规避跨目录相对导入），作为 RTF 框架的深度预测核心（AS 模型）。

class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        pad = (self.kernel_size - 1) // 2
        front = x[:, :, 0:1].repeat(1, 1, pad)
        end = x[:, :, -1:].repeat(1, 1, pad)
        return self.avg(torch.cat([front, x, end], dim=2))


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x - trend, trend


class AutoCorrelation(nn.Module):
    def __init__(self, factor: int = 1):
        super().__init__()
        self.factor = factor

    def time_delay_agg(self, values, corr):
        B, H, L, E = values.shape
        top_k = max(1, int(self.factor * math.log(L)))
        weights = torch.softmax(corr, dim=-1)
        corr_profile = weights.mean(dim=1)
        top_w, top_idx = torch.topk(corr_profile, top_k, dim=1)
        delay_agg = torch.zeros_like(values)
        arange = torch.arange(L, device=values.device)
        for i in range(top_k):
            lag = top_idx[:, i]
            w = top_w[:, i]
            idx = (arange.unsqueeze(0) - lag.unsqueeze(1)) % L
            idx = idx.view(B, 1, L, 1).expand(-1, H, -1, E)
            delay_agg += torch.gather(values, 2, idx) * w.view(B, 1, 1, 1)
        return delay_agg

    def forward(self, q, k, v, n_heads):
        B, L, D = q.shape
        S = k.shape[1]
        if L > S:
            pad = torch.zeros(B, L - S, D, device=k.device, dtype=k.dtype)
            k = torch.cat([k, pad], dim=1)
            v = torch.cat([v, pad], dim=1)
        elif L < S:
            k = k[:, :L, :]
            v = v[:, :L, :]
        H, E = n_heads, D // n_heads
        qh = q.reshape(B, L, H, E).permute(0, 2, 1, 3)
        kh = k.reshape(B, L, H, E).permute(0, 2, 1, 3)
        vh = v.reshape(B, L, H, E).permute(0, 2, 1, 3)
        q_fft = torch.fft.rfft(qh, dim=2)
        k_fft = torch.fft.rfft(kh, dim=2)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=L, dim=2)
        corr = corr.mean(dim=-1)
        out = self.time_delay_agg(vh, corr)
        return out.permute(0, 2, 1, 3).reshape(B, L, D)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model, n_heads, factor=1, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.autocorr = AutoCorrelation(factor=factor)

    def forward(self, q, k, v):
        Q, K, V = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        out = self.autocorr(Q, K, V, self.n_heads)
        return self.dropout(self.out_proj(out))


class _EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, factor, dropout, moving_avg_kernel):
        super().__init__()
        self.decomp1 = SeriesDecomp(moving_avg_kernel)
        self.decomp2 = SeriesDecomp(moving_avg_kernel)
        self.attn = AutoCorrelationLayer(d_model, n_heads, factor, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        seasonal, trend = self.decomp1(x + self.attn(x, x, x))
        seasonal, trend2 = self.decomp2(seasonal + self.ffn(seasonal))
        return seasonal, trend + trend2


class _DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, factor, dropout, moving_avg_kernel):
        super().__init__()
        self.decomp1 = SeriesDecomp(moving_avg_kernel)
        self.decomp2 = SeriesDecomp(moving_avg_kernel)
        self.decomp3 = SeriesDecomp(moving_avg_kernel)
        self.self_attn = AutoCorrelationLayer(d_model, n_heads, factor, dropout)
        self.cross_attn = AutoCorrelationLayer(d_model, n_heads, factor, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, cross, trend):
        seasonal, trend1 = self.decomp1(x + self.self_attn(x, x, x))
        seasonal, trend2 = self.decomp2(seasonal + self.cross_attn(seasonal, cross, cross))
        seasonal, trend3 = self.decomp3(seasonal + self.ffn(seasonal))
        return seasonal, trend + trend1 + trend2 + trend3


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout, use_pos, max_len=2048):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.use_pos = use_pos
        self.position_embedding = nn.Embedding(max_len, d_model) if use_pos else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.value_embedding(x)
        if self.use_pos:
            L = x.shape[1]
            pos = torch.arange(L, device=x.device).unsqueeze(0).expand(x.shape[0], L)
            out = out + self.position_embedding(pos)
        return self.dropout(out)


class Autoformer(nn.Module):
    """输入 [B, L, c_in] -> 输出 [B, pred_len, c_out]。"""

    def __init__(self, c_in, c_out, seq_len, pred_len, d_model=64, n_heads=8, d_ff=256,
                 encoder_layers=2, decoder_layers=1, dropout=0.1, moving_avg_kernel=25,
                 factor=1, label_len=150, use_pos=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = min(label_len, seq_len // 2)
        self.c_out = c_out

        self.decomp = SeriesDecomp(moving_avg_kernel)
        self.enc_embedding = DataEmbedding(c_in, d_model, dropout, use_pos)
        self.dec_embedding = DataEmbedding(c_in, d_model, dropout, use_pos)
        self.encoder = nn.ModuleList([
            _EncoderLayer(d_model, n_heads, d_ff, factor, dropout, moving_avg_kernel)
            for _ in range(encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            _DecoderLayer(d_model, n_heads, d_ff, factor, dropout, moving_avg_kernel)
            for _ in range(decoder_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.trend_proj = nn.Linear(c_in, d_model)

    def forward(self, x_enc):
        B = x_enc.shape[0]
        device = x_enc.device
        seasonal_init, trend_init = self.decomp(x_enc)
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(B, self.pred_len, x_enc.shape[2], device=device)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        enc_out = self.enc_embedding(x_enc)
        for layer in self.encoder:
            enc_out, _ = layer(enc_out)

        dec_out = self.dec_embedding(seasonal_init)
        trend_part = self.trend_proj(trend_init)
        for layer in self.decoder:
            dec_out, trend_part = layer(dec_out, enc_out, trend_part)

        dec_out = self.norm(dec_out + trend_part)
        out = self.projection(dec_out)
        return out[:, -self.pred_len:, :]


def _downsample_time(x, s):
    if s == 1:
        return x
    B, L, C = x.shape
    xp = x.permute(0, 2, 1)
    xp = F.avg_pool1d(xp, kernel_size=s, stride=s)
    return xp.permute(0, 2, 1)


def _upsample_time(x, target_len):
    B, H, C = x.shape
    if H == target_len:
        return x
    xp = x.permute(0, 2, 1).float()
    xp = F.interpolate(xp, size=target_len, mode='linear', align_corners=False)
    return xp.permute(0, 2, 1)


class Scaleformer(nn.Module):
    """多尺度外壳：输入按多个时间尺度下采样，各自用 Autoformer 预测后上采样并平均。"""

    def __init__(self, c_in, c_out, seq_len, pred_len, scales=(1, 2), **autoformer_kwargs):
        super().__init__()
        self.scales = tuple(scales)
        self.pred_len = pred_len
        self.models = nn.ModuleDict()
        label_len = autoformer_kwargs.pop('label_len', 150)
        for s in self.scales:
            L_s = max(1, seq_len // s)
            H_s = max(1, math.ceil(pred_len / s))
            kw = dict(autoformer_kwargs)
            kw['seq_len'] = L_s
            kw['pred_len'] = H_s
            kw['label_len'] = max(1, label_len // s)
            self.models[str(s)] = Autoformer(c_in, c_out, **kw)

    def forward(self, x):
        forecasts = []
        for s in self.scales:
            xs = _downsample_time(x, s)
            f_s = self.models[str(s)](xs)
            forecasts.append(_upsample_time(f_s, self.pred_len))
        return torch.stack(forecasts, dim=0).mean(dim=0)


class RTF(nn.Module):
    """RTF 的深度预测核心（AS = Scaleformer + Autoformer），输出单目标主蒸汽流量。

    Input (B, T, V) -> Scaleformer(c_in=V, c_out=1) -> (B, pred_len, 1) -> squeeze -> (B, pred_len)
    """

    def __init__(self, n_vars=1, seq_len=60, pred_len=5, scales=(1, 2), d_model=64,
                 n_heads=8, d_ff=256, encoder_layers=2, decoder_layers=1, dropout=0.1,
                 moving_avg_kernel=25, factor=1, label_len=None, use_pos=True):
        super().__init__()
        self.pred_len = pred_len
        if label_len is None:
            label_len = seq_len // 2
        self.model = Scaleformer(
            c_in=n_vars, c_out=1, seq_len=seq_len, pred_len=pred_len, scales=tuple(scales),
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers,
            dropout=dropout, moving_avg_kernel=moving_avg_kernel, factor=factor,
            label_len=label_len, use_pos=use_pos,
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)   # (B, pred_len)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Crossformer（DSW 维度-段嵌入 + 两阶段注意力：跨时间 + 跨维度）
# ═══════════════════════════════════════════════════════════════════════════════

class _CrossformerTSA(nn.Module):
    """Crossformer 两阶段注意力（Two-Stage Attention）：先跨时间（每个变量内部各段之间），
    再跨维度（每个段位置上各变量之间），各自 Pre-LN + 残差，后接 FFN。"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.cross_time = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_dim = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_time = nn.LayerNorm(d_model)
        self.norm_dim = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, V, S, D)
        B, V, S, D = x.shape
        # 跨时间：每个变量内部，S 个段之间做自注意力
        xt = x.reshape(B * V, S, D)
        xt = self.cross_time(self.norm_time(xt), self.norm_time(xt), self.norm_time(xt))[0]
        x = x + self.drop(xt.reshape(B, V, S, D))
        # 跨维度：每个段位置上，V 个变量之间做自注意力
        xd = x.permute(0, 2, 1, 3).reshape(B * S, V, D)
        xd = self.cross_dim(self.norm_dim(xd), self.norm_dim(xd), self.norm_dim(xd))[0]
        x = x + self.drop(xd.reshape(B, S, V, D).permute(0, 2, 1, 3))
        # FFN
        x = x + self.drop(self.ffn(self.norm_ffn(x)))
        return x


class Crossformer(nn.Module):
    """Crossformer：DSW（维度-段）嵌入 + 堆叠两阶段注意力，预测所有通道，测试取第 0 列。

    Input (B, T, V) -> 按 seg_len 切段 -> (B, V, S, seg_len) -> 每段线性嵌入 + 段/维度位置编码
        -> e_layers 个 TSA（跨时间 + 跨维度 + FFN）-> 每变量聚合 S 个段 -> 头 -> (B, V, pred_len)
        -> permute -> (B, pred_len, V)（multi 模式，最终测试取第 0 列主蒸汽流量）。
    """

    def __init__(self, n_vars=1, seq_len=60, pred_len=5, seg_len=6, d_model=64,
                 n_heads=4, e_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.seg_len = seg_len
        self.num_seg = math.ceil(seq_len / seg_len)
        self.pad_len = self.num_seg * seg_len - seq_len

        self.value_embedding = nn.Linear(seg_len, d_model)
        self.seg_pos = nn.Parameter(torch.randn(1, self.num_seg, d_model) * 0.02)
        self.dim_embed = nn.Parameter(torch.randn(1, n_vars, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            _CrossformerTSA(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(self.num_seg * d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x):
        B, T, V = x.shape
        x = x.permute(0, 2, 1)                                  # (B, V, T)
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len))                     # 补齐到 num_seg*seg_len
        x = x.reshape(B, V, self.num_seg, self.seg_len)         # (B, V, S, seg_len)
        h = self.value_embedding(x)                             # (B, V, S, D)
        h = self.drop(h + self.seg_pos.unsqueeze(1) + self.dim_embed.unsqueeze(2))
        for layer in self.layers:
            h = layer(h)                                        # (B, V, S, D)
        h = self.norm(h)
        h = h.reshape(B, V, -1)                                 # (B, V, S*D)
        out = self.head(h)                                      # (B, V, pred_len)
        return out.permute(0, 2, 1)                             # (B, pred_len, V)


# ═══════════════════════════════════════════════════════════════════════════════
# 8b. Crossformer（完整版）—— 复用 cross_models 原版实现，与精简版 Crossformer 并列
# ═══════════════════════════════════════════════════════════════════════════════

class CrossformerFull(nn.Module):
    """完整版 Crossformer（cross_models/cross_former.py 原版，不替代上面的精简版）。

    与精简版 Crossformer 的区别：完整版含 DSW 分段嵌入、多尺度（SegMerging）编码器、
    路由器式两阶段注意力（cross-time + cross-dimension）、以及按尺度累加预测的分层解码器。
    输出所有通道（multi 模式），最终测试取第 0 列主蒸汽流量。

    Input (B, seq_len, n_vars) -> (B, pred_len, n_vars)。
    """

    def __init__(self, n_vars=1, seq_len=60, pred_len=5, seg_len=6, win_size=2,
                 factor=10, d_model=64, n_heads=4, e_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        # 延迟导入：避免缺少 einops / cross_models 时影响其它模型的加载
        from cross_models.cross_former import Crossformer as _FullCrossformer
        self.model = _FullCrossformer(
            data_dim=n_vars, in_len=seq_len, out_len=pred_len, seg_len=seg_len,
            win_size=win_size, factor=factor, d_model=d_model, d_ff=d_ff,
            n_heads=n_heads, e_layers=e_layers, dropout=dropout,
            baseline=False, device=torch.device('cpu'),
        )

    def forward(self, x):
        return self.model(x)                                   # (B, pred_len, n_vars)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MLP（全连接神经网络，所有变量拼接后直接映射到预测）
# ═══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """全连接神经网络（多层感知机）：取最近 n_steps 步，把所有变量拼接成一个向量，直接映射到预测。

    Input (B, T, V) -> 取最近 n_steps 步 -> (B, n_steps, V) -> reshape (B, n_steps*V)
        -> 多层全连接（ReLU + Dropout）-> (B, pred_len)
        （single 模式，直接输出主蒸汽流量未来 pred_len 步）。
    """

    def __init__(self, n_vars=1, seq_len=60, pred_len=5, n_steps=10, hidden_dims=(256, 128), dropout=0.1):
        super().__init__()
        self.n_steps = min(n_steps, seq_len)
        in_dim = self.n_steps * n_vars
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, pred_len))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x[:, -self.n_steps:, :]                  # 最近 n_steps 步
        return self.net(x.reshape(x.shape[0], -1))   # (B, pred_len)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. LSTM-iTransformer（共享 LSTM 逐变量编码 + 变量维自注意力）
# ═══════════════════════════════════════════════════════════════════════════════

class LSTM_iTransformer(nn.Module):
    """共享 LSTM 逐变量编码 + 变量维自注意力（iTransformer 的改进版）。

    先对每个变量独立用同一个 LSTM(input_size=1) 编码整条序列，取最后隐状态作为该变量的编码；
    再把 V 个变量的编码作为 token，在变量维做自注意力（同 iTransformer），最后线性头取第 0 列。

    Input (B, T, V) -> 共享 LSTM -> (B, V, hidden) -> proj -> (B, V, d) -> 变量维自注意力
        -> head -> (B, V, pred_len) -> 取第 0 列 (B, pred_len)。
    """

    def __init__(self, n_vars=1, seq_len=60, pred_len=5, hidden_size=64, lstm_layers=2,
                 d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        # 同一个 LSTM 对所有变量独立编码（channel-independent）
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=lstm_layers,
                            batch_first=True, dropout=(dropout if lstm_layers > 1 else 0.0))
        self.proj = nn.Linear(hidden_size, d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, 'gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len)      # 每个 token -> pred_len

    def forward(self, x):
        B, T, V = x.shape
        x = x.permute(0, 2, 1).reshape(B * V, T, 1)   # (B*V, T, 1)
        out, _ = self.lstm(x)                         # (B*V, T, hidden)
        h = out[:, -1, :]                             # (B*V, hidden) 每变量编码
        h = self.proj(h).reshape(B, V, -1)            # (B, V, d)
        h = self.encoder(h)                           # (B, V, d) 变量维自注意力
        h = self.norm(h)
        out = self.head(h)                            # (B, V, pred_len)
        return out[:, 0, :]                           # (B, pred_len)


# ═══════════════════════════════════════════════════════════════════════════════
# 自检：验证每个模型在前向 / 反向传播中形状正确
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('模型形状自检（CPU，随机输入）')
    for seq_len, pred_len in [(60, 5)]:
        for name in MODEL_NAMES:
            mode = MODEL_OUTPUT_MODE[name]
            for n_vars in [1, 7, 13]:
                model = build_model(name, n_vars=n_vars, seq_len=seq_len, pred_len=pred_len)
                x = torch.randn(4, seq_len, n_vars)
                y = model(x)
                expected = (4, pred_len, n_vars) if mode == 'multi' else (4, pred_len)
                assert y.shape == expected, f'{name} n_vars={n_vars}: {tuple(y.shape)}'
                (y ** 2).mean().backward()
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f'  {name:16s}[{mode:6s}] n_vars={n_vars:2d} -> {tuple(y.shape)}  params={n_params:,}')
    print('[OK] 所有模型形状自检通过')
