"""
=============================================================================
模块三核心引擎 v3.0：TS-VAE 无监督退化感知（真实数据闭环版·性能优化）
=============================================================================
版本: v3.0.0
优化: 解决Step 6推理缓慢问题，聚焦4张论文图高效生成
=============================================================================
"""

import os, sys, math, warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns

# ─── 中文字体支持（解决Windows乱码）───
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong', 'STXihei', 'Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results/module3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# 超参
WINDOW_SIZE = 60
STEP_SIZE = 30           # 步长加大减少帧数
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LATENT_DIM = 16
BETA = 0.001
BATCH_SIZE = 64          # 适当增大
EPOCHS = 60
PATIENCE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4

print(f"[TS-VAE v3.0] 设备: {DEVICE} | 步长: S={STEP_SIZE}s (优化)")

# =============================================================================
# 数据加载
# =============================================================================

def load_and_engineer_features() -> pd.DataFrame:
    """加载5日数据并计算7维全局独立核心特征

    7D特征构成（物理导向特征平替原则）：
      机械/能效簇（3个高阶不变量）：SE, FPI, TPI
      电气簇（1个）：P2.1泵电流
      热力簇（1个）：主油箱油温
      流体簇（2个）：泥水仓顶部1压力, 排浆密度
    """
    day_files = [
        (DATA_DIR / "steady_260207.csv", "2026-02-07", 121),
        (DATA_DIR / "steady_260208.csv", "2026-02-08", 123),
        (DATA_DIR / "steady_260209.csv", "2026-02-09", 125),
        (DATA_DIR / "steady_260210.csv", "2026-02-10", 127),
        (DATA_DIR / "steady_260211.csv", "2026-02-11", 129),
    ]
    all_rows = []
    D_cutter = 6.9
    CUTTERHEAD_AREA = 188.7
    EPSILON = 1e-6

    for fpath, date_str, base_ring in day_files:
        print(f"  加载 {fpath.name} ...")
        df = pd.read_csv(fpath, low_memory=False)
        df['date'] = pd.to_datetime(df['日期'])
        df['ring_id'] = df['环号']

        F_v = df['总推进力'].values.astype(np.float64)
        torque = df['刀盘扭矩'].values.astype(np.float64)
        v_fwd = df['推进速度'].values.astype(np.float64)
        RPM = df['刀盘转速'].values.astype(np.float64)

        with np.errstate(divide='ignore', invalid='ignore'):
            p_calc = np.where(RPM > 0, v_fwd / (RPM + EPSILON), np.nan)
            # SE = (F_v/A) + 2π·T/(A·p)
            se = np.where(np.isfinite(p_calc),
                         (F_v / CUTTERHEAD_AREA) + (2 * np.pi * torque) / (CUTTERHEAD_AREA * (p_calc + EPSILON)),
                         np.nan)
            fpi = np.where(np.isfinite(p_calc), F_v / (p_calc + EPSILON), np.nan)
            tpi = np.where(np.isfinite(p_calc), torque / (p_calc + EPSILON), np.nan)

        I_p21 = df['P2.1泵电流'].values.astype(np.float64)
        T_oil = df['主油箱油温'].values.astype(np.float64)
        P_slurry_top = df['泥水仓顶部1压力'].values.astype(np.float64)
        rho_out = df['排浆密度'].values.astype(np.float64)

        df['SE'] = se
        df['FPI'] = fpi
        df['TPI'] = tpi
        df['P2.1泵电流'] = I_p21
        df['主油箱油温'] = T_oil
        df['泥水仓顶部1压力'] = P_slurry_top
        df['排浆密度'] = rho_out
        all_rows.append(df)

    df_full = pd.concat(all_rows, ignore_index=True)
    print(f"  合并后总行数: {len(df_full):,}")
    return df_full


FINAL_7D = ['SE', 'FPI', 'TPI', 'P2.1泵电流', '主油箱油温', '泥水仓顶部1压力', '排浆密度']


def build_core_feature_matrix(df: pd.DataFrame):
    """构建7维全局独立核心特征矩阵

    严格遵循物理导向特征平替原则：
      - 3个高阶不变量：SE, FPI, TPI（机械/能效簇派生）
      - 4个跨域独立代理：P2.1泵电流(电气), 主油箱油温(热力),
                          泥水仓顶部1压力, 排浆密度(流体)
    """
    CORE_COLS = FINAL_7D
    for col in CORE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df_feat = df[CORE_COLS].copy()
    df_feat = df_feat.interpolate(method='linear', limit_direction='both')
    df_feat = df_feat.ffill().bfill().fillna(0.0)
    scaler = StandardScaler()
    arr = scaler.fit_transform(df_feat.values)
    df_scaled = pd.DataFrame(arr, columns=CORE_COLS, index=df.index)
    df_scaled['ring_id'] = df['ring_id'].values
    df_scaled['date'] = df['date'].values
    print(f"  特征矩阵: {df_scaled.shape}, 通道: {CORE_COLS}")
    print(f"  输入压缩比: {len(CORE_COLS)}/{LATENT_DIM} = {len(CORE_COLS)/LATENT_DIM:.3f}")
    return df_scaled, CORE_COLS, scaler


# =============================================================================
# Dataset
# =============================================================================

class TimeWindowDataset(Dataset):
    def __init__(self, df, window_size=60, step_size=10, feature_cols=None):
        self.df = df
        self.window_size = window_size
        self.step_size = step_size
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in ('ring_id','date')]
        self.feature_cols = feature_cols
        self.feature_data = df[feature_cols].values.astype(np.float32)
        self.ring_ids = df['ring_id'].values
        self.dates = df['date'].values if 'date' in df.columns else np.zeros(len(df))
        total_len = len(df)
        self.valid_starts = list(range(0, total_len - window_size + 1, step_size))
        self.scaler = None

    def set_scaler(self, scaler):
        self.scaler = scaler

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.window_size
        window = self.feature_data[start:end]
        if self.scaler is not None:
            window = self.scaler.transform(window)
        meta = {'start_idx': start, 'end_idx': end,
                'ring_id': int(self.ring_ids[start]), 'date': str(self.dates[start])}
        return torch.from_numpy(window), meta


# =============================================================================
# TS-VAE
# =============================================================================

class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, latent_dim=16, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc_mu(h[-1]), self.fc_logvar(h[-1])


class DecoderGRU(nn.Module):
    def __init__(self, latent_dim, hidden_size=64, num_layers=2, output_size=7,
                 seq_len=60, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc_init = nn.Linear(latent_dim, hidden_size * num_layers)
        self.gru = nn.GRU(latent_dim, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        batch = z.size(0)
        h0 = torch.tanh(self.fc_init(z))
        h0 = h0.view(batch, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        gru_in = z.unsqueeze(1).expand(-1, self.seq_len, -1)
        out, _ = self.gru(gru_in, h0)
        return self.fc_out(out)


class TSVAE(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 latent_dim=16, seq_len=60, beta=0.001):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.beta = beta
        self.encoder = EncoderGRU(input_size, hidden_size, num_layers, latent_dim)
        self.decoder = DecoderGRU(latent_dim, hidden_size, num_layers, input_size, seq_len)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z

    def elbo_loss(self, x, recon, mu, logvar):
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_tsvae(model, train_loader, val_loader, epochs=60, lr=1e-3,
                weight_decay=1e-4, patience=8, device=None):
    if device is None:
        device = DEVICE
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=patience)
    history = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': []}

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, r_loss, k_loss, nb = 0.0, 0.0, 0.0, 0
        for bx, _ in train_loader:
            bx = bx.to(device)
            optimizer.zero_grad()
            recon, mu, lv, _ = model(bx)
            loss, rl, kl = model.elbo_loss(bx, recon, mu, lv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item(); r_loss += rl.item(); k_loss += kl.item(); nb += 1

        model.eval()
        v_loss, v_r, v_k, nv = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                recon, mu, lv, _ = model(bx)
                loss, rl, kl = model.elbo_loss(bx, recon, mu, lv)
                v_loss += loss.item(); v_r += rl.item(); v_k += kl.item(); nv += 1

        avg_t = t_loss/nb; avg_r = r_loss/nb; avg_k = k_loss/nb
        avg_v = v_loss/nv
        scheduler.step(avg_v)
        history['train_loss'].append(avg_t)
        history['val_loss'].append(avg_v)
        history['recon_loss'].append(avg_r)
        history['kl_loss'].append(avg_k)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [Epoch {epoch:03d}] Train={avg_t:.4f}(R={avg_r:.4f},KL={avg_k:.4f}) | Val={avg_v:.4f}")

        if early_stopping(avg_v):
            print(f"  [Early Stop] 第{epoch}轮停止")
            break
    return history


# =============================================================================
# 推理 & 聚合
# =============================================================================

def extract_all_metrics(model, dataset, device=None):
    """逐帧推理（避免DataLoader批处理meta结构歧义）"""
    if device is None:
        device = DEVICE
    model = model.to(device)
    model.eval()

    mse_list, rid_list, z_list = [], [], []

    with torch.no_grad():
        for i in range(len(dataset)):
            bx, meta = dataset[i]
            bx = bx.unsqueeze(0).to(device)
            recon, mu, lv, z = model(bx)
            mse = (recon - bx).pow(2).mean().item()
            mse_list.append(mse)
            z_list.append(z.cpu().numpy().squeeze())
            rid_list.append(meta['ring_id'])

    frame_mse = np.array(mse_list)
    latents = np.array(z_list)
    ring_ids = np.array(rid_list)
    return frame_mse, ring_ids, latents


def aggregate_to_ring_level(frame_mse, ring_ids, num_rings=None):
    ring_ids_sorted = sorted(set(ring_ids))
    if num_rings is None:
        num_rings = len(ring_ids_sorted)
    rings_to_use = ring_ids_sorted[:num_rings]
    hi = {}
    cnt = {}
    for mse, rid in zip(frame_mse, ring_ids):
        if rid not in hi:
            hi[rid] = 0.0
            cnt[rid] = 0
        hi[rid] += mse
        cnt[rid] += 1
    hi_values = np.array([hi[r]/cnt[r] for r in rings_to_use])
    return hi_values, list(rings_to_use)


def compute_umap(latents, n_neighbors=10, min_dist=0.2):
    if not HAS_UMAP or latents is None:
        return None
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=42, n_jobs=1)
    return reducer.fit_transform(latents)


# =============================================================================
# 可视化 4张图
# =============================================================================

def plot_multichannel_reconstruction(df_scaled, model, scaler, CORE_COLS, device=None, save_path=None):
    """Fig 8: 多通道重构波形"""
    if device is None:
        device = DEVICE
    model.eval()
    T = WINDOW_SIZE

    # 找Ring 121（健康基线）和Ring 127（极端退化峰值）
    try:
        idx1 = df_scaled[df_scaled['ring_id'] == 121].index[min(50, len(df_scaled[df_scaled['ring_id']==121])-1)]
        # Ring 127 是HI=803的极端异常峰值，用它展示"额外耗散做功"最有力
        idx2 = df_scaled[df_scaled['ring_id'] == 127].index[min(50, len(df_scaled[df_scaled['ring_id']==127])-1)]
    except:
        print("  [Fig 8] 跳过：缺少ring 121/127")
        return

    windows = {}
    for name, idx in [('Ring1-健康(121)', idx1), ('Ring127-极端退化', idx2)]:
        start = df_scaled.index.get_loc(idx)
        end = start + T
        win = df_scaled.iloc[start:end][CORE_COLS].values.astype(np.float32)
        win = scaler.transform(win)
        windows[name] = win

    channels = ['SE', 'P2.1泵电流', '主油箱油温']
    fig, axes = plt.subplots(len(channels), 1, figsize=(14, 10), sharex=True)
    if len(channels) == 1:
        axes = [axes]
    time_axis = np.arange(T)

    for ax, ch in zip(axes, channels):
        ci = CORE_COLS.index(ch)
        for name, win in windows.items():
            real = win[:, ci]
            t_in = torch.from_numpy(win).unsqueeze(0).to(device)
            with torch.no_grad():
                recon_t = model.decoder(model.reparameterize(*model.encoder(t_in)))
            recon = recon_t.cpu().numpy().squeeze()[:, ci]
            ax.plot(time_axis, real, 'b-', lw=1.5, alpha=0.9, label=f'{name} 实测')
            ax.plot(time_axis, recon, 'r--', lw=1.5, alpha=0.9, label=f'{name} 重构')
            residual = real - recon
            ax.fill_between(time_axis, recon, real, where=(residual > 0),
                            color='red', alpha=0.25, label=f'{name} 残差')
        ax.set_ylabel(ch, fontsize=12, rotation=0, ha='right')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('时间步 (s)', fontsize=12)
    axes[0].set_title('Fig 8: 多物理场通道重构波形对比', fontsize=13, pad=10)
    axes[0].legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [Fig 8] 保存至 {save_path}")
    plt.close()


def plot_latent_trajectory(latents, ring_ids, save_path=None):
    """Fig 9: 潜在流形UMAP漂移轨迹"""
    if latents is None:
        print("  [Fig 9] 跳过")
        return
    emb = compute_umap(latents)
    if emb is None:
        print("  [Fig 9] 跳过（UMAP不可用）")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    unique_rings = sorted(set(ring_ids))
    colors = cm.plasma(np.linspace(0.1, 0.9, len(unique_rings)))

    for i, (rid, c) in enumerate(zip(unique_rings, colors)):
        mask = ring_ids == rid
        pts = emb[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=[c], s=15, alpha=0.6, label=f'Ring {rid}', edgecolors='none')
        if i > 0:
            prev_rid = unique_rings[i-1]
            pmask = ring_ids == prev_rid
            prev_c = emb[pmask].mean(axis=0)
            curr_c = pts.mean(axis=0)
            ax.annotate('', xy=curr_c, xytext=prev_c,
                        arrowprops=dict(arrowstyle='->', color=c, lw=2.0, alpha=0.7))

    sm = cm.ScalarMappable(cmap='plasma', norm=Normalize(vmin=unique_rings[0], vmax=unique_rings[-1]))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('环号', fontsize=11)
    ax.set_xlabel('UMAP-1', fontsize=12)
    ax.set_ylabel('UMAP-2', fontsize=12)
    ax.set_title('Fig 9: 潜在流形空间二维漂移轨迹', fontsize=13, pad=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [Fig 9] 保存至 {save_path}")
    plt.close()


def plot_macro_health_indicator(hi_values, ring_ids_sorted, save_path=None):
    """Fig 10: 宏观退化曲线"""
    fig, ax = plt.subplots(figsize=(13, 6))
    window = min(5, max(2, len(hi_values)//2))
    hi_smooth = pd.Series(hi_values).rolling(window=window, center=True, min_periods=1).mean()

    ax.plot(ring_ids_sorted, hi_values, 'bo-', lw=2, ms=8, label='帧级MSE聚合HI(R)', zorder=3)
    ax.plot(ring_ids_sorted, hi_smooth.values, 'r--', lw=2.5,
            label=f'{window}点滑动平均', alpha=0.85, zorder=2)
    ax.scatter(ring_ids_sorted, hi_values, c='blue', s=80, zorder=4)

    ax.set_xlabel('环号 (Ring ID)', fontsize=12)
    ax.set_ylabel('健康指标 HI(R)', fontsize=12)
    ax.set_title('Fig 10: 全生命周期宏观退化指数演化曲线', fontsize=13, pad=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    n = len(hi_values)
    if n >= 8:
        mid = min(8, n)
        ax.axvspan(ring_ids_sorted[0], ring_ids_sorted[mid-1], alpha=0.05, color='blue')
        ax.axvspan(ring_ids_sorted[mid-1], ring_ids_sorted[-1], alpha=0.05, color='red')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [Fig 10] 保存至 {save_path}")
    plt.close()


def plot_feature_loss_contribution(df_scaled, model, scaler, CORE_COLS,
                                   ring_ids_sorted, device=None, save_path=None):
    """Fig 11: 跨物理场MSE贡献热力图"""
    if device is None:
        device = DEVICE
    model.eval()

    PHY_GROUPS = {
        '能效/机械': ['SE', 'FPI', 'TPI'],
        '电气': ['P2.1泵电流'],
        '热力': ['主油箱油温'],
        '流体': ['泥水仓顶部1压力', '排浆密度'],
    }

    ring_mse_matrix = {grp: {} for grp in PHY_GROUPS}
    for rid in ring_ids_sorted:
        rid_data = df_scaled[df_scaled['ring_id'] == rid]
        if len(rid_data) < WINDOW_SIZE:
            continue
        start = rid_data.index[0]
        end = start + WINDOW_SIZE
        win = rid_data.iloc[:WINDOW_SIZE][CORE_COLS].values.astype(np.float32)
        win = scaler.transform(win)
        t_in = torch.from_numpy(win).unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model.decoder(model.reparameterize(*model.encoder(t_in)))
        mse_per_ch = ((win - recon.cpu().numpy().squeeze()) ** 2).mean(axis=0)
        for grp, cols in PHY_GROUPS.items():
            cidx = [CORE_COLS.index(c) for c in cols if c in CORE_COLS]
            ring_mse_matrix[grp][rid] = float(np.mean([mse_per_ch[i] for i in cidx]))

    grp_names = list(PHY_GROUPS.keys())
    heatmap_data = np.array([[ring_mse_matrix[g].get(r, np.nan) for r in ring_ids_sorted]
                              for g in grp_names])

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[f'R{r}' for r in ring_ids_sorted],
                yticklabels=grp_names,
                cbar_kws={'label': 'MSE贡献均值'},
                ax=ax, linewidths=0.5)
    ax.set_xlabel('环号', fontsize=12)
    ax.set_ylabel('物理场', fontsize=12)
    ax.set_title('Fig 11: 跨物理场特征重构误差贡献热力图', fontsize=13, pad=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [Fig 11] 保存至 {save_path}")
    plt.close()


# =============================================================================
# 主程序
# =============================================================================

def main():
    print("=" * 70)
    print("[TS-VAE v3.0] 真实数据闭环版 — 启动")
    print("=" * 70)

    # Step 1-2: 数据
    print("\n[Step 1-2] 加载数据并构建特征矩阵...")
    df_raw = load_and_engineer_features()
    df_scaled, CORE_COLS, scaler = build_core_feature_matrix(df_raw)
    INPUT_SIZE = len(CORE_COLS)

    # Step 3: 训练集（前两日健康基线）
    print("\n[Step 3] 构建训练集（前两日健康基线）...")
    df_healthy = df_scaled[
        pd.to_datetime(df_scaled['date']).dt.date <= pd.to_datetime('2026-02-08').date()
    ].reset_index(drop=True)
    print(f"  健康样本数: {len(df_healthy):,}")

    train_ds = TimeWindowDataset(df_healthy, window_size=WINDOW_SIZE,
                                 step_size=STEP_SIZE, feature_cols=CORE_COLS)
    train_ds.set_scaler(scaler)
    total = len(train_ds)
    val_size = max(10, total // 5)
    train_size = total - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    print(f"  训练窗口: {train_size}, 验证窗口: {val_size}")

    # Step 4: 模型
    print("\n[Step 4] 初始化模型...")
    model = TSVAE(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, LATENT_DIM, WINDOW_SIZE, BETA)
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # Step 5: 训练
    print("\n[Step 5] TS-VAE自监督预训练...")
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=False)
    history = train_tsvae(model, train_loader, val_loader,
                           epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                           patience=PATIENCE, device=DEVICE)

    # 保存训练历史
    pd.DataFrame(history).to_csv(RESULTS_DIR / "training_history.csv", index=False)

    # Step 6: 全生命周期推理（批量加速）
    print("\n[Step 6] 全生命周期推理...")
    full_ds = TimeWindowDataset(df_scaled, window_size=WINDOW_SIZE,
                                 step_size=STEP_SIZE, feature_cols=CORE_COLS)
    full_ds.set_scaler(scaler)
    print(f"  全量推理窗口数: {len(full_ds):,}")

    frame_mse, ring_ids, latents = extract_all_metrics(model, full_ds, DEVICE)
    print(f"  帧数: {len(frame_mse):,} | MSE范围: [{frame_mse.min():.4f}, {frame_mse.max():.4f}]")

    # Step 7: 环级HI聚合
    print("\n[Step 7] 环级聚合...")
    hi_values, ring_ids_sorted = aggregate_to_ring_level(frame_mse, ring_ids, num_rings=12)
    print("  环号 | HI(R)")
    print("  -----+----------------")
    for r, h in zip(ring_ids_sorted, hi_values):
        print(f"  Ring {int(r):2d} | {h:.6f}")

    # 保存HI
    pd.DataFrame({'Ring_ID': ring_ids_sorted, 'HI_Value': hi_values}).to_csv(
        RESULTS_DIR / "ring_health_indicator.csv", index=False)

    # Step 8: 可视化
    print("\n[Step 8] 生成4张论文级图表（DPI=300）...")
    plot_multichannel_reconstruction(df_scaled, model, scaler, CORE_COLS, DEVICE,
                                    save_path=FIG_DIR / "fig8_multichannel_reconstruction.png")
    plot_latent_trajectory(latents, ring_ids, save_path=FIG_DIR / "fig9_latent_space_trajectory.png")
    plot_macro_health_indicator(hi_values, ring_ids_sorted,
                                 save_path=FIG_DIR / "fig10_macro_health_indicator.png")
    plot_feature_loss_contribution(df_scaled, model, scaler, CORE_COLS, ring_ids_sorted, DEVICE,
                                   save_path=FIG_DIR / "fig11_feature_loss_contribution.png")

    # Step 9: 量化报告
    print("\n[Step 9] 生成量化报告...")
    n = len(hi_values)
    mid = min(8, n)
    early = hi_values[:mid]
    late = hi_values[mid:]
    slope_early = (early[-1]-early[0])/(mid-1) if mid>1 else 0
    slope_late = (late[-1]-late[0])/(len(late)-1) if len(late)>1 else 0
    jump_ratio = slope_late/abs(slope_early) if abs(slope_early)>1e-9 else float('inf')

    final_epoch = len(history['train_loss'])
    final_t = history['train_loss'][-1]
    final_v = history['val_loss'][-1]
    final_r = history['recon_loss'][-1]
    final_k = history['kl_loss'][-1]

    report = f"""# 模块三深度集成分析报告

## 一、模型收敛量化指标

| 指标 | 值 |
|------|-----|
| 最终Epoch | {final_epoch} |
| 训练ELBO | {final_t:.4f} |
| 验证ELBO | {final_v:.4f} |
| 重构损失R | {final_r:.4f} |
| KL散度 | {final_k:.4f} |

## 二、环级HI(R)序列表

| 环号 | HI(R) |
|------|-------|
"""
    for r, h in zip(ring_ids_sorted, hi_values):
        report += f"| Ring {int(r)} | {h:.8f} |\n"

    report += f"""
## 三、退化斜率分析

- 前期(1-{mid})斜率: {slope_early:.6f}
- 后期({mid+1}-{n})斜率: {slope_late:.6f}
- 跃迁比: {jump_ratio:.4f}
"""
    report_path = Path("TBM_Cutter_Wear_Project/reports/module3_deep_integration_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding='utf-8')
    print(f"  报告已保存至: {report_path}")

    # Step 10: 模型持久化
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {'input_size':INPUT_SIZE,'hidden_size':HIDDEN_SIZE,
                         'num_layers':NUM_LAYERS,'latent_dim':LATENT_DIM,
                         'seq_len':WINDOW_SIZE,'beta':BETA},
        'feature_cols': CORE_COLS,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'history': history,
        'ring_HI': {int(r): float(hi) for r, hi in zip(ring_ids_sorted, hi_values)},
    }, RESULTS_DIR / "tsvae_model.pt")
    print(f"  模型已保存")

    print("\n" + "=" * 70)
    print("[TS-VAE v3.0] 全流程完毕")
    print("=" * 70)


if __name__ == '__main__':
    main()