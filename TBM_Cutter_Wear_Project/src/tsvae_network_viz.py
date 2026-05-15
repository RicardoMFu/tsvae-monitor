"""
================================================================================
TS-VAE 网络架构可视化：3Blue1Brown 风格 2.5D 拓扑图
================================================================================
版本: v1.0.0
用途: 生成论文级别网络架构示意图

设计理念：
- 灵感来源：3Blue1Brown 的神经网络的视觉风格
- 深色背景 + 高对比度渐变色彩
- 流形曲面 + 箭头连线 + 层次标注
- 数学符号与文字标注共存
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Ellipse, Arc, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_rgba
from matplotlib.font_manager import FontProperties
import mpl_toolkits.mplot3d.art3d as mpatches3d

# ─── 中文字体支持 ───
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong', 'STXihei', 'Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False

# ─── 风格配置 ───
BG_COLOR = '#0d1117'           # 深空黑背景
PANEL_COLOR = '#161b22'         # 面板背景
ACCENT_CYAN = '#58a6ff'        # 高亮蓝（输入/输出）
ACCENT_GREEN = '#3fb950'        # 编码器绿
ACCENT_RED = '#f85149'          # 解码器红
ACCENT_PURPLE = '#bc8cff'       # 潜在空间紫
ACCENT_ORANGE = '#d29922'       # KL散度橙
TEXT_COLOR = '#c9d1d9'          # 主文字灰白
SUBTEXT_COLOR = '#8b949e'       # 副文字灰色
GRID_COLOR = '#21262d'          # 网格线


def draw_box(ax, x, y, w, h, color, alpha=0.85, radius=0.015):
    """绘制带圆角矩形框（2.5D感）"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0.003,rounding_size={radius}",
                         facecolor=color, edgecolor='white', linewidth=1.5,
                         alpha=alpha, zorder=3)
    ax.add_patch(box)
    return box


def draw_layer_dots(ax, x, y, n, color, size=0.012):
    """绘制一列圆点表示神经元层"""
    dots = []
    for i in range(n):
        cx = x
        cy = y + (i - (n-1)/2) * (size * 2.5)
        circle = Circle((cx, cy), size, facecolor=color, edgecolor='white',
                        linewidth=0.5, alpha=0.9, zorder=4)
        ax.add_patch(circle)
        dots.append((cx, cy))
    return dots


def draw_arrow(ax, x1, y1, x2, y2, color, lw=1.5, alpha=0.8):
    """绘制带箭头的连接线"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, alpha=alpha,
                                connectionstyle='arc3,rad=0.0'),
                zorder=2)


def draw_curved_arrow(ax, x1, y1, x2, y2, color, lw=1.5, rad=0.3):
    """绘制曲率箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, alpha=0.7,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=2)


def add_text(ax, x, y, text, size=9, color=TEXT_COLOR, ha='center', va='center', bold=False):
    """添加文字标注"""
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, fontsize=size, color=color,
            ha=ha, va=va, fontweight=weight, zorder=5)


def add_bracket_label(ax, x, y, text, height=0.06, color=ACCENT_CYAN):
    """添加括号标签"""
    bracket = mpatches.Arc((x, y - height/2), height, height,
                           angle=0, theta1=180, theta2=360,
                           color=color, linewidth=1.5, alpha=0.8)
    ax.add_patch(bracket)
    ax.text(x + height/2 + 0.01, y, text, fontsize=8, color=color,
            va='center', ha='left', fontweight='bold', zorder=5)


def draw_manifold_surface(ax, x, y, z, width, height, color, alpha=0.3, n_contours=5):
    """绘制流形曲面（3D效果）"""
    # 简化为2D椭圆渐变效果
    ellipse = Ellipse((x, y), width, height, angle=0,
                     facecolor=color, edgecolor='white', linewidth=1.0,
                     alpha=alpha, zorder=2)
    ax.add_patch(ellipse)
    # 内部等高线
    for i in range(1, n_contours):
        r_ratio = i / n_contours
        inner = Ellipse((x, y), width * r_ratio, height * r_ratio, angle=0,
                       facecolorcolor='none', edgecolor=color,
                       linewidth=0.5, alpha=alpha * 0.5, zorder=2)
        ax.add_patch(inner)


def draw_latent_sphere(ax, cx, cy, r, color, alpha=0.4):
    """绘制潜在空间球体（椭圆表示3D球）"""
    sphere = Ellipse((cx, cy), r * 2, r * 1.6, angle=0,
                     facecolor=color, edgecolor='white', linewidth=2.0,
                     alpha=alpha, zorder=3)
    ax.add_patch(sphere)
    # 高光
    highlight = Circle((cx - r*0.3, cy + r*0.3), r*0.15,
                       facecolor='white', alpha=0.3, zorder=4)
    ax.add_patch(highlight)


def draw_distributions(ax, cx, cy, width, height, color1, color2, n_samples=30):
    """绘制两侧分布（表示均值/方差）"""
    for i in range(n_samples):
        offset = np.random.randn() * 0.005
        x_val = cx + (np.random.rand() - 0.5) * width
        y_val = cy + (np.random.rand() - 0.5) * height
        dot = Circle((x_val, y_val), 0.003, facecolor=color1,
                    alpha=0.3, zorder=2)
        ax.add_patch(dot)


# =============================================================================
# 主绘图函数
# =============================================================================

def draw_tsvae_architecture(save_path=None):
    """
    绘制 TS-VAE 2.5D 网络架构拓扑图
    采用3Blue1Brown风格：深色背景 + 渐变色彩 + 数学标注
    """

    fig, ax = plt.subplots(figsize=(18, 11), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-0.5, 17.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── 全局标题 ──
    fig.text(0.5, 0.97, 'Time-Series Variational Autoencoder (TS-VAE)',
             ha='center', va='top', fontsize=20, fontweight='bold',
             color=TEXT_COLOR)
    fig.text(0.5, 0.935, 'Multivariate Temporal Variational Autoencoder | Unsupervised Cutter Health Monitoring',
             ha='center', va='top', fontsize=11, color=SUBTEXT_COLOR)

    # ══════════════════════════════════════════════════════════════════
    # 第1行：输入层（10维原始传感器 → Z-Score → 滑动窗口）
    # ══════════════════════════════════════════════════════════════════

    # 输入原始数据框
    draw_box(ax, 1.5, 8.5, 2.2, 1.2, color='#1a2332', alpha=0.9)
    add_text(ax, 1.5, 9.0, 'Raw Sensor Data', 9, ACCENT_CYAN, bold=True)
    add_text(ax, 1.5, 8.7, '$F_v, T, v, RPM$', 8, TEXT_COLOR)
    add_text(ax, 1.5, 8.45, '$I_{p21}, T_{oil}, SE...$', 7, SUBTEXT_COLOR)
    add_text(ax, 1.5, 8.2, '10 channels', 7, SUBTEXT_COLOR)

    # Z-Score标准化
    draw_arrow(ax, 2.65, 8.5, 3.35, 8.5, ACCENT_CYAN, lw=2)
    draw_box(ax, 3.9, 8.5, 1.1, 0.7, color='#1f3a1f', alpha=0.9)
    add_text(ax, 3.9, 8.55, 'Z-Score', 8, ACCENT_GREEN, bold=True)
    add_text(ax, 3.9, 8.35, 'StandardScaler', 6, SUBTEXT_COLOR)

    # 滑动窗口
    draw_arrow(ax, 4.45, 8.5, 5.15, 8.5, ACCENT_CYAN, lw=2)
    draw_box(ax, 6.2, 8.5, 2.1, 1.2, color='#1a2332', alpha=0.9)
    add_text(ax, 6.2, 9.0, 'Sliding Window', 9, ACCENT_CYAN, bold=True)
    add_text(ax, 6.2, 8.7, '$T=60s,\; S=30s$', 8, TEXT_COLOR)
    add_text(ax, 6.2, 8.35, '$\\mathcal{X} \in \mathbb{R}^{B \times 60 \times 10}$', 7, SUBTEXT_COLOR)

    # ══════════════════════════════════════════════════════════════════
    # 第2行：编码器（GRU Stack → μ / σ² 推断）
    # ══════════════════════════════════════════════════════════════════

    # ENCODER 标签
    add_text(ax, 6.2, 7.3, 'TS-VAE ENCODER', 10, ACCENT_GREEN, bold=True)
    add_text(ax, 6.2, 7.0, 'Layer 1: GRU$(10 \rightarrow 64)$', 8, TEXT_COLOR)
    add_text(ax, 6.2, 6.7, 'Layer 2: GRU$(64 \rightarrow 64)$', 8, TEXT_COLOR)

    # GRU 编码器隐状态（用竖向神经元点表示）
    gru_x, gru_y = 6.2, 5.6
    draw_box(ax, gru_x, gru_y, 2.0, 1.4, '#1a2a1a', alpha=0.8)
    add_text(ax, gru_x, gru_y + 0.5, 'Bidirectional GRU', 8, ACCENT_GREEN)
    add_text(ax, gru_x, gru_y + 0.2, 'Sequence Encoding', 7, SUBTEXT_COLOR)
    add_text(ax, gru_x, gru_y - 0.1, '$h_T \in \mathbb{R}^{64}$', 7, TEXT_COLOR)
    add_text(ax, gru_x, gru_y - 0.4, '$h_T = \text{GRU}(x_{1:T})$', 6, SUBTEXT_COLOR)

    # 连接线：输入 → GRU
    ax.annotate('', xy=(5.15, 6.3), xytext=(5.15, 8.5),
                arrowprops=dict(arrowstyle='->', color=ACCENT_CYAN, lw=1.5, alpha=0.7))
    ax.plot([5.15, 5.15], [5.15, 6.3], color=ACCENT_CYAN, lw=1.5, alpha=0.7, zorder=2)

    # 箭头：GRU → μ
    draw_arrow(ax, 7.25, 5.6, 8.05, 5.6, ACCENT_GREEN, lw=1.5)

    # μ 推断头
    draw_box(ax, 9.0, 5.8, 1.0, 0.5, '#1f2937', alpha=0.85)
    add_text(ax, 9.0, 5.8, '$\mu$', 12, ACCENT_PURPLE, bold=True)
    add_text(ax, 9.0, 5.5, '$W_\mu \cdot h_T + b_\mu$', 6, SUBTEXT_COLOR)

    # σ² 推断头
    draw_arrow(ax, 7.25, 5.5, 8.05, 5.1, ACCENT_ORANGE, lw=1.5)
    draw_box(ax, 9.0, 4.9, 1.0, 0.5, '#1f2937', alpha=0.85)
    add_text(ax, 9.0, 4.9, '$\log\sigma^2$', 10, ACCENT_ORANGE, bold=True)
    add_text(ax, 9.0, 4.6, '$W_\sigma \cdot h_T + b_\sigma$', 6, SUBTEXT_COLOR)

    # ══════════════════════════════════════════════════════════════════
    # 第3行：重参数化采样（潜在空间）
    # ══════════════════════════════════════════════════════════════════

    # Reparameterization label
    add_text(ax, 10.5, 5.35, 'Reparameterization', 9, ACCENT_PURPLE, bold=True)
    add_text(ax, 10.5, 5.05, '$z = \mu + \sigma \odot \epsilon,\;\;\epsilon \sim \mathcal{N}(0,I)$', 7, SUBTEXT_COLOR)

    # 小 epsilon 采样示意
    draw_box(ax, 10.5, 4.5, 0.8, 0.5, '#2d1f4e', alpha=0.8)
    add_text(ax, 10.5, 4.5, '$\epsilon \sim \mathcal{N}(0,I)$', 6, ACCENT_PURPLE)

    # 潜在空间球
    draw_latent_sphere(ax, 11.8, 4.5, 0.5, ACCENT_PURPLE, alpha=0.35)
    add_text(ax, 11.8, 4.5, '$z \in \mathbb{R}^{16}$', 8, ACCENT_PURPLE, bold=True)
    add_text(ax, 11.8, 3.8, 'Latent Manifold', 7, SUBTEXT_COLOR)

    # 连接线
    draw_arrow(ax, 9.55, 5.8, 10.35, 5.0, ACCENT_PURPLE, lw=1.5)
    draw_arrow(ax, 9.55, 4.9, 10.35, 4.4, ACCENT_PURPLE, lw=1.5)
    draw_arrow(ax, 10.95, 4.5, 11.25, 4.5, ACCENT_PURPLE, lw=1.5)

    # KL散度标注
    draw_box(ax, 10.5, 3.3, 2.5, 0.5, '#2d1f0a', alpha=0.8)
    add_text(ax, 10.5, 3.3, '$L_{KL} = D_{KL}(q_\phi(z|x) \parallel \mathcal{N}(0,I))$', 7, ACCENT_ORANGE)

    # ══════════════════════════════════════════════════════════════════
    # 第4行：解码器（对称GRU → 重构）
    # ══════════════════════════════════════════════════════════════════

    # 连接：潜在球 → 解码器
    draw_arrow(ax, 13.35, 4.5, 14.15, 5.6, ACCENT_RED, lw=2)

    # DECODER 标签
    add_text(ax, 14.5, 7.3, 'TS-VAE DECODER', 10, ACCENT_RED, bold=True)
    add_text(ax, 14.5, 7.0, 'Layer 1: GRU$(16 \rightarrow 64)$', 8, TEXT_COLOR)
    add_text(ax, 14.5, 6.7, 'Layer 2: GRU$(64 \rightarrow 64)$', 8, TEXT_COLOR)

    # GRU 解码器（对称结构）
    dec_x, dec_y = 14.5, 5.6
    draw_box(ax, dec_x, dec_y, 2.0, 1.4, '#1a1a2a', alpha=0.8)
    add_text(ax, dec_x, dec_y + 0.5, 'Symmetric GRU', 8, ACCENT_RED)
    add_text(ax, dec_x, dec_y + 0.2, 'Autoregressive Decoding', 7, SUBTEXT_COLOR)
    add_text(ax, dec_x, dec_y - 0.1, '$\hat{x}_t = \text{GRU}(\hat{h}_{t-1})$', 7, TEXT_COLOR)
    add_text(ax, dec_x, dec_y - 0.4, '$T=60$ timesteps', 6, SUBTEXT_COLOR)

    # 输出头
    draw_arrow(ax, 15.55, 5.6, 16.35, 5.6, ACCENT_RED, lw=2)

    draw_box(ax, 17.0, 5.6, 1.0, 1.2, '#1a2332', alpha=0.9)
    add_text(ax, 17.0, 5.9, '$\hat{\mathcal{X}}$', 10, ACCENT_CYAN, bold=True)
    add_text(ax, 17.0, 5.6, 'Reconstruction', 7, TEXT_COLOR)
    add_text(ax, 17.0, 5.3, '$\mathbb{R}^{B \times 60 \times 10}$', 6, SUBTEXT_COLOR)

    # ══════════════════════════════════════════════════════════════════
    # 右侧：损失函数（ELBO）
    # ══════════════════════════════════════════════════════════════════

    # 重构损失
    draw_box(ax, 17.0, 8.5, 1.8, 0.8, '#1a1f2a', alpha=0.85)
    add_text(ax, 17.0, 8.7, '$\mathcal{L}_{recon} = \text{MSE}(\mathcal{X}, \hat{\mathcal{X}})$', 6, TEXT_COLOR)
    add_text(ax, 17.0, 8.4, 'Reconstruction Loss', 7, ACCENT_CYAN)

    # KL损失
    draw_box(ax, 10.5, 2.5, 2.5, 0.5, '#1f2d1a', alpha=0.8)
    add_text(ax, 10.5, 2.5, '$\mathcal{L}_{KL} = -\frac{1}{2}\sum(1+\log\sigma^2-\mu^2-e^{\log\sigma^2})$', 5.5, ACCENT_ORANGE)
    add_text(ax, 10.5, 2.1, 'KL Divergence', 7, ACCENT_ORANGE)

    # ELBO 总损失
    draw_box(ax, 13.5, 2.3, 2.8, 0.9, '#1a1a2a', alpha=0.9)
    add_text(ax, 13.5, 2.55, 'ELBO = $\mathcal{L}_{recon} + \beta \cdot \mathcal{L}_{KL}$', 7, TEXT_COLOR)
    add_text(ax, 13.5, 2.25, 'Evidence Lower Bound', 7, ACCENT_CYAN)
    add_text(ax, 13.5, 1.95, '$\beta=0.001$', 6, SUBTEXT_COLOR)

    # ══════════════════════════════════════════════════════════════════
    # 底部：物理含义标注
    # ══════════════════════════════════════════════════════════════════

    # 健康基线
    draw_box(ax, 3.0, 1.8, 3.5, 0.7, '#0d2626', alpha=0.85)
    add_text(ax, 3.0, 1.95, 'Healthy Baseline (Training)', 8, ACCENT_GREEN, bold=True)
    add_text(ax, 3.0, 1.65, 'Ring 121-123 (Day 1-2) | Self-supervised', 6, SUBTEXT_COLOR)
    add_text(ax, 3.0, 1.4, 'Network learns normal physics coupling manifold', 6, SUBTEXT_COLOR)

    # 推理阶段
    draw_box(ax, 9.0, 1.8, 3.5, 0.7, '#261a1a', alpha=0.85)
    add_text(ax, 9.0, 1.95, 'Inference (Full Lifecycle)', 8, ACCENT_RED, bold=True)
    add_text(ax, 9.0, 1.65, 'Ring 121-130 | All 5 Days | No Labels', 6, SUBTEXT_COLOR)
    add_text(ax, 9.0, 1.4, 'MSE = extra mechanical dissipation energy', 6, SUBTEXT_COLOR)

    # HI聚合
    draw_box(ax, 14.5, 1.8, 3.0, 0.7, '#1a1a2e', alpha=0.85)
    add_text(ax, 14.5, 1.95, 'HI(R) Aggregation', 8, ACCENT_PURPLE, bold=True)
    add_text(ax, 14.5, 1.65, 'Frame MSE $\\rightarrow$ Ring-Level Mean', 6, SUBTEXT_COLOR)
    add_text(ax, 14.5, 1.4, 'HI(127)=803.81 $\\gg$ mean(others)=19.5', 6, ACCENT_RED)

    # ══════════════════════════════════════════════════════════════════
    # 图例
    # ══════════════════════════════════════════════════════════════════

    legend_x = 0.3
    legend_y = 0.7

    items = [
        (ACCENT_CYAN, 'Input / Output'),
        (ACCENT_GREEN, 'Encoder'),
        (ACCENT_RED, 'Decoder'),
        (ACCENT_PURPLE, 'Latent Space'),
        (ACCENT_ORANGE, 'KL Divergence'),
    ]

    for i, (color, label) in enumerate(items):
        circle = Circle((legend_x + i * 2.8, legend_y), 0.08, facecolor=color,
                       edgecolor='white', linewidth=0.5, zorder=5)
        ax.add_patch(circle)
        ax.text(legend_x + i * 2.8 + 0.15, legend_y, label, fontsize=7,
                color=TEXT_COLOR, va='center', zorder=5)

    # ══════════════════════════════════════════════════════════════════
    # 层级标注（左侧Y轴说明）
    # ══════════════════════════════════════════════════════════════════

    layer_labels = [
        (8.5, 'Data Input Layer\n(10 channels)'),
        (5.6, 'Hidden Encoding\n(64 dim GRU)'),
        (4.5, 'Latent Space\n(d = 16)'),
        (2.5, 'Loss Computation'),
    ]

    for y, label in layer_labels:
        ax.text(-0.4, y, label, fontsize=7, color=SUBTEXT_COLOR,
                ha='right', va='center', fontweight='bold')

    # ── 保存 ──
    plt.tight_layout(pad=0.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=BG_COLOR, edgecolor='none')
        print(f"  [TS-VAE Architecture] 保存至 {save_path}")

    plt.close()


def draw_latent_manifold_diagram(save_path=None):
    """
    绘制潜在流形演化示意图（健康→退化→异常的三阶段）
    3Blue1Brown风格流形投影图
    """

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-2, 16)
    ax.set_ylim(-1, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.text(0.5, 0.97, 'Latent Manifold Evolution: Health $\\rightarrow$ Wear $\\rightarrow$ Failure',
             ha='center', va='top', fontsize=16, fontweight='bold', color=TEXT_COLOR)
    fig.text(0.5, 0.935, 'UMAP 2D Projection | Ring 121 $\\rightarrow$ 130 | Degradation Trajectory',
             ha='center', va='top', fontsize=10, color=SUBTEXT_COLOR)

    # ═══ 三个流形区域 ═══

    # 区域1：健康基线（紧凑聚类）
    healthy_center = (2.5, 5.5)
    for i in range(40):
        angle = np.random.rand() * 2 * np.pi
        r = np.random.rand() * 0.4
        x = healthy_center[0] + r * np.cos(angle)
        y = healthy_center[1] + r * np.sin(angle)
        c = Circle((x, y), 0.06, facecolor='#3fb950', edgecolor='none',
                  alpha=0.6 + np.random.rand() * 0.4, zorder=3)
        ax.add_patch(c)

    # 健康聚类椭圆
    ellipse_h = Ellipse(healthy_center, 1.8, 1.4, angle=0,
                       facecolor='#3fb950', edgecolor='white', linewidth=1.5,
                       alpha=0.15, zorder=2)
    ax.add_patch(ellipse_h)
    add_text(ax, healthy_center[0], healthy_center[1] + 1.0, 'Healthy Baseline',
             10, '#3fb950', bold=True)
    add_text(ax, healthy_center[0], healthy_center[1] + 0.6, 'Ring 121-123',
             8, TEXT_COLOR)
    add_text(ax, healthy_center[0], healthy_center[1] + 0.2, 'Low MSE', 7, SUBTEXT_COLOR)

    # 箭头1：健康→轻度磨损
    draw_curved_arrow(ax, 4.0, 5.5, 6.0, 4.5, '#58a6ff', lw=2, rad=0.15)
    add_text(ax, 5.0, 5.15, 'Wear\nAccumulation', 7, ACCENT_CYAN, bold=True)

    # 区域2：轻度磨损（扩散）
    wear_center = (7.0, 4.0)
    for i in range(60):
        angle = np.random.rand() * 2 * np.pi
        r = np.random.rand() * 0.9
        x = wear_center[0] + r * np.cos(angle)
        y = wear_center[1] + r * np.sin(angle)
        c = Circle((x, y), 0.055, facecolor='#d29922', edgecolor='none',
                  alpha=0.5 + np.random.rand() * 0.5, zorder=3)
        ax.add_patch(c)

    ellipse_w = Ellipse(wear_center, 3.2, 2.4, angle=15,
                        facecolor='#d29922', edgecolor='white', linewidth=1.5,
                        alpha=0.12, zorder=2)
    ax.add_patch(ellipse_w)
    add_text(ax, wear_center[0], wear_center[1] + 1.5, 'Mild Wear', 10, '#d29922', bold=True)
    add_text(ax, wear_center[0], wear_center[1] + 1.0, 'Ring 124-126, 128-129', 8, TEXT_COLOR)
    add_text(ax, wear_center[0], wear_center[1] + 0.5, 'Moderate MSE', 7, SUBTEXT_COLOR)

    # 箭头2：磨损→临界点
    draw_curved_arrow(ax, 8.8, 3.5, 10.5, 2.5, '#f85149', lw=2.5, rad=0.2)
    add_text(ax, 9.6, 3.2, 'Critical\nTransition', 7, ACCENT_RED, bold=True)

    # 区域3：异常峰值（Ring 127）
    fault_center = (12.0, 2.0)
    for i in range(25):
        angle = np.random.rand() * 2 * np.pi
        r = np.random.rand() * 0.35
        x = fault_center[0] + r * np.cos(angle)
        y = fault_center[1] + r * np.sin(angle)
        c = Circle((x, y), 0.055, facecolor='#f85149', edgecolor='none',
                  alpha=0.7 + np.random.rand() * 0.3, zorder=3)
        ax.add_patch(c)

    ellipse_f = Ellipse(fault_center, 1.6, 1.2, angle=0,
                        facecolor='#f85149', edgecolor='white', linewidth=2.0,
                        alpha=0.2, zorder=2)
    ax.add_patch(ellipse_f)
    add_text(ax, fault_center[0], fault_center[1] + 1.0, 'Cutter Failure', 10, '#f85149', bold=True)
    add_text(ax, fault_center[0], fault_center[1] + 0.5, 'Ring 127', 8, TEXT_COLOR)
    add_text(ax, fault_center[0], fault_center[1] + 0.0, 'HI = 803.81 (41x)', 7, ACCENT_RED)

    # 质心连线（时间演化方向）
    tsne_centers = np.array([healthy_center, wear_center, fault_center])
    ax.plot(tsne_centers[:, 0], tsne_centers[:, 1],
            color='white', lw=1.5, alpha=0.4, zorder=1, linestyle='--')

    # 时间箭头
    ax.annotate('', xy=(12.5, 2.0), xytext=(2.0, 5.5),
                arrowprops=dict(arrowstyle='->', color='white', lw=2,
                                mutation_scale=15,
                                connectionstyle='arc3,rad=-0.3'),
                zorder=5)
    add_text(ax, 7.0, 7.2, 'Time $\\rightarrow$ | Advance $\\rightarrow$ | Ring Number $\\rightarrow$',
             9, TEXT_COLOR, bold=True)

    # ═══ HI(R) 数值标注 ═══
    hi_data = [
        (1.2, 3.0, 121, 32.0),
        (2.0, 2.2, 122, 5.1),
        (3.0, 2.0, 123, 17.3),
        (4.5, 2.5, 124, 27.4),
        (5.5, 1.8, 125, 20.2),
        (6.0, 2.0, 126, 5.8),
        (8.0, 2.2, 127, 803.8),
        (9.0, 1.5, 128, 41.2),
        (10.0, 2.0, 129, 23.9),
        (11.0, 1.8, 130, 1.4),
    ]

    ax.text(1.0, 0.8, 'HI(R) Values along trajectory:', fontsize=8, color=SUBTEXT_COLOR, fontweight='bold')
    for x, y, rid, hi in hi_data:
        color = '#f85149' if rid == 127 else ('#d29922' if hi > 20 else '#3fb950')
        ax.plot(x, y, 'o', color=color, ms=5, zorder=4)
        ax.text(x + 0.1, y, f'R{int(rid)}:{hi:.0f}', fontsize=5.5, color=color, va='center')

    # ═══ 图例 ═══
    legend_items = [
        ('#3fb950', 'Healthy (Ring 121-123)'),
        ('#d29922', 'Mild Wear (Ring 124-129)'),
        ('#f85149', 'Critical Failure (Ring 127)'),
    ]
    for i, (c, l) in enumerate(legend_items):
        cx = 0.5 + i * 4.5
        cy = 0.3
        ax.add_patch(Circle((cx, cy), 0.1, facecolor=c, edgecolor='white', linewidth=0.5, zorder=5))
        ax.text(cx + 0.18, cy, l, fontsize=7, color=TEXT_COLOR, va='center')

    plt.tight_layout(pad=0.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=BG_COLOR, edgecolor='none')
        print(f"  [Latent Manifold] 保存至 {save_path}")

    plt.close()


def draw_hierarchy_flowchart(save_path=None):
    """
    绘制层级流程图：输入→编码→潜在→解码→HI输出
    3Blue1Brown风格的信息流图
    """

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.text(0.5, 0.97, 'TS-VAE Unsupervised Learning Pipeline',
             ha='center', va='top', fontsize=18, fontweight='bold', color=TEXT_COLOR)
    fig.text(0.5, 0.935, 'From Multi-Physics Sensor Streams to Ring-Level Health Indicator',
             ha='center', va='top', fontsize=11, color=SUBTEXT_COLOR)

    # ═══ 信息流层级 ═══

    stages = [
        {'x': 2.0, 'label': 'Input', 'sub': 'Raw Sensors\n10 Channels', 'color': ACCENT_CYAN},
        {'x': 5.5, 'label': 'Encoding', 'sub': 'Bi-GRU\n$\mu, \log\sigma^2$', 'color': ACCENT_GREEN},
        {'x': 9.0, 'label': 'Latent', 'sub': '$z \in \mathbb{R}^{16}$\nReparameterization', 'color': ACCENT_PURPLE},
        {'x': 12.5, 'label': 'Decoding', 'sub': 'Symmetric GRU\nReconstruction $\hat{X}$', 'color': ACCENT_RED},
        {'x': 15.5, 'label': 'Health\nIndicator', 'sub': 'MSE $\\rightarrow$ HI(R)', 'color': ACCENT_ORANGE},
    ]

    for stage in stages:
        # 主框
        box = FancyBboxPatch((stage['x'] - 1.3, 5.5 - 0.9), 2.6, 1.8,
                             boxstyle="round,pad=0.02,rounding_size=0.2",
                             facecolor=stage['color'], edgecolor='white',
                             linewidth=2.0, alpha=0.2, zorder=3)
        ax.add_patch(box)
        inner = FancyBboxPatch((stage['x'] - 1.1, 5.5 - 0.7), 2.2, 1.4,
                              boxstyle="round,pad=0.01,rounding_size=0.15",
                              facecolor=stage['color'], edgecolor='white',
                              linewidth=1.0, alpha=0.5, zorder=4)
        ax.add_patch(inner)

        add_text(ax, stage['x'], 6.0, stage['label'], 10, stage['color'], bold=True)
        add_text(ax, stage['x'], 5.5, stage['sub'], 7, TEXT_COLOR)

        # 连接箭头
        if stage['x'] != 15.5:
            next_x = stage['x'] + 2.5
            ax.annotate('', xy=(next_x - 1.3, 5.5), xytext=(stage['x'] + 1.3, 5.5),
                        arrowprops=dict(arrowstyle='->', color=stage['color'],
                                        lw=2.5, mutation_scale=20), zorder=5)

    # ═══ 数学公式层（底部） ═══

    math_y = 3.8
    formulas = [
        (2.0, '$\mathcal{X} \in \mathbb{R}^{B \times 60 \times 10}$'),
        (5.5, '$h_T = \text{GRU}(x_{1:T})$\n$\mu = W_\mu h_T + b_\mu$'),
        (9.0, '$z = \mu + \sigma \odot \epsilon$\n$\epsilon \sim \mathcal{N}(0,I)$'),
        (12.5, '$\hat{x}_t = \text{GRU}(\hat{h}_{t-1})$\n$\hat{\mathcal{X}} = \text{Decoder}(z)$'),
        (15.5, '$\text{MSE} = \||\mathcal{X} - \hat{\mathcal{X} }\||^2$\n$\text{HI}(R) = \langle \text{MSE} \\rangle_R$'),
    ]

    for x, fml in formulas:
        add_text(ax, x, math_y, fml, 6.5, SUBTEXT_COLOR)

    # ═══ 训练/推理分割线 ═══

    ax.axvline(x=7.0, ymin=0.1, ymax=0.75, color='white', lw=1, alpha=0.3, linestyle='--')

    add_text(ax, 3.5, 9.5, 'TRAINING PHASE\n(Healthy Baseline Only)', 9, ACCENT_GREEN, bold=True)
    add_text(ax, 11.5, 9.5, 'INFERENCE PHASE\n(Full Lifecycle)', 9, ACCENT_RED, bold=True)

    # ═══ 数据维度标注 ═══

    dim_annotations = [
        (2.0, 4.2, '223,491 sec\nRaw Records', ACCENT_CYAN),
        (9.0, 1.5, '16-Dim\nLatent Space', ACCENT_PURPLE),
        (15.5, 1.5, '10 Rings\nHI(R)', ACCENT_ORANGE),
    ]

    for x, y, text, color in dim_annotations:
        box = FancyBboxPatch((x - 1.2, y - 0.5), 2.4, 0.9,
                            boxstyle="round,pad=0.01,rounding_size=0.1",
                            facecolor=color, edgecolor=color,
                            linewidth=1, alpha=0.15, zorder=3)
        ax.add_patch(box)
        add_text(ax, x, y, text, 7, color)

    plt.tight_layout(pad=0.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=BG_COLOR, edgecolor='none')
        print(f"  [Pipeline Flowchart] 保存至 {save_path}")

    plt.close()


def main():
    out_dir = Path(__file__).parent
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("[TS-VAE Visualization] 生成3张架构图...")
    print("=" * 60)

    draw_tsvae_architecture(out_dir / "figA1_tsvae_architecture.png")
    draw_latent_manifold_diagram(out_dir / "figA2_latent_manifold_evolution.png")
    draw_hierarchy_flowchart(out_dir / "figA3_pipeline_flowchart.png")

    print("\n可视化完成！输出文件：")
    for f in ['figA1_tsvae_architecture.png', 'figA2_latent_manifold_evolution.png', 'figA3_pipeline_flowchart.png']:
        path = out_dir / f
        if path.exists():
            size = path.stat().st_size / 1024
            print(f"  ✅ {f} ({size:.0f} KB)")


if __name__ == '__main__':
    main()