"""
论文图表绘制脚本 - 模块一：稳态剥离与清洗可视化
=================================================
生成 publication-quality 图表，适合顶级期刊投稿
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# =====================================================================
# 字体配置
# =====================================================================
import matplotlib.font_manager as fm
import os

font_candidates = [
    'C:/Windows/Fonts/simhei.ttf',
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simsun.ttc',
]

Chinese_font_path = None
for fp in font_candidates:
    if os.path.exists(fp):
        Chinese_font_path = fp
        break

if Chinese_font_path:
    Chinese_font_name = fm.FontProperties(fname=Chinese_font_path).get_name()
else:
    Chinese_font_name = 'DejaVu Sans'

print(f"使用中文字体: {Chinese_font_name}")

plt.rcParams['font.sans-serif'] = [Chinese_font_name, 'DejaVu Sans', 'Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

from pathlib import Path
from matplotlib.patches import FancyBboxPatch
from datetime import datetime

SUBFOLDER_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(r"C:\NO CLUB\experience\TBM_Cutter_Wear_Project\results") / SUBFOLDER_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent1': '#F18F01',
    'accent2': '#C73E1D',
    'accent3': '#3B1F2B',
    'clean': '#28A745',
    'raw': '#6C757D',
    'steady': '#17A2B8',
    'text': '#333333',
}


def plot_before_after_comparison():
    """绘制五元组参数清洗前后对比图"""
    print("生成图1: 清洗前后数据对比...")

    raw_path = "TBM_Cutter_Wear_Project/data/raw/260210.csv"
    clean_path = "TBM_Cutter_Wear_Project/data/processed/steady_260210.csv"

    raw_chunk = pd.read_csv(raw_path, chunksize=50000, encoding='gbk', low_memory=False)
    df_raw = next(raw_chunk)
    df_clean = pd.read_csv(clean_path, encoding='utf-8') if Path(clean_path).exists() else pd.DataFrame()

    if df_clean.empty:
        print("  [警告] 清洗后数据为空，跳过对比图")
        return

    if '环号' in df_clean.columns and len(df_clean) > 500:
        ring_sample = df_clean['环号'].iloc[len(df_clean)//2]
        df_clean_ring = df_clean[df_clean['环号'] == ring_sample]
    else:
        df_clean_ring = df_clean.head(2000)

    step = max(1, len(df_raw) // 3000)
    df_raw_sample = df_raw.iloc[::step].reset_index(drop=True)

    step2 = max(1, len(df_clean_ring) // 2000)
    df_clean_sample = df_clean_ring.iloc[::step2].reset_index(drop=True) if len(df_clean_ring) > 0 else pd.DataFrame()

    params = ['总推进力', '刀盘扭矩', '推进速度', '刀盘转速', '贯入度']
    units = ['kN', 'kN·m', 'mm/min', 'rpm', 'mm/rev']
    colors_params = [COLORS['primary'], COLORS['secondary'], COLORS['accent1'], COLORS['accent2'], COLORS['accent3']]

    fig, axes = plt.subplots(5, 2, figsize=(14, 12))

    for i, (param, unit, color) in enumerate(zip(params, units, colors_params)):
        ax_raw = axes[i, 0]
        ax_clean = axes[i, 1]

        if param in df_raw_sample.columns:
            y_raw = df_raw_sample[param].values
            ax_raw.plot(range(len(y_raw)), y_raw, color=COLORS['raw'], linewidth=0.5, alpha=0.7)
            ax_raw.fill_between(range(len(y_raw)), 0, y_raw, color=COLORS['raw'], alpha=0.2)
            ax_raw.set_ylabel(f'{param}\n({unit})', fontsize=9)
            ax_raw.set_ylim(bottom=0)
            ax_raw.grid(True, alpha=0.3, linestyle='--')
            zero_mask = y_raw == 0
            if zero_mask.sum() > 0:
                ax_raw.fill_between(range(len(y_raw)), 0, y_raw,
                                    where=zero_mask, color=COLORS['accent2'], alpha=0.3, label='Shutdown')

        if not df_clean_sample.empty and param in df_clean_sample.columns:
            y_clean = df_clean_sample[param].values
            ax_clean.plot(range(len(y_clean)), y_clean, color=COLORS['clean'], linewidth=0.5, alpha=0.8)
            ax_clean.fill_between(range(len(y_clean)), 0, y_clean, color=COLORS['clean'], alpha=0.2)
            ax_clean.set_ylabel(f'{param}\n({unit})', fontsize=9)
            ax_clean.set_ylim(bottom=0)
            ax_clean.grid(True, alpha=0.3, linestyle='--')

        if i == 4:
            ax_raw.set_xlabel('Time Index (Sample Points)', fontsize=9)
            ax_clean.set_xlabel('Time Index (Sample Points)', fontsize=9)

    axes[0, 0].set_title('原始数据（含停机与异常）', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('清洗后数据（稳态纯掘进段）', fontsize=11, fontweight='bold')
    fig.suptitle('五元组参数清洗前后对比 - 260210示例', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIR / 'fig1_before_after_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  已保存: fig1_before_after_comparison.png")


def plot_pipeline_flowchart():
    """绘制三阶段串行清洗架构流程图 - 顶级期刊风格"""
    print("生成图2: 三阶段串行清洗架构流程图...")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 颜色定义
    STAGE_BG = '#D6E4F0'
    STAGE_BORDER = '#2E86AB'
    REJECT_BG = '#FDE8E8'
    REJECT_BORDER = '#C73E1D'
    INPUT_BG = '#EBF3FB'
    INPUT_BORDER = '#4A76A8'
    OUTPUT_BG = '#E8F8E8'
    OUTPUT_BORDER = '#28A745'
    ARROW_COLOR = '#2B4C7E'
    LABEL_BG = '#FFF8E8'  # 淡黄色标签背景

    def draw_box(ax, x, y, w, h, lines, bg, border, fontsize=8.5, bold=False):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.04,rounding_size=0.12",
                             facecolor=bg, edgecolor=border, linewidth=1.8)
        ax.add_patch(box)
        text = '\n'.join(lines)
        ax.text(x, y, text, fontsize=fontsize, ha='center', va='center',
                color='#333333', family='sans-serif',
                fontweight='bold' if bold else 'normal',
                multialignment='center')

    def draw_arrow(ax, x1, y1, x2, y2, color=ARROW_COLOR, lw=2.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                   mutation_scale=15,
                                   connectionstyle='arc3,rad=0'))

    def draw_label_box(ax, x, y, text, fontsize=10):
        """绘制带边框的标签"""
        box = FancyBboxPatch((x - 0.6, y - 0.25), 1.2, 0.5,
                             boxstyle="round,pad=0.05,rounding_size=0.08",
                             facecolor=LABEL_BG, edgecolor=ARROW_COLOR, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=fontsize, ha='center', va='center',
                color=ARROW_COLOR, fontweight='bold', family='sans-serif')

    def draw_reject_arrow(ax, x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#999999', lw=1.2,
                                   mutation_scale=10, linestyle='dashed',
                                   connectionstyle='arc3,rad=-0.2'))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2, label, fontsize=8,
                    ha='center', va='center', color=REJECT_BORDER,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=REJECT_BG,
                             edgecolor=REJECT_BORDER, alpha=0.9))

    # ========== 布局参数 ==========
    LEFT_MARGIN = 1.2
    NODE_CENTER_X = 5.5

    # 各节点Y坐标
    OUTPUT_Y = 1.0
    STAGE3_Y = 2.9
    STAGE2_Y = 4.9
    STAGE1_Y = 6.9
    INPUT_Y = 8.9

    # ========== 左侧步骤标注（加大加粗）==========
    steps = ['输入', '阶段一', '阶段二', '阶段三', '输出']
    y_positions = [INPUT_Y, STAGE1_Y, STAGE2_Y, STAGE3_Y, OUTPUT_Y]
    for label, y in zip(steps, y_positions):
        ax.text(LEFT_MARGIN, y, label, fontsize=12, ha='center', va='center',
                color=STAGE_BORDER, fontweight='bold', family='sans-serif',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F7FF',
                         edgecolor=STAGE_BORDER, linewidth=1.5, alpha=0.9))

    # ========== 箭头标签（带框，对齐对应阶段）==========
    # 标签位置在左侧，水平虚线连接箭头
    arrow_labels = ['稳态切片', '平滑切片', '空间裁剪', '纯净输出']
    arrow_y_positions = [
        (STAGE1_Y + INPUT_Y) / 2 + 0.1,
        (STAGE2_Y + STAGE1_Y) / 2 + 0.1,
        (STAGE3_Y + STAGE2_Y) / 2 + 0.1,
        (OUTPUT_Y + STAGE3_Y) / 2 + 0.1
    ]

    for label, y in zip(arrow_labels, arrow_y_positions):
        # 标签框
        draw_label_box(ax, LEFT_MARGIN + 0.5, y, label, fontsize=10)
        # 水平虚线连接到主流程
        ax.plot([LEFT_MARGIN + 1.1, NODE_CENTER_X - 1.8], [y, y],
                color=ARROW_COLOR, lw=1.2, linestyle=':', alpha=0.6)
        # 小箭头指向主流程
        ax.annotate('', xy=(NODE_CENTER_X - 1.8, y), xytext=(LEFT_MARGIN + 1.1, y),
                    arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.5,
                                   mutation_scale=12))

    # ========== 节点布局（居中偏右）==========
    # 输入节点
    draw_box(ax, NODE_CENTER_X, INPUT_Y, 3.2, 0.75,
             ['原始海量时序', '500 MB/日 · 1390+ 通道 · 秒级采样'],
             INPUT_BG, INPUT_BORDER, fontsize=9, bold=True)

    # 阶段一
    draw_box(ax, NODE_CENTER_X, STAGE1_Y, 3.6, 1.65,
             ['阶段一：稳态物理剥离 (流形平滑)',
              '─' * 22,
              '五元组联合约束判据：',
              '推力 F > F_set, 扭矩 T > T_set',
              '推进速度 v > v_set, 转速 N > N_set',
              '仓压稳定 ∈ [P_min, P_max]',
              '时间窗口 Δt ≥ 300 s',
              '─' * 22,
              '剔除：停机/换步/推进过渡态碎片'],
             STAGE_BG, STAGE_BORDER, fontsize=7.5, bold=True)

    # 阶段一剔除（右侧）
    draw_reject_arrow(ax, NODE_CENTER_X + 1.9, STAGE1_Y - 0.15, 8.3, STAGE1_Y - 0.15, '剔除')
    draw_box(ax, 8.8, STAGE1_Y - 0.15, 0.9, 0.45,
             ['停机/换步', '过渡态'],
             REJECT_BG, REJECT_BORDER, fontsize=7)

    # 阶段二
    draw_box(ax, NODE_CENTER_X, STAGE2_Y, 3.6, 1.55,
             ['阶段二：Rolling MAD 时域自愈',
              '─' * 22,
              '动态抗差中位数基线追踪：',
              '滑动窗口 W = 60 s, 阈值系数 τ = 3.5',
              'MAD → σ 转换因子 k = 1.4826',
              '─' * 22,
              '毛刺检出 → ffill 前向自愈',
              '保留：真实力学瞬态冲击'],
             STAGE_BG, STAGE_BORDER, fontsize=7.5, bold=True)

    # 阶段二剔除（右侧）
    draw_reject_arrow(ax, NODE_CENTER_X + 1.9, STAGE2_Y - 0.15, 8.3, STAGE2_Y - 0.15, '滤除')
    draw_box(ax, 8.8, STAGE2_Y - 0.15, 0.9, 0.45,
             ['高频噪声', '传感器毛刺'],
             REJECT_BG, REJECT_BORDER, fontsize=7)

    # 阶段三
    draw_box(ax, NODE_CENTER_X, STAGE3_Y, 3.6, 1.55,
             ['阶段三：多维 LOF 空间裁剪',
              '─' * 22,
              '耦合特征空间构建：',
              '[ 推力F, 扭矩T, 电流I, 贯入度p ]',
              'k-近邻 = 30, 污染率 ν = 0.005',
              '─' * 22,
              '计算可达密度拓扑',
              '整体剪除：通信死点/传导失真行'],
             STAGE_BG, STAGE_BORDER, fontsize=7.5, bold=True)

    # 阶段三剔除（右侧）
    draw_reject_arrow(ax, NODE_CENTER_X + 1.9, STAGE3_Y - 0.15, 8.3, STAGE3_Y - 0.15, '剪除')
    draw_box(ax, 8.8, STAGE3_Y - 0.15, 0.9, 0.45,
             ['通信死点', '传导失真'],
             REJECT_BG, REJECT_BORDER, fontsize=7)

    # 输出节点
    draw_box(ax, NODE_CENTER_X, OUTPUT_Y, 3.2, 0.65,
             ['纯净稳态张量', '保留全量1390+通道 · 时序连续性完整'],
             OUTPUT_BG, OUTPUT_BORDER, fontsize=9, bold=True)

    # ========== 主流程箭头（居中偏右）==========
    draw_arrow(ax, NODE_CENTER_X, INPUT_Y - 0.38, NODE_CENTER_X, STAGE1_Y + 0.83)
    draw_arrow(ax, NODE_CENTER_X, STAGE1_Y - 0.83, NODE_CENTER_X, STAGE2_Y + 0.78)
    draw_arrow(ax, NODE_CENTER_X, STAGE2_Y - 0.78, NODE_CENTER_X, STAGE3_Y + 0.78)
    draw_arrow(ax, NODE_CENTER_X, STAGE3_Y - 0.78, NODE_CENTER_X, OUTPUT_Y + 0.33)

    # ========== 图名（顶部，加大加粗）==========
    ax.text(5.5, 9.7, '盾构机多源参数物理级联清洗管线',
            fontsize=14, ha='center', va='center', fontweight='bold', color='#222222')

    # ========== 图例说明（底部）==========
    ax.text(5.5, 0.2,
            'F: 推力  T: 扭矩  v: 推进速度  N: 刀盘转速  p: 贯入度  |  '
            'W: 窗口宽度  τ: MAD阈值系数  k: 鲁棒标准差因子  ν: LOF污染率',
            fontsize=8, ha='center', va='center', color='#555555', family='sans-serif')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig2_pipeline_flowchart.png', bbox_inches='tight',
                dpi=300, facecolor='white', edgecolor='none')
    plt.close()
    print(f"  已保存: fig2_pipeline_flowchart.png")


def plot_steady_mask_effect():
    """可视化五元组稳态判据的效果"""
    print("生成图3: 稳态判定效果...")

    raw_path = "TBM_Cutter_Wear_Project/data/raw/260210.csv"
    df_raw = pd.read_csv(raw_path, chunksize=100000, encoding='gbk', low_memory=False)
    df_raw = next(df_raw)

    step = max(1, len(df_raw) // 3000)
    df_sample = df_raw.iloc[::step].reset_index(drop=True)

    mask_f = df_sample['总推进力'] > 0
    mask_t = df_sample['刀盘扭矩'] > 0
    mask_v = df_sample['推进速度'] > 0
    mask_n = df_sample['刀盘转速'] > 0
    mask_p = df_sample['贯入度'] > 0
    steady_mask = mask_f & mask_t & mask_v & mask_n & mask_p

    fig, axes = plt.subplots(6, 1, figsize=(14, 10), sharex=True)

    params_info = [
        ('总推进力', 'kN', COLORS['primary']),
        ('刀盘扭矩', 'kN·m', COLORS['secondary']),
        ('推进速度', 'mm/min', COLORS['accent1']),
        ('刀盘转速', 'rpm', COLORS['accent2']),
        ('贯入度', 'mm/rev', COLORS['accent3']),
    ]

    x = range(len(df_sample))

    for i, (param, unit, color) in enumerate(params_info):
        ax = axes[i]
        y = df_sample[param].values
        y_plot = np.where(y > 0, y, np.nan)

        ax.plot(x, y_plot, color=color, linewidth=0.5, alpha=0.8)
        ax.fill_between(x, 0, np.where(y > 0, y, 0), color=color, alpha=0.15)
        ax.fill_between(x, 0, np.where(y <= 0, 1, 0), color=COLORS['accent2'], alpha=0.4, label='零值区间')
        ax.set_ylabel(f'{param}\n({unit})', fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=7, ncol=2)

    ax = axes[5]
    ax.fill_between(x, 0, steady_mask.astype(int), color=COLORS['steady'], alpha=0.6, label='综合稳态判据')
    ax.set_ylabel('稳态\n掩码', fontsize=9)
    ax.set_ylim(-0.1, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['停机', '正常推进'])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('时间索引（采样点）', fontsize=10)

    total = len(df_sample)
    n_steady = steady_mask.sum()
    stat_text = f'总采样点：{total:,} | 稳态点：{n_steady:,} ({n_steady/total*100:.1f}%) | 停机点：{total-n_steady:,} ({(total-n_steady)/total*100:.1f}%)'
    fig.text(0.5, 0.02, stat_text, ha='center', fontsize=10, style='italic')

    fig.suptitle('五元组稳态判定效果（260210示例）', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(OUTPUT_DIR / 'fig3_steady_mask_effect.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  已保存: fig3_steady_mask_effect.png")


def plot_cleaning_statistics():
    """绘制清洗前后数据量对比"""
    print("生成图4: 清洗统计...")

    days = ['260207', '260208', '260209', '260210', '260211']
    raw_counts = []
    clean_counts = []

    for day in days:
        raw_path = f"TBM_Cutter_Wear_Project/data/raw/{day}.csv"
        clean_path = f"TBM_Cutter_Wear_Project/data/processed/steady_{day}.csv"

        raw_count = sum(1 for _ in pd.read_csv(raw_path, usecols=['日期'], chunksize=50000, encoding='gbk'))
        raw_counts.append(raw_count * 50000)

        if Path(clean_path).exists():
            df_clean = pd.read_csv(clean_path, encoding='utf-8')
            clean_counts.append(len(df_clean))
        else:
            clean_counts.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    x = np.arange(len(days))
    width = 0.35

    bars1 = ax1.bar(x - width/2, raw_counts, width, label='原始数据', color=COLORS['raw'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, clean_counts, width, label='清洗后数据', color=COLORS['clean'], alpha=0.8)

    ax1.set_xlabel('日期', fontsize=11)
    ax1.set_ylabel('数据行数', fontsize=11)
    ax1.set_title('清洗前后数据量对比', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(days)
    ax1.legend(fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax1.annotate(f'{height/1000:.1f}k',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax2 = axes[1]
    retention_rates = [c/r*100 if r > 0 else 0 for c, r in zip(clean_counts, raw_counts)]
    colors_ret = [COLORS['steady'] if r > 30 else COLORS['accent2'] for r in retention_rates]

    bars = ax2.bar(days, retention_rates, color=colors_ret, alpha=0.8, edgecolor='white', linewidth=1)

    ax2.set_xlabel('日期', fontsize=11)
    ax2.set_ylabel('保留率（%）', fontsize=11)
    ax2.set_title('稳态数据保留率', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='50%基准线')
    ax2.legend(fontsize=10)

    for bar, rate in zip(bars, retention_rates):
        height = bar.get_height()
        ax2.annotate(f'{rate:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig4_cleaning_statistics.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  已保存: fig4_cleaning_statistics.png")


def plot_mad_denoise_effect():
    """展示 Rolling MAD 去噪效果"""
    print("生成图5: MAD去噪效果...")

    clean_path = "TBM_Cutter_Wear_Project/data/processed/steady_260210.csv"
    if not Path(clean_path).exists():
        print("  [跳过] 清洗后数据不存在")
        return

    df_clean = pd.read_csv(clean_path, encoding='utf-8')
    if len(df_clean) < 500:
        print("  [跳过] 数据量不足")
        return

    step = max(1, len(df_clean) // 1500)
    df_sample = df_clean.iloc[::step].reset_index(drop=True)

    params = ['总推进力', '刀盘扭矩']
    units = ['kN', 'kN·m']
    colors_p = [COLORS['primary'], COLORS['secondary']]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for i, (param, unit, color) in enumerate(zip(params, units, colors_p)):
        if param not in df_sample.columns:
            continue

        ax_raw = axes[i, 0]
        y = df_sample[param].values

        ax_raw.plot(range(len(y)), y, color=color, linewidth=0.6, alpha=0.8)
        ax_raw.fill_between(range(len(y)), 0, y, color=color, alpha=0.2)
        ax_raw.set_title(f'{param} 清洗后数据', fontsize=11, fontweight='bold')
        ax_raw.set_ylabel(f'{param} ({unit})', fontsize=10)
        ax_raw.grid(True, alpha=0.3, linestyle='--')
        ax_raw.set_ylim(bottom=0)

        ax_stats = axes[i, 1]

        window = 60
        if len(y) > window:
            rolling_mean = pd.Series(y).rolling(window=window, center=True).mean()
            rolling_std = pd.Series(y).rolling(window=window, center=True).std() * 1.4826

            ax_stats.plot(range(len(y)), y, color=color, linewidth=0.5, alpha=0.4, label='原始')
            ax_stats.plot(range(len(y)), rolling_mean, color='black', linewidth=1.5,
                         label=f'滑动均值 W={window}', linestyle='--')
            ax_stats.fill_between(range(len(y)),
                                  rolling_mean - 3.5 * rolling_std,
                                  rolling_mean + 3.5 * rolling_std,
                                  color='orange', alpha=0.2, label='MAD阈值区间(±3.5σ)')
            ax_stats.set_title(f'{param} 局部统计特征', fontsize=11, fontweight='bold')
            ax_stats.set_ylabel(f'{param} ({unit})', fontsize=10)
            ax_stats.legend(loc='upper right', fontsize=8)
            ax_stats.grid(True, alpha=0.3, linestyle='--')
            ax_stats.set_ylim(bottom=0)

        axes[i, 0].set_xlabel('时间索引（采样点）', fontsize=9)
        axes[i, 1].set_xlabel('时间索引（采样点）', fontsize=9)

    fig.suptitle('Rolling MAD 去噪效果图（260210示例）', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIR / 'fig5_mad_denoise_effect.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  已保存: fig5_mad_denoise_effect.png")


def generate_all_figures():
    """生成所有图表"""
    print("=" * 60)
    print("论文图表生成 - 模块一：稳态剥离与清洗可视化")
    print("=" * 60)

    print(f"\n输出目录: {OUTPUT_DIR}")
    print(f"子文件夹: {SUBFOLDER_NAME}")

    import shutil
    for img_file in OUTPUT_DIR.glob("*.png"):
        shutil.copy(img_file, Path(r"C:\NO CLUB\experience\TBM_Cutter_Wear_Project\results"))

    plot_before_after_comparison()
    plot_pipeline_flowchart()
    plot_steady_mask_effect()
    plot_cleaning_statistics()
    plot_mad_denoise_effect()

    print("\n" + "=" * 60)
    print("所有图表生成完成！")
    print(f"输出路径: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_figures()