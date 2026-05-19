"""
7维全局独立核心特征——相关性分析与贪心选取可视化
=================================================
生成以下图表（顶级期刊排版）：
  1. 7维全局Spearman秩相关系数矩阵热力图（中文标签）
  2. 贪心选取算法可视化（各簇候选特征与已选集合的全局平均相关性）

Author: TBM Cutter Wear Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "feature_selection_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUTTERHEAD_AREA = 188.7
EPSILON = 1e-6

# 三大物理簇候选特征
ELECTRIC_CANDIDATES = ['P2.1泵电流', 'P0.1泵电流', 'P0.2泵电流', 'P1.1泵电流', 'P0.3泵电流']
THERMAL_CANDIDATES = ['主油箱油温', '齿轮油温', '齿轮油温2', '破碎机温度', '内循环水温度']
FLUID_CANDIDATES = ['泥水仓顶部1压力', '泥水仓顶部2压力', '泥水仓左中下压力',
                    '泥水仓右中上压力', '主进浆流量', '主排浆流量',
                    '进浆密度', '排浆密度', '进浆压力', '泥水仓液位', '内循环水压力']

# 7维全局独立核心特征
FINAL_7D = ['SE', 'FPI', 'TPI', 'P2.1泵电流', '主油箱油温', '泥水仓顶部1压力', '排浆密度']

# 中文简称（用于图形标签）
NAME_CN = {
    'SE': 'SE\n(切削比能)',
    'FPI': 'FPI\n(推力贯入度指数)',
    'TPI': 'TPI\n(扭矩贯入度指数)',
    'P2.1泵电流': 'P2.1泵电流',
    '主油箱油温': '主油箱油温',
    '泥水仓顶部1压力': '泥水仓\n顶部1压力',
    '排浆密度': '排浆密度',
}

# 英文简称
NAME_EN = {
    'SE': 'SE',
    'FPI': 'FPI',
    'TPI': 'TPI',
    'P2.1泵电流': 'I_p21',
    '主油箱油温': 'T_oil',
    '泥水仓顶部1压力': 'P_top',
    '排浆密度': 'ρ_out',
}


def load_and_derive(n_rows=20000):
    """加载数据并派生高阶不变量"""
    df = pd.read_csv(PROCESSED_DIR / "steady_260207.csv", low_memory=False, nrows=n_rows)
    Fv = df['总推进力'].values
    T = df['刀盘扭矩'].values
    v = df['推进速度'].values
    n = df['刀盘转速'].values
    with np.errstate(divide='ignore', invalid='ignore'):
        p_calc = np.where(n > 0, v / (n + EPSILON), np.nan)
        df['SE'] = np.where(np.isfinite(p_calc),
                            (Fv / CUTTERHEAD_AREA) + (2*np.pi*T) / (CUTTERHEAD_AREA * (p_calc + EPSILON)),
                            np.nan)
        df['FPI'] = np.where(np.isfinite(p_calc), Fv / (p_calc + EPSILON), np.nan)
        df['TPI'] = np.where(np.isfinite(p_calc), T / (p_calc + EPSILON), np.nan)
    return df


def compute_spearman(df, cols):
    """计算有效列的Spearman相关性矩阵"""
    available = [c for c in cols if c in df.columns
                 and not df[c].isna().all()
                 and df[c].std() > EPSILON]
    df_sub = df[available].apply(pd.to_numeric, errors='coerce').dropna()
    if len(df_sub) < 100:
        return None, []
    return df_sub.corr(method='spearman'), available


def greedy_select_global(df, candidates, already_selected, max_features=2):
    """
    全局贪心选取：每步选与已选集合平均相关性最低的特征

    Args:
        df: DataFrame
        candidates: 候选特征列表
        already_selected: 已经选中的特征列表（来自其他簇）
        max_features: 最多选几个

    Returns:
        selected: 选中的特征列表
        selection_scores: 每步选取的平均|r|分数
    """
    # 过滤常量列
    available = [c for c in candidates
                 if c in df.columns
                 and not df[c].isna().all()
                 and df[c].std() > EPSILON]
    if not available:
        return [], {}

    df_sub = df[available].apply(pd.to_numeric, errors='coerce')
    valid_available = [c for c in available if df_sub[c].isna().mean() < 0.5]
    if not valid_available:
        return [], {}

    corr_mat = df_sub[valid_available].corr(method='spearman')

    selected = []
    selection_scores = {}

    remaining = valid_available.copy()

    while remaining and len(selected) < max_features:
        best_score = float('inf')
        best_c = None

        for c in remaining:
            if not selected:
                # 第一个：选标准差最大的（信息量最丰富）
                score = -df_sub[c].std()
            else:
                # 计算与已选特征的平均|r|（越小越独立）
                row = corr_mat.loc[c, selected].dropna()
                score = np.abs(row).mean() if len(row) > 0 else 0.0

            if score < best_score:
                best_score = score
                best_c = c

        if best_c:
            selected.append(best_c)
            remaining.remove(best_c)
            if remaining:
                row = corr_mat.loc[best_c, remaining].dropna()
                final_score = np.abs(row).mean() if len(row) > 0 else 0.0
                selection_scores[best_c] = final_score

    return selected, selection_scores


# =========================================================================
# 图1：7维全局Spearman相关性矩阵（顶级期刊排版）
# =========================================================================
def plot_7d_correlation_matrix(df):
    """绘制7维全局Spearman秩相关系数矩阵"""
    corr_7d, avail_7d = compute_spearman(df, FINAL_7D)
    if corr_7d is None:
        print("  [错误] 数据不足，无法计算相关性")
        return

    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor='white')

    # 使用中文简称作为标签
    labels = [NAME_EN.get(c, c) for c in avail_7d]
    mat = corr_7d.values

    # 绘制热力图
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # 设置刻度标签
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_yticklabels(labels, fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)

    # 标注数值
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            # 高相关对用白色字，低相关对用黑色字
            color = 'white' if abs(val) > 0.6 else 'black'
            fontweight = 'bold' if abs(val) > 0.5 else 'normal'
            fontsize = 11 if abs(val) > 0.5 else 10
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=fontsize, color=color, fontweight=fontweight)

    # 高亮高相关对（|r|>0.7）- 金色边框
    for i, c1 in enumerate(avail_7d):
        for j, c2 in enumerate(avail_7d):
            if i != j and abs(corr_7d.loc[c1, c2]) > 0.7:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                           fill=False, edgecolor='gold', linewidth=2.5))

    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Spearman 秩相关系数', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # 标题
    ax.set_title('7维全局独立核心特征 Spearman 秩相关系数矩阵', fontsize=13, fontweight='bold', pad=12)

    # 添加图例说明
    legend_text = '金色边框: |r|>0.7 (高相关对)'
    ax.text(1.02, 0.02, legend_text, transform=ax.transAxes, fontsize=9,
            va='bottom', ha='left', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gold', alpha=0.9))

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'fig1_7d_correlation_matrix.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [图1] 已保存: {output_path.name}")
    plt.close()

    return corr_7d, avail_7d


# =========================================================================
# 图2：贪心选取算法可视化（各簇候选特征的全局平均相关性）
# =========================================================================
def plot_greedy_selection_criteria(df):
    """
    展示贪心选取算法原理：
    对每个簇的候选特征，计算其与已选中特征（来自所有其他簇）的全局平均|r|

    这解释了一个问题：为什么选P2.1泵电流而不是其他泵电流？
    答案：因为P2.1泵电流与SE/FPI/TPI/已选热力/流体特征的全局平均相关性最低
    """
    # 先计算SE, FPI, TPI（已经固定选中）
    Fv = df['总推进力'].values
    T = df['刀盘扭矩'].values
    v = df['推进速度'].values
    n = df['刀盘转速'].values
    with np.errstate(divide='ignore', invalid='ignore'):
        p_calc = np.where(n > 0, v / (n + EPSILON), np.nan)
        df['SE'] = np.where(np.isfinite(p_calc),
                            (Fv / CUTTERHEAD_AREA) + (2*np.pi*T) / (CUTTERHEAD_AREA * (p_calc + EPSILON)),
                            np.nan)
        df['FPI'] = np.where(np.isfinite(p_calc), Fv / (p_calc + EPSILON), np.nan)
        df['TPI'] = np.where(np.isfinite(p_calc), T / (p_calc + EPSILON), np.nan)

    # 已选中的特征（3个高阶不变量）
    already_selected_3 = ['SE', 'FPI', 'TPI']

    # 计算每个候选特征与已选特征的全局平均相关性
    def compute_global_avg_corr(df, candidate, selected_features):
        """计算候选特征与已选特征的全局平均|r|"""
        available = [c for c in [candidate] + selected_features
                     if c in df.columns and not df[c].isna().all() and df[c].std() > EPSILON]
        if len(available) < 2:
            return None
        df_sub = df[available].apply(pd.to_numeric, errors='coerce').dropna()
        if len(df_sub) < 100:
            return None
        corr_mat = df_sub.corr(method='spearman')
        if candidate not in corr_mat.index or candidate not in corr_mat.columns:
            return None
        row = corr_mat.loc[candidate, selected_features].dropna()
        return np.abs(row).mean()

    # 电气簇分析
    elec_results = {}
    for c in ELECTRIC_CANDIDATES:
        if c in df.columns:
            score = compute_global_avg_corr(df, c, already_selected_3)
            if score is not None:
                elec_results[c] = score

    # 热力簇分析（在电气簇选完后，已选中包含SE/FPI/TPI + 电气选中）
    already_selected_4 = already_selected_3 + ['P2.1泵电流']
    therm_results = {}
    for c in THERMAL_CANDIDATES:
        if c in df.columns:
            score = compute_global_avg_corr(df, c, already_selected_4)
            if score is not None:
                therm_results[c] = score

    # 流体簇分析（在电气+热力选完后）
    already_selected_5 = already_selected_4 + ['主油箱油温']
    fluid_results = {}
    for c in FLUID_CANDIDATES:
        if c in df.columns:
            score = compute_global_avg_corr(df, c, already_selected_5)
            if score is not None:
                fluid_results[c] = score

    # 绘制三簇对比条形图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor='white')

    cluster_info = [
        ('电气簇\n(候选5个 → 选中1个)', elec_results, ['P2.1泵电流'], '#2E86AB'),
        ('热力簇\n(候选5个 → 选中1个)', therm_results, ['主油箱油温'], '#E07B39'),
        ('流体簇\n(候选11个 → 选中2个)', fluid_results, ['泥水仓顶部1压力', '排浆密度'], '#4A7C59'),
    ]

    for idx, (title, results, selected, color) in enumerate(cluster_info):
        ax = axes[idx]
        if not results:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=11, fontweight='bold')
            continue

        # 按avg |r|排序
        sorted_items = sorted(results.items(), key=lambda x: x[1])
        names = [NAME_EN.get(c, c.split('泵')[0] if '泵' in c else c[:6]) for c, _ in sorted_items]
        values = [v for _, v in sorted_items]

        # 短名称映射
        short_names = []
        for c, _ in sorted_items:
            if '泵电流' in c:
                short_names.append(c.replace('泵电流', ''))
            elif '压力' in c:
                short_names.append(c.replace('泥水仓', '').replace('压力', '压'))
            elif '温度' in c:
                short_names.append(c.replace('温度', ''))
            elif '密度' in c:
                short_names.append(c.replace('密度', ''))
            elif '流量' in c:
                short_names.append(c.replace('流量', ''))
            elif '液位' in c:
                short_names.append('液位')
            else:
                short_names.append(c[:5])

        x = np.arange(len(sorted_items))
        bars = ax.bar(x, values, color=color, alpha=0.75, edgecolor='white', linewidth=1.2)

        # 高亮选中的特征
        for i, (c, _) in enumerate(sorted_items):
            if c in selected:
                bars[i].set_edgecolor('gold')
                bars[i].set_linewidth(3)
                bars[i].set_alpha(1.0)

        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('全局平均 |Spearman r|', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(values) * 1.15)

        # 标注数值
        for bar, (_, val) in zip(bars, sorted_items):
            ax.text(bar.get_x() + bar.get_width()/2., val + max(values)*0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 图例
        legend_patch = mpatches.Patch(color=color, alpha=0.75, label='候选特征')
        selected_patch = mpatches.Patch(facecolor=color, edgecolor='gold', linewidth=3, label='最终选中')
        ax.legend(handles=[legend_patch, selected_patch], loc='upper right', fontsize=8)

    fig.suptitle('贪心选取算法：各簇候选特征与已选特征的全局平均相关性\n'
                 '(Global Average Correlation with Already-Selected Features)',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'fig2_greedy_selection_criteria.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [图2] 已保存: {output_path.name}")
    plt.close()

    return elec_results, therm_results, fluid_results


# =========================================================================
# 图3：贪心算法三步流程图
# =========================================================================
def plot_greedy_algorithm_flowchart():
    """绘制贪心选取算法的三步流程图"""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('贪心选取算法流程图\n(Greedy Feature Selection Algorithm)', fontsize=13, fontweight='bold', pad=15)

    # 步骤框
    steps = [
        (1.5, 3, '步骤一\n过滤常量列', '#E3F2FD', '剔除 std<1e-6\n的候选特征'),
        (5, 3, '步骤二\n计算相关性矩阵', '#E8F5E9', '构建候选特征间\nSpearman相关矩阵'),
        (8.5, 3, '步骤三\n贪心选取', '#FFF3E0', '每步选与已选集合\n平均|r|最低的特征'),
    ]

    for x, y, title, color, detail in steps:
        # 主框
        box = mpatches.FancyBboxPatch((x-1.2, y-0.8), 2.4, 1.6,
                                       boxstyle="round,pad=0.1",
                                       facecolor=color, edgecolor='#333', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y+0.3, title, ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(x, y-0.4, detail, ha='center', va='center', fontsize=9, style='italic')

    # 箭头
    arrow_style = dict(arrowstyle='->', color='#555', lw=2)
    ax.annotate('', xy=(3.5, 3), xytext=(2.7, 3), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 3), xytext=(6.2, 3), arrowprops=arrow_style)

    # 算法伪代码
    pseudo_code = """
贪心选取算法伪代码：

function greedy_select(candidates, already_selected, max_features):
    available = filter_valid_candidates(candidates)  # 步骤一
    corr_mat = compute_spearman_matrix(available)    # 步骤二

    selected = []
    remaining = available.copy()

    while remaining and len(selected) < max_features:  # 步骤三
        for c in remaining:
            if not selected:
                score = -std(c)  # 第一个：选信息量最大的
            else:
                score = mean(|corr_mat[c, selected]|)  # 全局平均r

            if score < best_score:
                best_score = score
                best_c = c

        selected.append(best_c)
        remaining.remove(best_c)

    return selected
"""
    ax.text(0.5, 1.8, pseudo_code, fontsize=9, family='monospace',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', fc='#F5F5F5', ec='#CCCCCC'))

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'fig3_greedy_algorithm_flowchart.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [图3] 已保存: {output_path.name}")
    plt.close()


# =========================================================================
# 主程序
# =========================================================================
def run():
    print("=" * 70)
    print("7维全局独立核心特征——相关性分析与贪心选取可视化")
    print("=" * 70)

    df = load_and_derive()
    print(f"\n数据加载完成: {df.shape}")

    print("\n[图1] 7维全局Spearman相关性矩阵...")
    corr_7d, avail_7d = plot_7d_correlation_matrix(df)

    print("\n[图2] 贪心选取算法可视化...")
    elec_r, therm_r, fluid_r = plot_greedy_selection_criteria(df)

    print("\n[图3] 贪心算法流程图...")
    plot_greedy_algorithm_flowchart()

    # 打印统计结果
    print("\n" + "=" * 70)
    print("统计结果汇总")
    print("=" * 70)

    if corr_7d is not None:
        print("\n【7维全局Spearman相关性矩阵】")
        display_corr = corr_7d.copy()
        display_corr.index = [NAME_EN.get(c, c) for c in display_corr.index]
        display_corr.columns = [NAME_EN.get(c, c) for c in display_corr.columns]
        print(display_corr.round(3).to_string())

        print("\n【高相关对 (|r|>0.7)】")
        for i, c1 in enumerate(avail_7d):
            for j, c2 in enumerate(avail_7d[i+1:], start=i+1):
                r = corr_7d.loc[c1, c2]
                if abs(r) > 0.7:
                    print(f"  {NAME_EN.get(c1,c1)} <-> {NAME_EN.get(c2,c2)}: r={r:+.3f}")

        print("\n【各特征冗余度统计】")
        for c in avail_7d:
            row = corr_7d.loc[c].drop(c, errors='ignore')
            n_high = (abs(row) > 0.7).sum()
            max_r = abs(row).max()
            print(f"  {NAME_EN.get(c,c)}: {n_high}个高相关对, max|r|={max_r:.3f}")

    print("\n【贪心选取结果】")
    print(f"  电气簇: {list(elec_r.keys()) if elec_r else 'N/A'}")

    print("\n输出文件清单：")
    import os
    for f in OUTPUT_DIR.glob('*.png'):
        print(f"  {f.name}: {os.path.getsize(f)/1024:.0f} KB")

    print("\n完成！")


if __name__ == "__main__":
    run()