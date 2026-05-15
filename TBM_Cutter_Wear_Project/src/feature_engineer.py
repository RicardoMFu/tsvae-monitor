"""
模块二：多物理场分组与特征衍生
=================================================
将1390+列传感器数据解耦为四大物理簇，并注入高阶物理不变量

功能：
1. 物理字段分簇提取（机械/电气/热力/流体）
2. 物理先验特征构造（SE, FPI, TPI）
3. 按环(Ring)聚合降维统计
4. 生成模块二可视化结果

Author: TBM Cutter Wear Project
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 四大物理簇字段映射（基于CLAUDE.md规划）
# =====================================================================

# 刀盘参数（机械动力簇核心）
MECHANIC_CLUSTER = [
    '总推进力', '刀盘扭矩', '推进速度', '刀盘转速', '贯入度',
    '左推力', '右推力', '上推力', '下推力',  # 推进分区压力
    '推进泵1压力', '推进泵2压力', '推进泵3压力',  # 推进系统压力
]

# 电气响应簇核心字段（基于实际数据列名）
ELECTRIC_CLUSTER = [
    'P2.1泵电流', 'P0.1泵电流', 'P0.2泵电流', 'P1.1泵电流', 'P0.3泵电流',
]

# 热力耗散簇核心字段
THERMAL_CLUSTER = [
    '主油箱油温', '齿轮油温', '破碎机温度', '内循环水温度',
    '电机温度', '减速机温度', '轴承温度',
]

# 流体平衡簇核心字段
FLUID_CLUSTER = [
    '泥水仓顶部1压力', '泥水仓顶部2压力', '排浆密度', '进浆密度',
    '主进浆流量', '主排浆流量', '泥水仓液位',
]

# 所有簇合并
ALL_CLUSTERS = MECHANIC_CLUSTER + ELECTRIC_CLUSTER + THERMAL_CLUSTER + FLUID_CLUSTER

# 物理常数
CUTTERHEAD_AREA = 188.7  # 刀盘截面积 (m²)，直径15.5m盾构机
EPSILON = 1e-6  # 防止除零溢出


# =====================================================================
# 特征工程核心函数
# =====================================================================

def extract_available_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """从DataFrame中提取实际存在的列，按物理簇分类

    Args:
        df: 稳态DataFrame（含1390+列）

    Returns:
        四大簇的实际可用列
    """
    available = {cluster: [] for cluster in ['mechanic', 'electric', 'thermal', 'fluid']}
    cluster_map = {
        'mechanic': MECHANIC_CLUSTER,
        'electric': ELECTRIC_CLUSTER,
        'thermal': THERMAL_CLUSTER,
        'fluid': FLUID_CLUSTER,
    }

    all_cols = df.columns.tolist()
    for cluster_name, cols in cluster_map.items():
        for col in cols:
            if col in all_cols:
                available[cluster_name].append(col)

    return available


def compute_physics_informed_features(df: pd.DataFrame) -> pd.DataFrame:
    """注入高阶物理不变量：贯入度、SE、FPI、TPI

    Args:
        df: 稳态DataFrame（含五元组核心参数）

    Returns:
        添加物理先验特征后的DataFrame
    """
    df = df.copy()

    # 获取核心参数
    Fv = df['总推进力'].values  # 推力 kN
    T = df['刀盘扭矩'].values   # 扭矩 kN·m
    v = df['推进速度'].values   # 推进速度 mm/min
    n = df['刀盘转速'].values   # 转速 rpm
    p = df['贯入度'].values     # 贯入度 mm/rev

    # ===== 1. 贯入度计算（如果原始数据不准确则重新计算）=====
    # p = v / n（mm/rev）
    with np.errstate(divide='ignore', invalid='ignore'):
        p_calc = v / (n + EPSILON)
        p_valid = np.where((n > 0) & (v > 0), p_calc, np.nan)

    df['贯入度_计算'] = p_valid

    # ===== 2. 切削比能 SE = (Fv/A) + (2π·T)/(A·p) =====
    # 单位: Fv(kN), T(kN·m), p(mm/rev), A(m²)
    # 结果单位: kN/m² = kPa (比能密度)
    SE = (Fv / CUTTERHEAD_AREA) + (2 * np.pi * T) / (CUTTERHEAD_AREA * (p_valid + EPSILON))
    df['SE_切削比能'] = np.where(np.isfinite(SE), SE, np.nan)

    # ===== 3. 推力贯入度指数 FPI = Fv / p =====
    FPI = Fv / (p_valid + EPSILON)
    df['FPI_推力贯入度指数'] = np.where(np.isfinite(FPI), FPI, np.nan)

    # ===== 4. 扭矩贯入度指数 TPI = T / p =====
    TPI = T / (p_valid + EPSILON)
    df['TPI_扭矩贯入度指数'] = np.where(np.isfinite(TPI), TPI, np.nan)

    # ===== 5. 额外物理特征 =====

    # 推进功率 (kW): Fv * v / 60 / 1000
    thrust_power = Fv * v / 60000
    df['推进功率_kW'] = thrust_power

    # 旋转功率 (kW): 2π·T·n / 60
    rotation_power = 2 * np.pi * T * n / 60
    df['旋转功率_kW'] = rotation_power

    # 总机械功率
    df['总机械功率_kW'] = thrust_power + rotation_power

    # 能量比 (旋转/推进)
    energy_ratio = rotation_power / (thrust_power + EPSILON)
    df['能量比_旋转_推进'] = np.where(np.isfinite(energy_ratio), energy_ratio, np.nan)

    return df


def compute_ring_aggregations(
    df: pd.DataFrame,
    ring_col: str = '环号'
) -> pd.DataFrame:
    """按环号进行统计聚合，为无监督学习准备特征矩阵

    Args:
        df: 添加物理特征后的DataFrame
        ring_col: 环号列名

    Returns:
        环级聚合特征矩阵，形状为 (环数, 特征数)
    """
    # 选择需要聚合的列（排除元数据列）
    exclude_cols = ['日期', '时间', '环号', '掘进时间', '行程']
    agg_cols = [c for c in df.columns if c not in exclude_cols and not c.endswith('_计算')]

    # 统计函数列表
    agg_funcs = {
        col: ['mean', 'std', 'skew', 'kurtosis'] for col in agg_cols
    }

    # 执行分组聚合（仅保留均值和标准差，避免kurtosis兼容性问题）
    ring_features = df.groupby(ring_col)[agg_cols].agg(['mean', 'std'])

    # 展平多级列名
    ring_features.columns = ['_'.join(col).strip() for col in ring_features.columns.values]

    # 添加环号作为索引名
    ring_features.index.name = ring_col

    return ring_features


def process_five_days_feature_engineering(
    data_dir: str = "TBM_Cutter_Wear_Project/data/processed",
    output_dir: str = "TBM_Cutter_Wear_Project/data/processed",
) -> Dict[str, pd.DataFrame]:
    """处理五日数据，完成模块二特征工程

    Args:
        data_dir: 清洗后数据目录
        output_dir: 输出目录

    Returns:
        各日特征工程结果统计
    """
    from pathlib import Path

    os.makedirs(output_dir, exist_ok=True)
    data_path = Path(data_dir)

    day_files = sorted(data_path.glob("steady_2602*.csv"))
    results = {}

    print("=" * 60)
    print("模块二：多物理场分组与特征衍生")
    print("=" * 60)

    for day_file in day_files:
        print(f"\n处理日期: {day_file.stem}")

        # 读取数据
        df = pd.read_csv(day_file, encoding='utf-8')

        # Step 1: 提取可用列
        available = extract_available_columns(df)
        print(f"  可用列分布: 机械{len(available['mechanic'])} "
              f"电气{len(available['electric'])} "
              f"热力{len(available['thermal'])} "
              f"流体{len(available['fluid'])}")

        # Step 2: 注入物理先验特征
        df_phys = compute_physics_informed_features(df)

        # 显示新特征统计
        print(f"  新增特征: SE={df_phys['SE_切削比能'].mean():.1f} kPa, "
              f"FPI={df_phys['FPI_推力贯入度指数'].mean():.1f}, "
              f"TPI={df_phys['TPI_扭矩贯入度指数'].mean():.1f}")

        # Step 3: 按环聚合
        ring_features = compute_ring_aggregations(df_phys)

        # 保存环级特征
        out_path = Path(output_dir) / f"features_{day_file.stem}.csv"
        ring_features.to_csv(out_path)

        # 统计
        results[day_file.stem] = {
            '原始行数': len(df),
            '环数': len(ring_features),
            '特征列数': len(ring_features.columns),
            '输出文件': str(out_path),
        }

        print(f"  环数: {len(ring_features)}, 特征列数: {len(ring_features.columns)}")
        print(f"  保存至: {out_path}")

    # 汇总
    print(f"\n{'='*60}")
    print("特征工程汇总")
    print(f"{'='*60}")
    total_rings = sum(v['环数'] for v in results.values())
    total_features = max(v['特征列数'] for v in results.values())
    print(f"  总环数: {total_rings}")
    print(f"  最大特征列数: {total_features}")

    return results


# =====================================================================
# 可视化模块
# =====================================================================

def generate_module2_visualizations(
    data_path: str = "data/processed",
    output_dir: str = None,
):
    """生成模块二可视化结果

    Args:
        data_path: 数据目录
        output_dir: 输出目录（默认自动创建带时间戳的子目录）
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.font_manager as fm
    import os
    import pandas as pd

    # 字体配置
    font_candidates = ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']
    Chinese_font_path = None
    for fp in font_candidates:
        if os.path.exists(fp):
            Chinese_font_path = fp
            break
    Chinese_font = fm.FontProperties(fname=Chinese_font_path).get_name() if Chinese_font_path else 'DejaVu Sans'

    plt.rcParams['font.sans-serif'] = [Chinese_font, 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300

    # 创建输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/module2_{timestamp}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("模块二可视化生成")
    print(f"{'='*60}")

    # 读取五日清洗后数据
    data_dir = Path(data_path)
    day_files = sorted(data_dir.glob("steady_2602*.csv"))

    # =====================================================================
    # 图6: 四大物理簇特征分布（Z-Score标准化版）
    # 解决不同物理量级导致的坐标轴视觉塌陷问题
    # =====================================================================
    print("生成图6: 四大物理簇特征分布（Z-Score标准化版）...")

    from scipy import stats

    cluster_info = [
        ('机械动力簇\nMechanical Dynamics', MECHANIC_CLUSTER, '#2E86AB'),
        ('电气响应簇\nElectrical Response', ELECTRIC_CLUSTER, '#A23B72'),
        ('热力耗散簇\nThermal Dissipation', THERMAL_CLUSTER, '#F18F01'),
        ('流体平衡簇\nFluid Balance', FLUID_CLUSTER, '#17A2B8'),
    ]

    all_dfs = []
    for day_file in day_files:
        df = pd.read_csv(day_file, encoding='utf-8')
        df['日期'] = day_file.stem
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for idx, (cluster_name, cluster_cols, color) in enumerate(cluster_info):
        ax = axes[idx // 2, idx % 2]

        # 提取该簇存在的列
        available_cols = [c for c in cluster_cols if c in combined_df.columns]

        if available_cols:
            # Z-Score 标准化：对每个特征独立标准化
            data_to_plot = []
            labels = []
            for col in available_cols:
                col_data = combined_df[col].dropna().values
                if len(col_data) > 0:
                    z_data = (col_data - col_data.mean()) / (col_data.std() + EPSILON)
                    # 每个特征按日取平均，再绘制分布
                    day_means = combined_df.groupby('日期')[col].mean().values
                    z_day_means = (day_means - col_data.mean()) / (col_data.std() + EPSILON)
                    data_to_plot.append(z_day_means)
                    labels.append(col)

            if data_to_plot:
                # 水平分条箱线图
                bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels,
                               vert=False, positions=range(len(labels)))

                # 设置颜色
                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.65)
                    patch.set_edgecolor('white')
                    patch.set_linewidth(1.5)

                for median in bp['medians']:
                    median.set_color('white')
                    median.set_linewidth(2)

                for flier in bp['fliers']:
                    flier.set(marker='o', markerfacecolor=color, alpha=0.3, markersize=3)

                ax.set_yticklabels(labels, fontsize=9)
                ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_title(cluster_name, fontsize=11, fontweight='bold', pad=10)
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        ax.set_xlabel('Z-Score (标准化分布)', fontsize=10)

    fig.suptitle('四大物理簇特征分布\n(Z-Score Normalized Distribution)',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_dir / 'fig6_physics_clusters.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  已保存: fig6_physics_clusters.png")

    # =====================================================================
    # 图7: 物理不变量随环号演化趋势（独立Y轴三行布局）
    # 解决SE/FPI/TPI不同量纲无法共享Y轴的问题
    # =====================================================================
    print("生成图7: 物理不变量随环号演化趋势...")

    # 读取所有五日数据计算环级统计
    all_ring_data = []

    for day_idx, day_file in enumerate(day_files):
        df = pd.read_csv(day_file, encoding='utf-8')

        # 计算物理特征
        Fv = df['总推进力'].values
        T = df['刀盘扭矩'].values
        v = df['推进速度'].values
        n = df['刀盘转速'].values
        p_calc = v / (n + EPSILON)

        SE = (Fv / CUTTERHEAD_AREA) + (2 * np.pi * T) / (CUTTERHEAD_AREA * (p_calc + EPSILON))
        FPI = Fv / (p_calc + EPSILON)
        TPI = T / (p_calc + EPSILON)

        if '环号' in df.columns:
            # 按环聚合：计算每环的统计量
            df_temp = df.copy()
            df_temp['SE'] = SE
            df_temp['FPI'] = FPI
            df_temp['TPI'] = TPI

            ring_stats = df_temp.groupby('环号').agg({
                'SE': ['mean', 'std'],
                'FPI': ['mean', 'std'],
                'TPI': ['mean', 'std']
            }).reset_index()
            ring_stats.columns = ['环号', 'SE_mean', 'SE_std', 'FPI_mean', 'FPI_std', 'TPI_mean', 'TPI_std']
            ring_stats['日期索引'] = day_idx
            ring_stats['日期标签'] = day_file.stem
            all_ring_data.append(ring_stats)

    # 合并所有数据
    all_rings_df = pd.concat(all_ring_data, ignore_index=True)

    # 创建三行子图
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    # 定义颜色映射
    day_colors = {'steady_260207': '#1f77b4', 'steady_260208': '#ff7f0e', 'steady_260209': '#2ca02c',
                   'steady_260210': '#d62728', 'steady_260211': '#9467bd'}
    markers = ['o', 's', '^', 'D', 'v']

    # 图7a: SE演化
    ax = axes[0]
    for day_idx, day_file in enumerate(day_files):
        day_data = all_rings_df[all_rings_df['日期标签'] == day_file.stem]
        if len(day_data) > 0:
            x_vals = range(len(day_data))
            y_vals = day_data['SE_mean'].values
            yerr = day_data['SE_std'].values if 'SE_std' in day_data.columns else None

            ax.plot(x_vals, y_vals, color=day_colors[day_file.stem],
                   marker=markers[day_idx], markersize=8, linewidth=2,
                   label=day_file.stem, alpha=0.9)

            if yerr is not None and len(yerr) > 0:
                ax.fill_between(x_vals,
                               y_vals - yerr,
                               y_vals + yerr,
                               color=day_colors[day_file.stem], alpha=0.15)

    ax.set_ylabel('SE (kPa)', fontsize=11, fontweight='bold')
    ax.set_title('(a) 切削比能演化趋势 | Specific Energy Evolution', fontsize=12, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=9, ncol=5, framealpha=0.9)
    ax.axhline(y=all_rings_df['SE_mean'].mean(), color='red', linestyle='--',
              linewidth=1.5, alpha=0.6, label='全局均值')

    # 图7b: FPI演化
    ax = axes[1]
    for day_idx, day_file in enumerate(day_files):
        day_data = all_rings_df[all_rings_df['日期标签'] == day_file.stem]
        if len(day_data) > 0:
            x_vals = range(len(day_data))
            y_vals = day_data['FPI_mean'].values
            yerr = day_data['FPI_std'].values if 'FPI_std' in day_data.columns else None

            ax.plot(x_vals, y_vals, color=day_colors[day_file.stem],
                   marker=markers[day_idx], markersize=8, linewidth=2,
                   label=day_file.stem, alpha=0.9)

            if yerr is not None and len(yerr) > 0:
                ax.fill_between(x_vals,
                               y_vals - yerr,
                               y_vals + yerr,
                               color=day_colors[day_file.stem], alpha=0.15)

    ax.set_ylabel('FPI (kN/mm)', fontsize=11, fontweight='bold')
    ax.set_title('(b) 推力贯入度指数演化趋势 | Force Penetration Index Evolution', fontsize=12, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=all_rings_df['FPI_mean'].mean(), color='red', linestyle='--',
              linewidth=1.5, alpha=0.6)

    # 图7c: TPI演化
    ax = axes[2]
    for day_idx, day_file in enumerate(day_files):
        day_data = all_rings_df[all_rings_df['日期标签'] == day_file.stem]
        if len(day_data) > 0:
            x_vals = range(len(day_data))
            y_vals = day_data['TPI_mean'].values
            yerr = day_data['TPI_std'].values if 'TPI_std' in day_data.columns else None

            ax.plot(x_vals, y_vals, color=day_colors[day_file.stem],
                   marker=markers[day_idx], markersize=8, linewidth=2,
                   label=day_file.stem, alpha=0.9)

            if yerr is not None and len(yerr) > 0:
                ax.fill_between(x_vals,
                               y_vals - yerr,
                               y_vals + yerr,
                               color=day_colors[day_file.stem], alpha=0.15)

    ax.set_ylabel('TPI (kN·m/mm)', fontsize=11, fontweight='bold')
    ax.set_xlabel('环号索引 | Ring Index', fontsize=11)
    ax.set_title('(c) 扭矩贯入度指数演化趋势 | Torque Penetration Index Evolution', fontsize=12, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=all_rings_df['TPI_mean'].mean(), color='red', linestyle='--',
              linewidth=1.5, alpha=0.6)

    # 设置统一的X轴标签
    max_rings = max(len(all_rings_df[all_rings_df['日期标签'] == f]) for f in all_rings_df['日期标签'].unique())
    axes[2].set_xticks(range(max_rings))
    axes[2].set_xticklabels([f'Ring {i+1}' for i in range(max_rings)], fontsize=10)

    fig.suptitle('物理不变量随掘进环号演化趋势\nPhysics Invariant Evolution with Ring Progression',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / 'fig7_physics_invariants.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  已保存: fig7_physics_invariants.png")

    # =====================================================================
    # 补充分析：跨物理场协同特征相关性矩阵（Spearman秩相关）
    # =====================================================================
    print("\n生成跨物理场协同特征相关性矩阵报告...")

    # 读取260210数据计算相关性
    df_corr = pd.read_csv(data_dir / "steady_260210.csv", encoding='utf-8')

    # 计算SE
    Fv = df_corr['总推进力'].values
    T = df_corr['刀盘扭矩'].values
    v = df_corr['推进速度'].values
    n = df_corr['刀盘转速'].values
    p_calc = v / (n + EPSILON)
    SE = (Fv / CUTTERHEAD_AREA) + (2 * np.pi * T) / (CUTTERHEAD_AREA * (p_calc + EPSILON))

    df_corr['SE'] = SE

    # 选择核心特征进行相关性分析
    core_features = []
    for feat_name, feat_col in [
        ('SE', 'SE'),
        ('主油箱油温', None),
        ('P2.1泵电流', None),
        ('总推进力', '总推进力'),
    ]:
        if feat_col is None:
            # 查找可能的列名
            possible_cols = [c for c in df_corr.columns if feat_name in c]
            if possible_cols:
                core_features.append((feat_name, possible_cols[0]))
        else:
            if feat_col in df_corr.columns:
                core_features.append((feat_name, feat_col))

    # 如果没找到P2.1泵电流，使用其他泵电流列
    if len(core_features) < 4:
        pump_cols = [c for c in df_corr.columns if '泵电流' in c or '电流' in c]
        if pump_cols and ('P2.1泵电流', pump_cols[0]) not in core_features:
            core_features.append(('泵电流', pump_cols[0]))

    # 提取数据
    corr_data = {}
    for name, col in core_features:
        if col in df_corr.columns:
            corr_data[name] = df_corr[col].values

    # 计算Spearman相关系数矩阵
    import pandas as pd
    corr_df = pd.DataFrame(corr_data)
    spearman_corr = corr_df.corr(method='spearman')

    print("\n" + "="*70)
    print("跨物理场协同特征相关性矩阵报告")
    print("Spearman Rank Correlation Matrix")
    print("="*70)
    print("\n核心特征列表:")
    for i, (name, _) in enumerate(core_features):
        print(f"  [{i}] {name}")

    print("\nSpearman相关系数矩阵:")
    print("-"*70)
    print(spearman_corr.round(3).to_string())

    print("\n关键发现:")
    # 找出相关性最强的特征对
    corr_pairs = []
    cols = spearman_corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            corr_pairs.append((cols[i], cols[j], spearman_corr.iloc[i, j]))

    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print("\n特征对相关性排序（按绝对值）:")
    for feat1, feat2, corr in corr_pairs:
        print(f"  {feat1} <-> {feat2}: {corr:.4f}")

    # 保存相关性矩阵到CSV
    spearman_corr.to_csv(output_dir / 'spearman_correlation_matrix.csv', encoding='utf-8-sig')
    print(f"\n相关性矩阵已保存至: {output_dir / 'spearman_correlation_matrix.csv'}")

    # =====================================================================
    # 图8: 环级特征热力图
    # =====================================================================
    print("生成图8: 环级特征热力图...")

    # 读取特征文件（如存在）
    feature_file = data_dir.parent / "features_260210.csv"
    if feature_file.exists():
        df_feat = pd.read_csv(feature_file, index_col=0)

        # 选择核心特征列
        core_cols = [c for c in df_feat.columns if any(x in c for x in ['SE', 'FPI', 'TPI', '总推进力', '刀盘扭矩'])]
        core_cols = core_cols[:20]  # 限制列数

        if len(core_cols) > 0:
            fig, ax = plt.subplots(figsize=(14, 8))
            data = df_feat[core_cols].T

            # 标准化绘制
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=data.values.min(), vmax=data.values.max())

            im = ax.imshow(data.values, aspect='auto', cmap='RdYlBu_r', norm=norm)
            ax.set_yticks(range(len(core_cols)))
            ax.set_yticklabels([c[:20] for c in core_cols], fontsize=8)
            ax.set_xlabel('环号索引', fontsize=10)
            ax.set_title('环级核心特征热力图', fontsize=12, fontweight='bold')

            plt.colorbar(im, ax=ax, label='标准化值')
            plt.tight_layout()
            fig.savefig(output_dir / 'fig8_ring_features_heatmap.png', bbox_inches='tight', dpi=300)
            plt.close()
    else:
        # 如果没有特征文件，创建模拟热力图
        print("  [跳过] 特征文件不存在")

    print(f"  已保存: fig8_ring_features_heatmap.png")

    # =====================================================================
    # 复制到results根目录
    # =====================================================================
    import shutil
    results_root = Path("TBM_Cutter_Wear_Project/results")
    for img_file in output_dir.glob("*.png"):
        shutil.copy(img_file, results_root)

    print(f"\n可视化结果已保存至: {output_dir}")
    return output_dir


# =====================================================================
# 主程序入口
# =====================================================================

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1 and sys.argv[1] == '--visualize-only':
        # 仅生成可视化
        generate_module2_visualizations()
    else:
        # 执行完整特征工程 + 可视化
        print("执行模块二特征工程...")
        results = process_five_days_feature_engineering()
        generate_module2_visualizations()

        print("\n模块二完成！")
        print(f"特征矩阵输出目录: TBM_Cutter_Wear_Project/data/processed/")
        print(f"可视化输出目录: TBM_Cutter_Wear_Project/results/")