"""模块一：稳态物理剥离与双重动态去噪

架构约束：严禁全量加载，依赖 Pandas chunksize 进行流式推断
核心原则：保留全量 1390+ 列特征，usecols 仅用于计算决策掩码，不做列过滤

五元组稳态判据（五值全部 > 0 表示正常推进，任一为 0 即停机）：
    - 总推进力 > 0 kN
    - 刀盘扭矩 > 0 kN·m
    - 推进速度 > 0 mm/min
    - 刀盘转速 > 0 rpm
    - 贯入度 > 0 mm/rev

双重去噪策略：
    1. Rolling MAD：单变量滑动绝对中位差去毛刺（针对高频突跳的力学通道）
    2. LOF：多物理场局部异常因子，解耦毛刺与真实力学瞬态冲击
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from typing import Optional, List


# =====================================================================
# 五元组物理阈值配置（阈值 > 0，五值全为正表示正常推进）
# =====================================================================
STEADY_STATE_THRESHOLDS = {
    "总推进力": 0.0,       # kN，> 0 表示正常推进
    "刀盘扭矩": 0.0,       # kN·m，> 0 表示正常推进
    "推进速度": 0.0,       # mm/min，> 0 表示正常推进
    "刀盘转速": 0.0,       # rpm，> 0 表示正常推进
    "贯入度": 0.0,         # mm/rev，> 0 表示正常推进
}

# Rolling MAD 去噪参数
MAD_WINDOW = 60          # 60秒滑动窗口
MAD_TAU = 3.5            # 阈值系数
MAD_K_FACTOR = 1.4826    # 将MAD转换为鲁棒标准差的系数

# LOF 多变量去噪参数
LOF_N_NEIGHBORS = 30     # 捕捉局部拓扑密度
LOF_CONTAMINATION = 0.005  # 仅捕获绝对脱离物理耦合流形的孤立毛刺

# 稳态连续性约束
MIN_STEADY_DURATION = 300  # 持续时间 >= 300秒 视为纯正稳态段

# 核心决策通道（仅用于计算稳态掩码，不做列过滤）
DECISION_COLS = [
    "总推进力", "刀盘扭矩", "推进速度", "刀盘转速", "贯入度"
]


def _extract_steady_state_mask(chunk: pd.DataFrame) -> pd.Series:
    """计算五元组稳态掩码（五值全 > 0 为正常推进，任一为 0 即停机）"""
    mask_thrust = chunk["总推进力"] > STEADY_STATE_THRESHOLDS["总推进力"]
    mask_torque = chunk["刀盘扭矩"] > STEADY_STATE_THRESHOLDS["刀盘扭矩"]
    mask_speed = chunk["推进速度"] > STEADY_STATE_THRESHOLDS["推进速度"]
    mask_rpm = chunk["刀盘转速"] > STEADY_STATE_THRESHOLDS["刀盘转速"]
    mask_penetration = chunk["贯入度"] > STEADY_STATE_THRESHOLDS["贯入度"]
    return mask_thrust & mask_torque & mask_speed & mask_rpm & mask_penetration


def _filter_steady_blocks(chunk: pd.DataFrame, steady_mask: pd.Series) -> pd.DataFrame:
    """连续性时间窗口过滤：剔除持续时间 < 300秒的碎片化工况"""
    # 利用 cumsum 计算连续运行的 block ID
    block_ids = (~steady_mask).cumsum()
    # 仅保留稳态点，并按 block_id 分组过滤长度
    steady_chunks = chunk[steady_mask].copy()
    steady_chunks["_block_id"] = block_ids[steady_mask.values]

    valid_blocks = (
        steady_chunks.groupby("_block_id")
        .filter(lambda x: len(x) >= MIN_STEADY_DURATION)
    )
    return valid_blocks.drop(columns=["_block_id"])


def _rolling_mad_denoise(df: pd.DataFrame, col: str) -> pd.Series:
    """Rolling MAD 单变量动态基线去噪（毛刺初筛）

    针对极易产生高频突跳的力学通道执行局部鲁棒过滤：
    1. 计算局部中位数（动态基线）
    2. 计算绝对残差的局部中位数
    3. 识别超出阈值的点并用前向填充平滑
    """
    series = df[col].copy()

    # 局部中位数（动态基线）
    rolling_median = series.rolling(window=MAD_WINDOW, min_periods=1, center=True).median()

    # 绝对残差的局部中位数
    abs_dev = (series - rolling_median).abs()
    rolling_mad = abs_dev.rolling(window=MAD_WINDOW, min_periods=1, center=True).median()

    # 鲁棒标准差
    robust_sigma = MAD_K_FACTOR * rolling_mad

    # 识别电气毛刺点
    glitch_mask = abs_dev > (MAD_TAU * robust_sigma)

    # 利用前向填充平滑单点异常，严禁整行删除
    series.loc[glitch_mask] = np.nan
    return series.ffill().bfill()


def _lof_multivariate_filter(df: pd.DataFrame) -> pd.DataFrame:
    """多物理场 LOF 空间密度约束

    提取跨域强耦合特征构建多维特征矩阵，识别违背传导逻辑的多维空间孤立毛刺点。
    label == 1 为正常物理耦合点（包含突发硬岩的高载荷耦合点）
    label == -1 为违背传导逻辑的多维空间孤立毛刺点

    特征空间：[总推进力, 刀盘扭矩, 推进速度, 贯入度]
    """
    lof_features = df[["总推进力", "刀盘扭矩", "推进速度", "贯入度"]].values

    lof_detector = LocalOutlierFactor(
        n_neighbors=LOF_N_NEIGHBORS,
        contamination=LOF_CONTAMINATION,
        n_jobs=-1
    )
    outlier_labels = lof_detector.fit_predict(lof_features)

    return df[outlier_labels == 1].copy()


def process_chunk(chunk: pd.DataFrame) -> Optional[pd.DataFrame]:
    """单块数据处理流水线

    Args:
        chunk: 原始数据块

    Returns:
        清洗后的稳态数据块，若无有效稳态数据则返回 None
    """
    # Step 1: 五元组稳态物理剥离
    steady_mask = _extract_steady_state_mask(chunk)
    valid_blocks = _filter_steady_blocks(chunk, steady_mask)

    if valid_blocks.empty:
        return None

    # Step 2: Rolling MAD 单变量动态基线去噪（毛刺初筛）
    for col in ["总推进力", "刀盘扭矩", "推进速度", "贯入度"]:
        valid_blocks[col] = _rolling_mad_denoise(valid_blocks, col)

    # Step 3: 多物理场 LOF 空间密度约束
    final_clean = _lof_multivariate_filter(valid_blocks)

    return final_clean


def load_and_process_streaming(
    raw_csv_path: str,
    chunk_size: int = 50000,
) -> pd.DataFrame:
    """流式分块读取与处理主函数

    核心原则：读取全量 1390+ 列，按行截取稳态片段，不丢弃任何传感器通道

    Args:
        raw_csv_path: 原始 CSV 文件路径（每日 500MB 超宽时序数据）
        chunk_size: 分块大小，默认 50000（调小以防止全量列 OOM）

    Returns:
        合并后的纯净稳态 DataFrame（保留全量列）
    """
    clean_chunks = []

    # 读取全量列：usecols 仅用于指定核心决策通道作为 dtype 过滤，
    # 实际处理时对全量 DataFrame 执行行掩码截取
    for chunk in pd.read_csv(raw_csv_path, chunksize=chunk_size, encoding="gbk", low_memory=False):
        processed = process_chunk(chunk)
        if processed is not None:
            clean_chunks.append(processed)

    if clean_chunks:
        return pd.concat(clean_chunks, axis=0, ignore_index=True)
    return pd.DataFrame()


# =====================================================================
# 向后兼容接口（保留旧代码结构避免大规模重构）
# =====================================================================

# 物理约束边界（旧接口保留，但实际使用五元组判据）
PARAMETER_BOUNDS = {
    "总推力_kN": (0, 50000),
    "刀盘扭矩_kNm": (0, 10000),
    "推进速度_mm_min": (0, 200),
    "刀盘转速_rpm": (0, 20),
    "贯入度_mm_rev": (0, 100),
}


def remove_outliers(series: pd.Series, method: str = "iqr", threshold: float = 3.0) -> pd.Series:
    """基于IQR或Z-score去除异常值（向后兼容接口）"""
    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return series.clip(lower=lower, upper=upper)
    elif method == "zscore":
        z = np.abs((series - series.mean()) / series.std())
        return series.where(z < threshold, np.nan)
    return series


def moving_average_filter(series: pd.Series, window: int = 5) -> pd.Series:
    """简单滑动平均滤波（向后兼容接口）"""
    return series.rolling(window=window, center=True, min_periods=1).mean()


def resample_to_ring(df: pd.DataFrame, ring_col: str = "环号") -> pd.DataFrame:
    """将高频秒级数据按环号重采样（向后兼容接口）"""
    agg_funcs = ["mean", "std", "min", "max"]
    param_cols = [col for col in df.columns if col != ring_col]
    df_agg = df.groupby(ring_col)[param_cols].agg(agg_funcs)
    df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]
    return df_agg.reset_index()


def preprocess_parameters(
    df: pd.DataFrame,
    apply_filter: bool = True,
    filter_window: int = 5,
    remove_outliers_flag: bool = True,
    resample: bool = True,
) -> pd.DataFrame:
    """高频参数预处理主流程（向后兼容接口）

    注意：新代码建议使用 load_and_process_streaming() 替代此接口，
    以获得五元组稳态剥离 + 双重去噪的完整处理流程。
    """
    df = df.copy()

    for col, (lower, upper) in PARAMETER_BOUNDS.items():
        if col in df.columns:
            if remove_outliers_flag:
                df[col] = remove_outliers(df[col])
            df[col] = df[col].clip(lower=lower, upper=upper)

    if apply_filter:
        for col in PARAMETER_BOUNDS:
            if col in df.columns:
                df[col] = moving_average_filter(df[col], window=filter_window)

    if resample:
        return resample_to_ring(df)

    return df


# =====================================================================
# 主程序入口：五日数据流式清洗
# =====================================================================

def process_five_days(
    data_dir: str = "data/raw",
    output_dir: str = "data/processed",
    chunk_size: int = 100000,
) -> dict[str, pd.DataFrame]:
    """流式处理五日原始数据并保存清洗结果

    Args:
        data_dir: 原始 CSV 所在目录
        output_dir: 输出目录（清洗后 CSV）
        chunk_size: 分块大小

    Returns:
        各日清洗结果的统计字典
    """
    import os
    from pathlib import Path

    os.makedirs(output_dir, exist_ok=True)
    data_path = Path(data_dir)

    # 按文件名排序确保顺序一致
    day_files = sorted(data_path.glob("*.csv"))
    results = {}

    for day_file in day_files:
        print(f"\n{'='*60}")
        print(f"处理日期: {day_file.stem}")
        print(f"{'='*60}")

        clean_df = load_and_process_streaming(
            str(day_file),
            chunk_size=chunk_size,
        )

        # 保存清洗后数据
        out_path = Path(output_dir) / f"steady_{day_file.stem}.csv"
        clean_df.to_csv(out_path, index=False)

        results[day_file.stem] = {
            "原始块数": sum(1 for _ in pd.read_csv(day_file, usecols=["日期"], chunksize=chunk_size, encoding="gbk")),
            "清洗后行数": len(clean_df),
            "输出文件": str(out_path),
        }

        print(f"  原始块数: {results[day_file.stem]['原始块数']}")
        print(f"  清洗后行数: {results[day_file.stem]['清洗后行数']}")
        print(f"  保存至: {out_path}")

    # 汇总统计
    print(f"\n{'='*60}")
    print("五日清洗汇总")
    print(f"{'='*60}")
    total_clean = sum(v["清洗后行数"] for v in results.values())
    for day, stats in results.items():
        print(f"  {day}: {stats['清洗后行数']:,} 行")
    print(f"  合计: {total_clean:,} 行")

    return results


def diagnose_day(raw_csv_path: str, chunk_size: int = 100000) -> pd.DataFrame:
    """诊断某日数据的五元组各参数分布"""
    stats = {col: [] for col in DECISION_COLS + ["泥水仓顶部1压力"]}
    stats["行数"] = []

    for chunk in pd.read_csv(raw_csv_path, chunksize=chunk_size, encoding="gbk", low_memory=False):
        for col in DECISION_COLS:
            if col in chunk.columns:
                stats[col].append(chunk[col])
        if "泥水仓顶部1压力" in chunk.columns:
            stats["泥水仓顶部1压力"].append(chunk["泥水仓顶部1压力"])
        stats["行数"].append(len(chunk))

    result = {}
    for col in list(DECISION_COLS) + ["泥水仓顶部1压力"]:
        if stats[col]:
            series = pd.concat(stats[col], ignore_index=True)
            result[col] = {
                "count": series.count(),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "满足条件数": (series > STEADY_STATE_THRESHOLDS[col]).sum() if col in STEADY_STATE_THRESHOLDS else 0,
            }

    total_rows = sum(stats["行数"])
    print(f"\n{'='*60}")
    print(f"诊断结果: {raw_csv_path}")
    print(f"总行数: {total_rows:,}")
    print(f"{'='*60}")
    for col, s in result.items():
        print(f"\n{col}:")
        print(f"  有效值: {s['count']:,} / {total_rows:,}")
        print(f"  范围: [{s['min']:.2f}, {s['max']:.2f}], 均值={s['mean']:.2f}, 标准差={s['std']:.2f}")
        if col in STEADY_STATE_THRESHOLDS:
            print(f"  满足阈值({STEADY_STATE_THRESHOLDS[col]}): {s['满足条件数']:,}")
        elif col == "泥水仓顶部1压力":
            in_range = ((series >= STEADY_STATE_THRESHOLDS["泥水仓顶部1压力_min"]) &
                       (series <= STEADY_STATE_THRESHOLDS["泥水仓顶部1压力_max"])).sum()
            print(f"  满足[1.5, 3.5]区间: {in_range:,}")

    return pd.DataFrame(result).T


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnose":
        day_file = sys.argv[2] if len(sys.argv) > 2 else "TBM_Cutter_Wear_Project/data/raw/260207.csv"
        diagnose_day(day_file)
    else:
        process_five_days(
            data_dir="TBM_Cutter_Wear_Project/data/raw",
            output_dir="TBM_Cutter_Wear_Project/data/processed",
            chunk_size=100000,
        )
