"""核心对齐模块：将高频施工参数与低频换刀记录对齐。

核心矛盾：
    - 高频参数：秒级/分钟级连续时序
    - 换刀记录：按环/天结算，存在空白期（如2.04-2.12后跳到2.22）

对齐策略：
    1. 以"两次开仓换刀"为一个观测周期
    2. 将该周期内的所有环参数累积聚合
    3. 与周期末的磨损标签映射
"""

import pandas as pd
import numpy as np
from typing import Optional


def find_replacement_cycles(wear_df: pd.DataFrame) -> list[tuple]:
    """识别所有换刀周期（两次开仓之间）。

    Args:
        wear_df: 结构化换刀记录，需包含 [日期, 环号] 列

    Returns:
        换刀周期列表 [(start_date, end_date, start_ring, end_ring), ...]
    """
    # 按日期排序，找出所有换刀时间点
    wear_df = wear_df.sort_values("日期").reset_index(drop=True)
    dates = wear_df["日期"].unique()

    cycles = []
    for i in range(len(dates) - 1):
        start_date = dates[i]
        end_date = dates[i + 1]
        start_ring = wear_df[wear_df["日期"] == start_date]["环号"].min()
        end_ring = wear_df[wear_df["日期"] == end_date]["环号"].max()
        cycles.append((start_date, end_date, start_ring, end_ring))

    return cycles


def align_parameters_to_cycle(
    param_df: pd.DataFrame,
    cycle: tuple,
    wear_df: pd.DataFrame,
) -> pd.DataFrame:
    """将周期内的高频参数与该周期的磨损标签对齐。

    Args:
        param_df: 预处理后的高频参数（已按环聚合）
        cycle: 换刀周期元组 (start_date, end_date, start_ring, end_ring)
        wear_df: 结构化换刀记录

    Returns:
        对齐后的特征-标签数据集
    """
    start_date, end_date, start_ring, end_ring = cycle

    # 筛选周期内的环参数
    mask = (param_df["环号"] >= start_ring) & (param_df["环号"] <= end_ring)
    cycle_params = param_df[mask]

    # 聚合周期内所有环的特征（均值、极值等）
    param_cols = [col for col in cycle_params.columns if col != "环号"]
    aggregated = {}
    for col in param_cols:
        aggregated[f"{col}_cycle_mean"] = cycle_params[col].mean()
        aggregated[f"{col}_cycle_max"] = cycle_params[col].max()
        aggregated[f"{col}_cycle_std"] = cycle_params[col].std()

    # 获取该周期结束时的磨损标签
    wear_labels = wear_df[wear_df["日期"] == end_date]
    labels = {}
    for _, row in wear_labels.iterrows():
        cutter_id = row["刀号"]
        labels[f"刀{cutter_id}_磨损类型"] = row["磨损类型"]
        labels[f"刀{cutter_id}_磨损量"] = row["磨损量(mm)"]

    return {**aggregated, **labels}


def build_labeled_dataset(
    param_df: pd.DataFrame,
    wear_df: pd.DataFrame,
) -> pd.DataFrame:
    """构建完整的带标签数据集。

    Args:
        param_df: 预处理后的高频参数（已按环聚合）
        wear_df: 结构化换刀记录

    Returns:
        周期级特征-标签数据集
    """
    cycles = find_replacement_cycles(wear_df)
    aligned_rows = []

    for cycle in cycles:
        row = align_parameters_to_cycle(param_df, cycle, wear_df)
        row["换刀周期_start"] = cycle[0]
        row["换刀周期_end"] = cycle[1]
        row["起始环号"] = cycle[2]
        row["结束环号"] = cycle[3]
        aligned_rows.append(row)

    return pd.DataFrame(aligned_rows)


def handle_gap_periods(
    param_df: pd.DataFrame,
    wear_df: pd.DataFrame,
    gap_start: str,
    gap_end: str,
) -> pd.DataFrame:
    """处理换刀记录空白期（如2.16-2.19）的预测集构建。

    空白期无标签，可作为趋势预测的测试集。

    Args:
        param_df: 预处理后的高频参数
        wear_df: 结构化换刀记录
        gap_start: 空白期起始日期
        gap_end: 空白期结束日期

    Returns:
        空白期特征表（无标签）
    """
    gap_df = wear_df.copy()
    gap_df = gap_df[(gap_df["日期"] >= gap_start) & (gap_df["日期"] <= gap_end)]

    # 该区间的施工参数作为无标签预测输入
    # 具体实现依赖于实际数据中的日期-环号映射
    return gap_df
