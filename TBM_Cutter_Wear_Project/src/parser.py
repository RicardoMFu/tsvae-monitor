"""解析半结构化换刀记录文本，提取结构化标签数据。

典型输入格式示例：
    "13正常磨损3毫米，15刀圈崩断"
    "第7环：14正常磨损2mm，16偏磨，18刀圈崩断"

输出：DataFrame [环号, 日期, 刀号, 磨损类型, 磨损量(mm)]
"""

import re
from typing import Optional

import pandas as pd


# 磨损类型映射
WEAR_TYPE_MAP = {
    "正常磨损": "normal",
    "偏磨": "偏磨",  # 保留原始标签
    "刀圈崩断": "broken",
    "崩断": "broken",
}


def parse_ring_label(text: str) -> list[dict]:
    """解析单条换刀记录文本。

    Args:
        text: 如 "13正常磨损3毫米，15刀圈崩断"

    Returns:
        如 [{"刀号": 13, "磨损类型": "normal", "磨损量": 3.0}, ...]
    """
    results = []
    # 匹配模式：刀号(数字) + 状态描述 + 可选磨损量(mm/mm毫米)
    pattern = r"(\d+)(正常磨损|偏磨|刀圈崩断|崩断)(\d+)毫米?"

    for match in re.finditer(pattern, text):
        cutter_id = int(match.group(1))
        wear_type_raw = match.group(2)
        wear_amount = float(match.group(3))

        wear_type = WEAR_TYPE_MAP.get(wear_type_raw, wear_type_raw)
        results.append({
            "刀号": cutter_id,
            "磨损类型": wear_type,
            "磨损量(mm)": wear_amount,
        })

    # 处理没有磨损量的情况（如 "刀圈崩断" 无具体数值）
    pattern_no_amount = r"(\d+)(正常磨损|偏磨|刀圈崩断|崩断)(?!\d)"
    for match in re.finditer(pattern_no_amount, text):
        # 检查是否已被上面的模式捕获
        cutter_id = int(match.group(1))
        if not any(r["刀号"] == cutter_id for r in results):
            wear_type_raw = match.group(2)
            wear_type = WEAR_TYPE_MAP.get(wear_type_raw, wear_type_raw)
            results.append({
                "刀号": cutter_id,
                "磨损类型": wear_type,
                "磨损量(mm)": None,
            })

    return results


def parse_wear_records(filepath: str) -> pd.DataFrame:
    """批量解析换刀记录CSV。

    Args:
        filepath: 换刀记录CSV路径，需包含 [环号, 日期, 记录文本] 列

    Returns:
        结构化DataFrame [环号, 日期, 刀号, 磨损类型, 磨损量(mm)]
    """
    df_raw = pd.read_csv(filepath)
    rows = []

    for _, row in df_raw.iterrows():
        ring = row["环号"]
        date = row["日期"]
        text = row["记录文本"]

        parsed = parse_ring_label(text)
        for item in parsed:
            rows.append({
                "环号": ring,
                "日期": date,
                "刀号": item["刀号"],
                "磨损类型": item["磨损类型"],
                "磨损量(mm)": item["磨损量(mm)"],
            })

    return pd.DataFrame(rows)
