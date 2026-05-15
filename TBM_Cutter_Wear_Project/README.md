# 盾构机刀具磨损状态评估

基于多维施工参数的数据驱动框架。

## 项目结构

```
TBM_Cutter_Wear_Project/
├── data/
│   ├── raw/                 # 原始数据
│   └── processed/           # 预处理后的结构化数据
├── src/
│   ├── __init__.py
│   ├── parser.py            # 解析半结构化换刀记录文本
│   ├── preprocessor.py      # 高频参数去噪、滤波、重采样
│   ├── feature_engineer.py  # 构造复合特征（比能、TPI等）
│   └── alignment.py         # 高频参数与低频换刀记录对齐
├── notebooks/               # 探索性数据分析
└── main.py                  # 主流程入口
```

## 使用方法

### 数据准备

将原始数据放入 `data/raw/`：
- `换刀记录.csv` — 半结构化文本（刀号、磨损类型、磨损量）
- `高频参数.csv` — 秒级施工参数（推力、扭矩、速度、转速、贯入度）

### 运行完整流程

```bash
python main.py --wear data/raw/换刀记录.csv --params data/raw/高频参数.csv
```

### 分步运行

```python
from src.parser import parse_wear_records
from src.preprocessor import preprocess_parameters
from src.feature_engineer import add_composite_features
from src.alignment import build_labeled_dataset

# 1. 解析标签
wear_df = parse_wear_records("data/raw/换刀记录.csv")

# 2. 预处理参数
param_df = preprocess_parameters(pd.read_csv("data/raw/高频参数.csv"))

# 3. 构造特征
param_df = add_composite_features(param_df)

# 4. 对齐构建数据集
labeled_df = build_labeled_dataset(param_df, wear_df)
```

## 核心模块说明

### parser.py
解析形如 `"13正常磨损3毫米，15刀圈崩断"` 的文本记录，输出结构化DataFrame。

### preprocessor.py
- `remove_outliers()` — 基于IQR去异常值
- `moving_average_filter()` — 滑动平均滤波
- `preprocess_parameters()` — 完整预处理流程

### feature_engineer.py
- `compute_specific_energy()` — 比能 (SE)
- `compute_tpi()` — 扭矩-贯入度指数 (TPI)
- `compute_energy_per_ring()` — 单环积分能量

### alignment.py
处理高频/低频数据的时间分辨率不匹配问题，以换刀周期为单元对齐特征与标签。

## 数据空白期说明

2月份换刀记录存在空白期（2.12后跳到2.22），中间数据可作为无标签预测测试集。
