# TSVAE-Monitor: Unsupervised TBM Cutter Wear Detection

**A multivariate time-series variational autoencoder (TS-VAE) framework for label-free cutter wear assessment in tunnel boring machines.**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 一、项目概述

本项目实现了一个**物理引导的无监督学习pipeline**，通过多物理场传感器信号检测盾构机（TBM）刀具磨损演化。不同于传统监督方法需要大量标注磨损数据，我们的方法：

- 利用**7维全局独立特征张量**（7D输入）捕获跨物理场协同传导规律
- 通过**时序变分自编码器（TS-VAE）**学习健康模态流形
- 在无标签条件下生成环级**健康指标 HI(R)**

### 核心成果：10D → 7D 多重共线性消除

原始10维特征方案存在严重共线性（SE/FPI/TPI由底层机械参数派生）。我们通过**物理导向特征平替原则**解决：

| 10D问题 | 7D解决方案 |
|--------|-----------|
| SE/FPI/TPI由Fv,T,v派生 → 共线性 | 保留3个高阶不变量（SE, FPI, TPI） |
| 底层机械参数与派生量冗余 | 强制剔除所有底层机械参数 |
| 无控制跨簇相关 | 全局Spearman相关性贪心选取 |

**输入压缩比**: 7/16 ≈ 0.44（确保TS-VAE学习隐式协同传导，而非线性同构）

---

## 二、架构总览

```
原始传感器数据 (1390+ 列)
         │
         ▼
┌─────────────────────────────────────────────┐
│  模块一：五元组稳态检测与双重去噪             │
│  Fv>0 ∧ T>0 ∧ v>0 ∧ n>0 ∧ p>0             │
│  Rolling MAD + LOF 双重降噪                 │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  模块二：7维特征工程                         │
│  ─────────────────────────────────────────  │
│  高阶不变量：SE, FPI, TPI                   │
│  跨域代理：P2.1泵电流、主油箱油温、           │
│           泥水仓压、排浆密度                 │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  模块三：TS-VAE 无监督学习                   │
│  ─────────────────────────────────────────  │
│  输入:  X ∈ ℝ^(B×60×7)                     │
│  Encoder: GRU(7→64→64)                     │
│  Latent:  Z ∈ ℝ^(B×16)  (压缩比 0.44)      │
│  Decoder: GRU(16→64→64)                    │
│  输出: X̂ ∈ ℝ^(B×60×7)                     │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  模块四：HI(R) 环级健康指标                  │
│  UMAP流形轨迹 + 退化曲线 + MSE热力图        │
└─────────────────────────────────────────────┘
```

### 7D特征规范

| # | 特征 | 物理场 | 说明 |
|---|------|--------|------|
| 1 | SE | 能效/机械 | 切削比能（能量/体积） |
| 2 | FPI | 能效/机械 | 推力贯入度指数 |
| 3 | TPI | 能效/机械 | 扭矩贯入度指数 |
| 4 | P2.1泵电流 | 电气 | 主驱动泵负载代理 |
| 5 | 主油箱油温 | 热力 | 热耗散累积滞后 |
| 6 | 泥水仓顶部1压力 | 流体 | 水土压力边界 |
| 7 | 排浆密度 | 流体 | 携岩能力 |

### Spearman相关性（设计约束）

```
         SE    FPI    TPI  I_p21  T_oil  P_top  rho_out
SE      1.000  0.765  0.941  0.029 -0.120  0.006  0.028
FPI     0.765  1.000  0.774 -0.191 -0.160 -0.010 -0.211
TPI     0.941  0.774  1.000  0.010  0.030 -0.015  0.011
I_p21   0.029 -0.191  0.010  1.000  0.085 -0.053  0.504
```

> SE↔TPI(r=0.941), SE↔FPI(r=0.765), FPI↔TPI(r=0.774) 为设计层面物理约束妥协。

---

## 三、环级HI(R)结果

| 环号 | HI(R) | 状态 |
|------|-------|------|
| 121 | 227.10 | 健康基线 |
| 122 | 309.86 | 健康基线 |
| 123 | 112.27 | 健康基线 |
| 124 | 119.04 | 正常掘进 |
| 125 | 126.79 | 正常掘进 |
| 126 | 189.81 | 略有上升 |
| 127 | 210.16 | 异常信号 |
| 128 | 121.30 | 恢复 |
| 129 | 176.01 | 波动 |
| **130** | **576.62** | **极端退化峰值** |

**关键发现**: Ring 130的HI值达576.62，约为健康基线的2.7倍，表明7D特征对极端退化状态敏感。

---

## 四、目录结构

```
TBM_Cutter_Wear_Project/
├── data/
│   ├── raw/                    # 原始CSV (1390+列)
│   └── processed/             # 清洗后稳态数据 + 特征文件
├── src/
│   ├── preprocessor.py         # 模块一：五元组检测 + 去噪
│   ├── feature_engineer.py    # 模块二：7D特征工程
│   ├── unsupervised_engine.py # 模块三：TS-VAE训练与推理
│   └── *.py                   # 可视化与分析工具
├── viz/
│   └── plot_pipeline_workflow.py  # 流程图生成脚本
├── results/
│   ├── fig1_pipeline_workflow.png  # 最终版工作流图 (300dpi)
│   ├── module3/               # TS-VAE模型与训练结果
│   │   ├── tsvae_model.pt
│   │   ├── training_history.csv
│   │   ├── ring_health_indicator.csv
│   │   └── figures/          # Fig 8-11 论文图
│   └── feature_selection_analysis/  # 7D特征相关性分析图
├── docs/                       # 原理说明文档
├── reports/                    # 技术报告
└── README.md / CLAUDE.md
```

---

## 五、快速开始

### 环境依赖

```bash
pip install -r requirements.txt
# numpy>=1.24, pandas>=2.0, scikit-learn>=1.3, torch>=2.0, umap-learn>=0.5
```

### 运行完整流程

```bash
# 模块一：数据预处理
python src/preprocessor.py

# 模块二：特征工程
python src/feature_engineer.py

# 模块三：TS-VAE训练与推理
python src/unsupervised_engine.py

# 生成工作流图
python viz/plot_pipeline_workflow.py
```

---

## 六、核心配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入维度 | 7 | 7D特征张量 |
| 潜在维度 | 16 | d=16 |
| GRU隐藏单元 | 64 | hidden=64 |
| GRU层数 | 2 | num_layers=2 |
| 输入压缩比 | 7/16 ≈ 0.44 | 防线性同构 |
| KL权重β | 0.001 | 防后验崩塌 |
| 健康基线 | Ring 121-123 | 前两日数据 |
| 滑动窗口 | T=60s, S=30s | 步长30减少帧数 |
| 最小稳态 | 300s | 五元组持续时间 |

---

## 七、引用

```bibtex
@misc{tsvae-monitor,
  title = {TSVAE-Monitor: Unsupervised TBM Cutter Wear Detection},
  author = {TBM Cutter Wear Project Team},
  year = {2026},
}
```