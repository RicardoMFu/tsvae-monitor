# tsvae-monitor

**Unsupervised TBM Cutter Wear Monitoring System**

基于盾构机传感数据的无监督刀具退化感知系统，通过多物理场时序变分自编码器（TS-VAE）实现无需标签的刀具寿命预测。

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)

---

## 项目简介

盾构施工中，每次刀具标定必须停机开仓，成本高昂且存在安全风险，导致标定数据极度稀疏（全生命周期数据密度仅 4.7%）。本项目提出一种**完全无监督**的时序变分自编码器范式，仅需健康基线数据训练，即可通过传感器全量数据实时量化刀具退化状态，填补标定盲区。

> **核心问题**：在 95.3% 的掘进时间里，机电液系统对刀具状态处于"盲测"状态。
>
> **解决思路**：固化"正常多物理场耦合传导流形"——刀具磨损导致信号偏离健康流形，解码器无法还原异变信号 → 重构误差 MSE 放大，作为额外机械耗散做功的精准度量。

---

## 系统架构

```
原始传感器数据 (1390+ 列/日, 500MB/日)
        │
        ▼
┌─────────────────────────────────────────┐
│  模块一：稳态物理信号剥离                 │
│  五元组判据 + 双重去噪 (MAD + LOF)       │
└─────────────────────────────────────────┘  steady_*.csv
        │  10个物理字段提取
        ▼
┌─────────────────────────────────────────┐
│  模块二：多物理场特征衍生                 │
│  SE · FPI · TPI · power_proxy           │
└─────────────────────────────────────────┘  features_steady_*.csv
        │  滑动窗口 (T=60s, S=30s)
        ▼
┌─────────────────────────────────────────┐
│  模块三：TS-VAE 无监督表征学习  ← 核心引擎 │
│  ELBO = L_recon + β·L_KL               │
│  85,162 参数 | d=16 | h=64 | GRU×2      │
└─────────────────────────────────────────┘  tsvae_model.pt
        │
        ▼  HI(R) 健康指标
┌─────────────────────────────────────────┐
│  模块四：稀疏性热力图可视化               │
│  刀具标定数据 colour_idx 语义映射        │
└─────────────────────────────────────────┘  论文插图
```

---

## 核心模块

### 模块一：稳态物理信号剥离

**五元组稳态判据**（五值全部 $>0$ 表示正常推进，任一为 $0$ 即停机）：

| 参数 | 阈值 | 含义 |
|------|------|------|
| 总推进力 | $>0$ kN | 正常切削 |
| 刀盘扭矩 | $>0$ kN·m | 正常切削 |
| 推进速度 | $>0$ mm/min | 正常切削 |
| 刀盘转速 | $>0$ rpm | 正常切削 |
| 贯入度 | $>0$ mm/rev | 正常切削 |

持续时间 $<300$s 的碎片化工况直接剔除。

**双重去噪**：Rolling MAD（单变量毛刺初筛） + LOF（多变量孤立瞬态冲击捕获，contamination=0.005）。

### 模块二：多物理场特征衍生

四大物理簇分组建模：

| 物理场 | 代表通道 |
|--------|----------|
| 机械 (MECHANIC) | F\_v, T\_current, v\_fwd, RPM, 贯入度 |
| 电气 (ELECTRIC) | P2.1泵电流, P0.1~0.3泵电流 |
| 热力 (THERMAL) | 主油箱油温, 齿轮油温, 电机温度 |
| 流体 (FLUID) | 泥水仓压力, 排浆/进浆密度 |

三个核心物理不变量：

$$SE = \frac{F_v \cdot v_{fwd}}{\pi \cdot (D_{cutter}/2)^2}, \quad FPI = \frac{v_{fwd}}{RPM}, \quad power\_proxy = T_{current} \cdot RPM$$

### 模块三：TS-VAE 无监督表征学习

**网络架构**（Multivariate TS-VAE，输入 $X \in \mathbb{R}^{B \times 60 \times 10}$）：

```
EncoderGRU:
  Layer1: nn.GRU(10→64, batch_first=True)
  Layer2: nn.GRU(64→64)
  h_T → μ = W_μ·h_T + b_μ  → (B, 16)
       → logσ² = W_σ·h_T + b_σ → (B, 16)

z = μ + σ ⊙ ε,  ε ~ N(0,I)       ← Reparameterization Trick

DecoderGRU:
  h₀ = tanh(W_init·z)            ← (2, B, 64)
  for t in 1..60:
      h_t = GRU(h_{t-1}, z)     ← 自回归展开
  X̂ = W_out·h_60                 ← (B, 60, 10)
```

**损失函数**：

$$L_{recon} = \frac{1}{B}\sum\|X - \hat{X}\|^2, \quad L_{KL} = -\frac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - e^{\log\sigma^2})$$
$$ELBO = L_{recon} + \beta \cdot L_{KL}, \quad \beta = 0.001$$

**健康基线**：仅使用 Ring 121~123（2026-02-07 ~ 2026-02-08）数据训练。

**HI(R) 推理流程**：

$$\text{全生命周期5日数据} \xrightarrow{T=60,s=30} 7,448 \text{窗口} \xrightarrow{TS-VAE} MSE_w \xrightarrow{\text{按环聚合}} HI(R) = \text{mean}(MSE_{ring})$$

**HI(R) 物理意义**：刀具磨损 → 物理信号偏离健康流形 → 解码器无法还原异变信号 → 重构误差 MSE 放大 = **额外机械耗散做功的精准度量**。

### 模块四：刀具标定稀疏性可视化

通过 `xlrd` 直接读取 Excel 单元格字体 `colour_idx`，映射真实颜色语义：

| colour\_idx | RGB | 含义 |
|-------------|-----|------|
| 10 | RGB(255,0,0) | 标红异常（偏磨≥10mm / 刀圈崩断） |
| 8 | RGB(0,0,0) | 特殊/正常（≤5mm） |
| 32767 | Excel系统缺省 | 普通磨损（中灰色） |
| 57 | RGB(51,153,102) | 探头故障/密封破损等特殊 |

> **32767 关键词升级**：colour\_idx=32767 但文本含 `'崩断'` 或 `'偏磨'` → 升级为标红异常。

统计数据（77 条记录）：标红 27 条（35%）| 普通灰 48 条（62%）| 特殊黑 2 条 | 特殊绿 2 条 | 数据密度 4.7%。

---

## 目录结构

```
TBM_Cutter_Wear_Project/
├── src/
│   ├── preprocessor.py          # 模块一：五元组稳态判据 + 双重去噪
│   ├── feature_engineer.py     # 模块二：四大物理簇 + 特征衍生
│   ├── unsupervised_engine.py  # 模块三：TS-VAE 核心引擎
│   ├── visualization_paper.py  # 论文级可视化
│   ├── alignment.py            # 高频/低频数据对齐
│   ├── parser.py               # 换刀记录文本解析
│   └── tsvae_network_viz.py   # TS-VAE 架构可视化
├── reports/                    # 技术报告
└── results/module3/           # 模型权重 + 训练曲线 + 4张论文图

viz/
├── plot_cutter_sparsity.py     # 刀具标定稀疏性热力图
├── paper_filling_material.md   # 论文填充材料（可直接填入 LaTeX）
├── generate_tsvae_architecture.py  # TS-VAE 架构 SVG 生成
└── svg/                         # 输出的热力图 PNG
```

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/RicardoMFu/tsvae-monitor.git
cd tsvae-monitor

# 安装依赖
pip install -r TBM_Cutter_Wear_Project/requirements.txt

# 运行模块三完整流程
python TBM_Cutter_Wear_Project/src/unsupervised_engine.py

# 生成刀具标定稀疏性热力图
python viz/plot_cutter_sparsity.py

# 生成 TS-VAE 架构图
python viz/generate_tsvae_architecture.py
```

---

## 输出物料

| 文件 | 说明 |
|------|------|
| `tsvae_model.pt` | 训练好的 TS-VAE 模型权重 |
| `training_history.csv` | 训练/验证损失曲线 |
| `ring_health_indicator.csv` | 环号 → HI(R) 映射表 |
| `fig8~fig11` (PNG) | 4 张论文级图表 |

---

## 引用

如果你在研究中使用了本项目，请引用：

```
@misc{tsvae-monitor,
  author = {RicardoMFu},
  title = {tsvae-monitor: Unsupervised TBM Cutter Wear Monitoring System},
  year = {2026},
  url = {https://github.com/RicardoMFu/tsvae-monitor}
}
```