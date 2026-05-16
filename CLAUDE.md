# CLAUDE.md — 盾构刀具健康监测项目

## 项目概述
基于盾构机传感数据的无监督刀具退化感知系统，通过多物理场时序变分自编码器实现无需标签的刀具寿命预测。

---

## 管线四大核心模块

### 模块零：相关性分析与五元组参数筛选

**核心问题**：1390列传感器参数中，为何选取这五个参数作为稳态判据？

**筛选方法**：
1. 从1390列中筛选"实时测量型"参数（排除PLC设定值、累计量、状态标志等）
2. 定义**机械功率代理变量** `proxy_power = 刀盘扭矩 × 刀盘转速`（无磨损标签时的替代指标）
3. 计算候选参数与 proxy_power 的 Spearman 相关性
4. 筛选出与掘进状态强相关（|r|>0.5）且内部独立性较强（内部|r|在0.4~0.7之间）的参数组合

**相关性分析结果**（对50,000行采样数据计算）：
- 总推进力 → proxy_power：r=+0.761（与推进系统负载强正相关）
- 刀盘扭矩 → proxy_power：r=+0.997（功率代理变量本身，高度一致）
- 推进速度 → proxy_power：r=+0.874（掘进效率正相关）
- 刀盘转速 → proxy_power：r=+0.899（切削频率正相关）
- 贯入度 → proxy_power：r=+0.868（单转切削厚度正相关）

**五元组内部独立性**（|r|均在0.7~1.0之间，存在共线性但各有物理独立意义）：
|  | 总推进力 | 刀盘扭矩 | 推进速度 | 刀盘转速 | 贯入度 |
|--|---------|---------|---------|---------|--------|
| 总推进力 | 1.000 | 0.757 | 0.714 | 0.778 | 0.708 |
| 刀盘扭矩 | 0.757 | 1.000 | 0.864 | 0.887 | 0.859 |
| 推进速度 | 0.714 | 0.864 | 1.000 | 0.890 | **1.000** |
| 刀盘转速 | 0.778 | 0.887 | 0.890 | 1.000 | 0.883 |
| 贯入度 | 0.708 | 0.859 | **1.000** | 0.883 | 1.000 |

> 注：推进速度与贯入度完全正相关（r=1.000），因为贯入度 = 推进速度 / 刀盘转速，在转速恒定时二者等价。保留贯入度是因为它消除了转速归一化效应，是更稳定的物理量。

**脚本**：`src/correlation_analysis.py`（采样5万行，避免OOM）

### 模块一：数据预处理与参数清洗
- 原始传感器数据解析（推力F、扭矩T、推进速度v、刀盘转速RPM等）
- 异常值剔除与缺失值填补
- **五元组稳态判据**：总推力>0 AND 刀盘扭矩>0 AND 推进速度>0 AND 刀盘转速>0 AND 贯入度>0
  - 五值全部 $>0$ 表示正常推进，任一为 $0$ 即停机

### 模块二：特征工程与物理不变量衍生
- 切削比能 SE = F × v / πD²（单位体积切削能耗）
- 贯入度指数 FPI = 推进速度 / 刀盘转速，TPI = 推进速度 / 刀盘转速
- 多物理场通道融合：SE, FPI, TPI, F_v, T, I_p21, τ_oil, power_proxy, v_forward, RPM（10维核心特征）
- **数据源**：原始环号（RING ID）直接读取自施工日志CSV（第121环起始，每日环号由PLC自动记录）

### 模块三：无监督表征学习与退化指标构建（核心引擎）

#### 3.1 网络架构：Multivariate TS-VAE

```
输入 X ∈ ℝ^(B×60×10)
  │
  ▼
┌─ EncoderGRU ─────────────────────────────────────────────┐
│  Layer1: nn.GRU(10→64)  batch_first=True                  │
│  Layer2: nn.GRU(64→64)                                   │
│  取最后时刻隐状态 h_T: (B, 64)                            │
│  μ = W_μ·h_T + b_μ  → (B, 16)                            │
│  logσ² = W_σ·h_T + b_σ → (B, 16)                        │
└──────────────────────────────────────────────────────────┘
  │
  │  Reparameterization Trick: z = μ + σ⊙ε, ε~𝒩(0,I)
  ▼
潜在空间 Z ∈ ℝ^(B×16)
  │
  ▼
┌─ DecoderGRU ─────────────────────────────────────────────┐
│  h₀ = tanh(W_init·z + b_init) → (2, B, 64)                │
│  Layer1: nn.GRU(16→64)  batch_first=True                  │
│  Layer2: nn.GRU(64→64)                                   │
│  自回归展开T=60步，每步GRU输入=前一隐状态                    │
│  X̂ = W_out·h_T + b_out → (B, 60, 10)                    │
└──────────────────────────────────────────────────────────┘
  │
  ▼
重构输出 X̂ ∈ ℝ^(B×60×10)
```

**参数量**：85,162 | **潜在维度**：d=16 | **GRU隐藏单元**：64 | **层数**：2

#### 3.2 训练范式

- **健康基线定义**：仅使用前两日（2026-02-07 ~ 2026-02-08，Ring 121-123）掘进数据训练
- **损失函数**：ELBO = L_recon + β·L_KL
  - L_recon = MSE(X, X̂) = (1/B)·∑∥X - X̂∥²
  - L_KL = D_KL(q(z|X) ∥ N(0,I)) = -0.5·∑(1 + logσ² - μ² - exp(logσ²))
  - β = 0.001（KL散度权重）
- **优化器**：AdamW (lr=1e-3, weight_decay=1e-4)
- **调度器**：ReduceLROnPlateau（patience=2, factor=0.5）
- **早停**：Patience=8，验证ELBO连续8轮未改善则停止

#### 3.3 推理与HI(R)计算

```
全生命周期5日数据（223,491条秒级记录）
  │
  ▼ 滑动窗口切取 T=60s, S=30s
7,448个时序窗口
  │
  ▼ TS-VAE推理（完全无监督，无标签）
每帧MSE ∈ ℝ (标量)
  │
  ▼ 按环号分组聚合
Ring 121: HI(121) = mean(MSE_i)
Ring 122: HI(122) = mean(MSE_i)
...
Ring 130: HI(130) = mean(MSE_i)
```

**HI(R)物理意义**：
- 网络固化"正常多物理场耦合传导流形"
- 刀具磨损 → 物理信号（推力/电流/温度）偏离健康流形
- 解码器无法还原异变信号 → 重构误差MSE放大
- MSE放大本质 = 额外机械耗散做功的精准度量

#### 3.4 核心通道配置（C=10）

| 物理场 | 通道名 | 说明 |
|--------|--------|------|
| 机械 | F_v | 总推进力 |
| 机械 | T_current | 刀盘扭矩 |
| 机械 | v_forward | 推进速度 |
| 机械 | RPM | 刀盘转速 |
| 电气 | I_p21 | P2.1泵电流 |
| 电气 | power_proxy | 功率代理（扭矩×转速） |
| 热力 | T_oil | 主油箱油温 |
| 能效 | SE | 切削比能 |
| 能效 | FPI | 贯入度指数 |
| 能效 | TPI | 贯入度指数 |

### 模块四：退化表征与临床验证
- UMAP流形轨迹投影（健康原点 → 磨损边界漂移）
- 马氏距离交叉验证
- 关键异变转折点定位（Ring 127异常峰值）

---

## 技术约束

### Windows安全防线（最高优先级）
- **`num_workers=0`**：所有DataLoader强制单进程，彻底杜绝多进程死锁
- **入口保护**：`if __name__ == '__main__': main()`
- **主进程封装**：所有模型初始化、数据加载、训练循环严格封装在`main()`中

### 防OOM策略
- 滑动窗口Dataset零拷贝设计：从大型DataFrame按需切片，避免一次性全量加载
- `STEP_SIZE=30`（而非10）：减少推理窗口总数至7,448，提升推理效率

### 图表输出规范
- 全部4张图表：DPI=300，bbox_inches='tight'
- 中文字体支持：`plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', ...]`
- **图表清单**：
  1. `fig8_multichannel_reconstruction.png` — SE/I_p21/F_v三通道实测vs重构波形+残差填充
  2. `fig9_latent_space_trajectory.png` — UMAP 2D潜在流形轨迹（环号plasma色彩编码）
  3. `fig10_macro_health_indicator.png` — HI(R)全生命周期退化曲线+5点滑动平均
  4. `fig11_feature_loss_contribution.png` — 机械/电气/热力/能效四大物理场MSE贡献热力图

---

## 输出物料清单

| 文件 | 路径 | 说明 |
|------|------|------|
| `tsvae_model.pt` | `results/module3/` | 训练好的模型权重+配置 |
| `training_history.csv` | `results/module3/` | 训练/验证损失曲线 |
| `ring_health_indicator.csv` | `results/module3/` | 环号→HI(R)映射表 |
| `fig8~fig11 PNG` | `results/module3/figures/` | 4张论文级图表 |
| `module3_deep_integration_report.md` | `reports/` | 量化指标报告 |

---

## 目录结构

```
TBM_Cutter_Wear_Project/
├── data/
│   ├── raw/          # 原始500MB/日 CSV（1390+列）
│   └── processed/    # steady_*.csv（稳态切片）+ features_steady_*.csv
├── src/
│   ├── correlation_analysis.py   # 模块零：相关性分析与五元组筛选
│   ├── parser.py              # 解析半结构化换刀记录文本
│   ├── preprocessor.py        # 模块一：五元组稳态判据 + 双重去噪
│   ├── alignment.py          # 高频参数与低频换刀记录对齐
│   ├── feature_engineer.py   # 模块二：四大物理簇 + 特征衍生
│   ├── unsupervised_engine.py # 模块三核心引擎 v3.0 (TS-VAE)
│   ├── visualization_paper.py # 论文级可视化
│   ├── font_config.py         # 中文字体配置
│   └── tsvae_network_viz.py   # TS-VAE架构可视化
├── reports/                    # 技术报告
└── results/
    ├── module3/               # 训练结果 + 4张论文图
    └── correlation_analysis/ # 五元组相关性分析输出
        ├── candidate_spearman_corr.csv
        └── five_tuple_corr.csv

viz/                            # 刀具标定稀疏性可视化
├── plot_cutter_sparsity.py    # 热力图生成主脚本
├── paper_filling_material.md  # 论文填充材料（可直接填入LaTeX）
├── generate_tsvae_architecture.py  # TS-VAE架构SVG图生成器
└── svg/                        # 输出的热力图PNG
    ├── cutter_sparsity_heatmap.png
    ├── cutter_sparsity_heatmap_by_date.png
    └── cutter_sparsity_combined.png

models/                         # 发布级模型权重
└── tsvae_model.pt              # 与 results/module3/ 同步
```

---

## 刀具标定稀疏性模块（viz/）

### 颜色语义映射（xlrd colour_idx → 实际颜色）

使用 `xlrd.open_workbook(..., formatting_info=True)` 直接读取Excel单元格字体的 `colour_index`：

| colour_idx | RGB值 | 含义 |
|-----------|-------|------|
| 10 | RGB(255,0,0) | 用户手动标红异常（偏磨≥10mm / 刀圈崩断） |
| 8 | RGB(0,0,0) | 特殊/正常记录（≤5mm） |
| 32767 | Excel系统缺省色 | 普通磨损（无手动标记，≤5mm属正常范围） |
| 57 | RGB(51,153,102) | 探头故障/密封破损等特殊记录 |

**32767默认色关键词升级规则**：
```
if colour_idx == 32767 AND ('崩断' in text OR '偏磨' in text):
    → 升级为标红异常
else:
    → 维持中灰色（普通磨损）
```

### 热力图排版规范
- 灰色背景（0.88灰度）表示空白无记录区域
- 数据密度框：左上角，fontsize=14，白底圆角框
- 标红异常记录数：右上角，fontsize=14，红色文字，淡红底
- 顶部注释（如"TS-VAE健康基线区间"）：右上角标题旁
- 色条：已移除（颜色语义由 `mpatches.Patch` 图例表达，不再误导）
- `ax.text(0.99, 1.02, ... va='top', ha='right')` 用于右上角文本定位

### 统计数据（77条记录）
- 标红异常：27条（35%）
- 普通磨损：48条（62%）
- 特殊黑色：2条
- 特殊绿色：2条
- 数据密度：4.7%（95.3%空白）

---

## 常用开发命令

```bash
# 运行模块三完整流程
python TBM_Cutter_Wear_Project/src/unsupervised_engine.py

# 生成刀具标定稀疏性热力图
python viz/plot_cutter_sparsity.py

# 生成 TS-VAE 架构 SVG 图
python viz/generate_tsvae_architecture.py

# 查看训练历史
python -c "import pandas as pd; print(pd.read_csv('TBM_Cutter_Wear_Project/results/module3/training_history.csv').tail())"
```