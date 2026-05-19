"""
========================================
TSVAE-Monitor 顶刊级三阶段工作流全局架构图
========================================
基于绝对坐标系的 Matplotlib 绘制，无重叠，高级灰调配色
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

# =====================================================================
# 配色方案（顶刊级极简灰调）
# =====================================================================
COLORS = {
    # 数据层（蓝）
    'data_border': '#1B4F72',
    'data_fill': '#EAF2F8',
    # 特征层（紫）
    'feature_border': '#512E5F',
    'feature_fill': '#F5EEF8',
    # 算法层（橙）
    'algo_border': '#935116',
    'algo_fill': '#FEF5E7',
    # 箭头与文字
    'arrow': '#7F8C8D',
    'arrow_bold': '#2C3E50',
    'title': '#2C3E50',
    'subtitle': '#566573',
    'shadow': '#BDC3C7',
}

# =====================================================================
# 制图参数（绝对坐标系）
# =====================================================================
FIG_SIZE = (22, 16)
# X轴分配
COL1_X = 4.0    # 列1中心
COL2_X = 11.0   # 列2中心
COL3_X = 18.0   # 列3中心

# Y轴分配
Y_TITLE = 15.0        # 全局主标题
Y_COL_HEADER = 13.5   # 列标题
Y_ROW1 = 11.2         # 第一行框
Y_ROW2 = 8.6          # 第二行框
Y_ROW3 = 6.0          # 第三行框
Y_ROW4 = 3.4          # 第四行框
Y_BOTTOM_DIV = 1.8    # 底部分隔线
Y_BOTTOM_TOPO = 0.8   # 底部拓扑区

# 框体尺寸
BOX_W = 6.0
BOX_H = 1.6

# 阴影偏移
SHADOW_DX = 0.12
SHADOW_DY = -0.12

# =====================================================================
# 辅助函数
# =====================================================================

def draw_box_with_shadow(ax, cx, cy, w, h, label, sublabels, border_color, fill_color):
    """绘制带扁平阴影的圆角矩形框"""
    x0, y0 = cx - w/2, cy - h/2

    # 扁平阴影（先画，在主框下方）
    shadow = FancyBboxPatch((x0 + SHADOW_DX, y0 + SHADOW_DY), w, h,
                            boxstyle="round,pad=0.08",
                            facecolor=COLORS['shadow'],
                            edgecolor='none',
                            alpha=0.4,
                            zorder=2)
    ax.add_patch(shadow)

    # 主框体
    box = FancyBboxPatch((x0, y0), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=fill_color,
                          edgecolor=border_color,
                          linewidth=2.5,
                          alpha=1.0,
                          zorder=3)
    ax.add_patch(box)

    # 主标题
    ax.text(cx, cy + 0.35, label,
           ha='center', va='center', fontsize=13, fontweight='bold', color=border_color, zorder=4)

    # 副标题（多行）
    if sublabels:
        start_y = cy - 0.15
        line_spacing = 0.42
        for i, sub in enumerate(sublabels):
            ax.text(cx, start_y - i * line_spacing, sub,
                   ha='center', va='center', fontsize=10, color=COLORS['subtitle'], zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color=None, lw=2.5, rad=0.0):
    """绘制箭头连接线"""
    if color is None:
        color = COLORS['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                              connectionstyle=f'arc3,rad={rad}'),
                zorder=5)


def draw_horizontal_arrow(ax, x1, y1, x2, y2, label=None):
    """绘制水平箭头（阶段递进）"""
    draw_arrow(ax, x1, y1, x2, y2, color=COLORS['arrow_bold'], lw=3.0)
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 + 0.25
        ax.text(mid_x, mid_y, label, ha='center', va='bottom',
               fontsize=11, color=COLORS['arrow_bold'], fontstyle='italic', zorder=6)


def draw_box_label(ax, cx, cy, txt, size=14, color='#2C3E50', bold=True):
    """绘制纯文字标签（列标题用）"""
    weight = 'bold' if bold else 'normal'
    ax.text(cx, cy, txt, ha='center', va='center',
           fontsize=size, fontweight='bold' if bold else 'normal', color=color, zorder=4)


# =====================================================================
# 主绘图函数
# =====================================================================

def generate_figure(save_path):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 16)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # =====================================================================
    # 全局主标题
    # =====================================================================
    ax.text(11, Y_TITLE, '图1  智能盾构刀具磨损三阶段感知工作流',
           ha='center', va='center', fontsize=18, fontweight='bold', color='#1C2833')
    ax.text(11, Y_TITLE - 0.55,
           'TSVAE-Monitor: Unsupervised TBM Cutter Wear Detection via Multivariate TS-VAE',
           ha='center', va='center', fontsize=11, color='#566573', fontstyle='italic')

    # =====================================================================
    # 列标题（三大模块）
    # =====================================================================

    # 模块一标题
    header_h = 0.8
    box1 = FancyBboxPatch((COL1_X - 2.8, Y_COL_HEADER - header_h/2), 5.6, header_h,
                           boxstyle="round,pad=0.05",
                           facecolor='#EAF2F8', edgecolor=COLORS['data_border'],
                           linewidth=2.5, zorder=3)
    ax.add_patch(box1)
    ax.text(COL1_X, Y_COL_HEADER + 0.1, '模块一 · 数据层', ha='center', va='center',
           fontsize=14, fontweight='bold', color=COLORS['data_border'], zorder=4)
    ax.text(COL1_X, Y_COL_HEADER - 0.32, 'Data Layer · 五元组稳态检测与双重去噪',
           ha='center', va='center', fontsize=9, color='#2471A3', fontstyle='italic', zorder=4)

    # 模块二标题
    box2 = FancyBboxPatch((COL2_X - 3.2, Y_COL_HEADER - header_h/2), 6.4, header_h,
                           boxstyle="round,pad=0.05",
                           facecolor='#F5EEF8', edgecolor=COLORS['feature_border'],
                           linewidth=2.5, zorder=3)
    ax.add_patch(box2)
    ax.text(COL2_X, Y_COL_HEADER + 0.1, '模块二 · 特征层', ha='center', va='center',
           fontsize=14, fontweight='bold', color=COLORS['feature_border'], zorder=4)
    ax.text(COL2_X, Y_COL_HEADER - 0.32, 'Feature Layer · 多物理场解耦与7维特征工程',
           ha='center', va='center', fontsize=9, color='#8E44AD', fontstyle='italic', zorder=4)

    # 模块三标题
    box3 = FancyBboxPatch((COL3_X - 3.2, Y_COL_HEADER - header_h/2), 6.4, header_h,
                           boxstyle="round,pad=0.05",
                           facecolor='#FEF5E7', edgecolor=COLORS['algo_border'],
                           linewidth=2.5, zorder=3)
    ax.add_patch(box3)
    ax.text(COL3_X, Y_COL_HEADER + 0.1, '模块三 · 算法层', ha='center', va='center',
           fontsize=14, fontweight='bold', color=COLORS['algo_border'], zorder=4)
    ax.text(COL3_X, Y_COL_HEADER - 0.32, 'Algorithm Layer · TS-VAE无监督深度生成',
           ha='center', va='center', fontsize=9, color='#BA4A00', fontstyle='italic', zorder=4)

    # =====================================================================
    # 列间水平递进箭头
    # =====================================================================
    draw_horizontal_arrow(ax, 7.5, Y_COL_HEADER, 9.0, Y_COL_HEADER, '阶段递进')
    draw_horizontal_arrow(ax, 14.5, Y_COL_HEADER, 16.0, Y_COL_HEADER, '阶段递进')

    # =====================================================================
    # 模块一：数据层（4行框）
    # ========================================================================

    # 框1：海量原始传感数据阵列
    draw_box_with_shadow(
        ax, COL1_X, Y_ROW1, BOX_W, BOX_H,
        '海量原始传感数据阵列',
        ['1390+ 列高频传感器记录', '空间盲区高达 95.3%'],
        COLORS['data_border'], COLORS['data_fill']
    )

    # 框2：五元组联合稳态判据
    draw_box_with_shadow(
        ax, COL1_X, Y_ROW2, BOX_W, BOX_H,
        '五元组联合稳态判据',
        ['Fv>0 ∧ T>0 ∧ v>0 ∧ n>0 ∧ p>0', '剔除瞬态失真与工况转换干扰'],
        COLORS['data_border'], COLORS['data_fill']
    )

    # 框3：双重自适应流式降噪
    draw_box_with_shadow(
        ax, COL1_X, Y_ROW3, BOX_W, BOX_H,
        '双重自适应流式降噪',
        ['Rolling MAD 滤除电学毛刺', '+ LOF 剔除物理失真'],
        COLORS['data_border'], COLORS['data_fill']
    )

    # 框4：纯净掘进动力学切片
    draw_box_with_shadow(
        ax, COL1_X, Y_ROW4, BOX_W, BOX_H,
        '纯净掘进动力学切片',
        ['消除背景工况干扰', '确立高保真稳态底座'],
        COLORS['data_border'], COLORS['data_fill']
    )

    # =====================================================================
    # 模块二：特征层（4行框）
    # ========================================================================

    # 框1：四大物理场语义解耦
    draw_box_with_shadow(
        ax, COL2_X, Y_ROW1, BOX_W, BOX_H,
        '四大物理场语义解耦',
        ['机械动力 / 电气响应', '热力耗散 / 流体平衡'],
        COLORS['feature_border'], COLORS['feature_fill']
    )

    # 框2：高阶物理不变量衍生
    draw_box_with_shadow(
        ax, COL2_X, Y_ROW2, BOX_W, BOX_H,
        '高阶物理不变量衍生',
        ['SE 表征单位体积切削能耗', 'FPI/TPI 表征法向与剪切阻力'],
        COLORS['feature_border'], COLORS['feature_fill']
    )

    # 框3：共线性消除与贪心选取
    draw_box_with_shadow(
        ax, COL2_X, Y_ROW3, BOX_W, BOX_H,
        '共线性消除与贪心选取',
        ['强制剔除底层基础参数', '基于相关性自适应抓取4个独立代理'],
        COLORS['feature_border'], COLORS['feature_fill']
    )

    # 框4：7维全局独立核心特征张量
    draw_box_with_shadow(
        ax, COL2_X, Y_ROW4, BOX_W, BOX_H,
        '7维全局独立核心特征张量',
        ['3个高阶衍生 + 4个跨域解耦代理', '彻底消除冗余'],
        COLORS['feature_border'], COLORS['feature_fill']
    )

    # =====================================================================
    # 模块三：算法层（4行框）
    # ========================================================================

    # 框1：微观时序序列窗口构建
    draw_box_with_shadow(
        ax, COL3_X, Y_ROW1, BOX_W, BOX_H,
        '微观时序序列窗口构建',
        ['输入张量: 60帧 × 7特征 = 420', '全生命周期滑动切片'],
        COLORS['algo_border'], COLORS['algo_fill']
    )

    # 框2：TS-VAE深度生成引擎
    draw_box_with_shadow(
        ax, COL3_X, Y_ROW2, BOX_W, BOX_H,
        'TS-VAE深度生成引擎',
        ['双层GRU时空编码 | 潜在维度d=16', '拓扑压缩比: 7/16 ≈ 0.44'],
        COLORS['algo_border'], COLORS['algo_fill']
    )

    # 框3：健康模态隐式传导记忆
    draw_box_with_shadow(
        ax, COL3_X, Y_ROW3, BOX_W, BOX_H,
        '健康模态隐式传导记忆',
        ['仅利用前期健康数据', '固化正常多场协同流形'],
        COLORS['algo_border'], COLORS['algo_fill']
    )

    # 框4：HI(R)无监督健康退化指标
    draw_box_with_shadow(
        ax, COL3_X, Y_ROW4, BOX_W, BOX_H,
        'HI(R)无监督健康退化指标',
        ['解码器帧级重构残差MSE环级聚合', '精准锁定异常节点'],
        COLORS['algo_border'], COLORS['algo_fill']
    )

    # =====================================================================
    # 垂直箭头（各列内部）
    # =====================================================================

    for col_x in [COL1_X, COL2_X, COL3_X]:
        # Row1 -> Row2
        draw_arrow(ax, col_x, Y_ROW1 - BOX_H/2, col_x, Y_ROW2 + BOX_H/2 + 0.2, COLORS['arrow'], lw=2.0)
        # Row2 -> Row3
        draw_arrow(ax, col_x, Y_ROW2 - BOX_H/2, col_x, Y_ROW3 + BOX_H/2 + 0.2, COLORS['arrow'], lw=2.0)
        # Row3 -> Row4
        draw_arrow(ax, col_x, Y_ROW3 - BOX_H/2, col_x, Y_ROW4 + BOX_H/2 + 0.2, COLORS['arrow'], lw=2.0)

    # =====================================================================
    # 跨层箭头
    # =====================================================================

    # 数据层 -> 特征层
    draw_arrow(ax, COL1_X + BOX_W/2, Y_ROW4, COL2_X - BOX_W/2, Y_ROW4,
              color='#2980B9', lw=2.5)
    ax.text((COL1_X + BOX_W/2 + COL2_X - BOX_W/2)/2, Y_ROW4 + 0.3,
           '特征工程', ha='center', va='bottom', fontsize=10, color='#2980B9', zorder=6)

    # 特征层 -> 算法层
    draw_arrow(ax, COL2_X + BOX_W/2, Y_ROW4, COL3_X - BOX_W/2, Y_ROW4,
              color='#8E44AD', lw=2.5)
    ax.text((COL2_X + BOX_W/2 + COL3_X - BOX_W/2)/2, Y_ROW4 + 0.3,
           '7D输入', ha='center', va='bottom', fontsize=10, color='#8E44AD', zorder=6)

    # =====================================================================
    # 底部分隔线
    # =====================================================================
    ax.axhline(y=Y_BOTTOM_DIV, color='#BDC3C7', linewidth=1.5, linestyle='--',
              xmin=0.02, xmax=0.98, zorder=1)

    # =====================================================================
    # 底部区域：TS-VAE 网络拓扑 + 理念说明
    # ========================================================================

    ax.text(4, Y_BOTTOM_DIV - 0.3, 'TS-VAE 网络拓扑', ha='center', va='top',
           fontsize=14, fontweight='bold', color='#2C3E50', zorder=4)

    # 网络拓扑元素
    topo_y = Y_BOTTOM_TOPO + 0.6
    topo_h = 0.8
    topo_w = 1.8

    # Input
    box_input = FancyBboxPatch((1.5, topo_y - topo_h/2), topo_w, topo_h,
                               boxstyle="round,pad=0.04",
                               facecolor='#EAF2F8', edgecolor='#1B4F72',
                               linewidth=2, zorder=3)
    ax.add_patch(box_input)
    ax.text(2.4, topo_y + 0.12, 'Input', ha='center', va='center',
           fontsize=11, fontweight='bold', color='#1B4F72', zorder=4)
    ax.text(2.4, topo_y - 0.15, '[60×7]', ha='center', va='center',
           fontsize=9, color='#2471A3', zorder=4)

    # Encoder箭头
    ax.annotate('', xy=(4.0, topo_y), xytext=(3.3, topo_y),
                arrowprops=dict(arrowstyle='->', color='#1B4F72', lw=2),
                zorder=5)
    ax.text(3.65, topo_y + 0.2, 'Encoder', ha='center', va='bottom',
           fontsize=8, color='#2471A3', zorder=6)

    # Latent
    box_latent = FancyBboxPatch((4.0, topo_y - topo_h/2), topo_w, topo_h,
                                boxstyle="round,pad=0.04",
                                facecolor='#F5EEF8', edgecolor='#512E5F',
                                linewidth=2, zorder=3)
    ax.add_patch(box_latent)
    ax.text(4.9, topo_y + 0.12, 'Latent', ha='center', va='center',
           fontsize=11, fontweight='bold', color='#512E5F', zorder=4)
    ax.text(4.9, topo_y - 0.15, 'Z=16', ha='center', va='center',
           fontsize=9, color='#8E44AD', zorder=4)

    # Decoder箭头
    ax.annotate('', xy=(6.5, topo_y), xytext=(5.8, topo_y),
                arrowprops=dict(arrowstyle='->', color='#935116', lw=2),
                zorder=5)
    ax.text(6.15, topo_y + 0.2, 'Decoder', ha='center', va='bottom',
           fontsize=8, color='#BA4A00', zorder=6)

    # Output
    box_output = FancyBboxPatch((6.5, topo_y - topo_h/2), topo_w, topo_h,
                                boxstyle="round,pad=0.04",
                                facecolor='#FEF5E7', edgecolor='#935116',
                                linewidth=2, zorder=3)
    ax.add_patch(box_output)
    ax.text(7.4, topo_y + 0.12, 'Output', ha='center', va='center',
           fontsize=11, fontweight='bold', color='#935116', zorder=4)
    ax.text(7.4, topo_y - 0.15, '[60×7]', ha='center', va='center',
           fontsize=9, color='#BA4A00', zorder=4)

    # ELBO箭头
    ax.annotate('', xy=(8.8, topo_y), xytext=(8.3, topo_y),
                arrowprops=dict(arrowstyle='->', color='#7D3C98', lw=2),
                zorder=5)

    # ELBO Loss
    box_elbo = FancyBboxPatch((8.8, topo_y - topo_h/2), topo_w, topo_h,
                              boxstyle="round,pad=0.04",
                              facecolor='#F5EEF8', edgecolor='#7D3C98',
                              linewidth=2, zorder=3)
    ax.add_patch(box_elbo)
    ax.text(9.7, topo_y + 0.12, 'ELBO', ha='center', va='center',
           fontsize=11, fontweight='bold', color='#7D3C98', zorder=4)
    ax.text(9.7, topo_y - 0.15, 'L+0.001*L_KL', ha='center', va='center',
           fontsize=9, color='#9B59B6', zorder=4)

    # =====================================================================
    # 右侧：无监督与降维理念
    # ========================================================================

    ax.text(14, Y_BOTTOM_DIV - 0.3, '核心理念', ha='center', va='top',
           fontsize=14, fontweight='bold', color='#2C3E50', zorder=4)

    ideas = [
        '隐风险 95.3% 标签缺失死结',
        '0.44 压缩比防线性复制',
        '第一性原理引导',
    ]

    for i, idea in enumerate(ideas):
        ax.text(14, Y_BOTTOM_TOPO + 0.5 - i * 0.45, f'• {idea}',
               ha='center', va='top', fontsize=11, color='#566573', zorder=4)

    # =====================================================================
    # 底部HI(R)环号序列
    # ========================================================================

    ax.text(4, 0.3, 'HI(R) 全生命周期退化', ha='center', va='bottom',
           fontsize=11, fontweight='bold', color='#2C3E50', zorder=4)

    rings = [
        (1.5, 121, 227.1, '#27AE60'),
        (2.7, 122, 309.9, '#27AE60'),
        (3.9, 123, 112.3, '#27AE60'),
        (5.1, 124, 119.0, '#27AE60'),
        (6.3, 125, 126.8, '#27AE60'),
        (7.5, 126, 189.8, '#F39C12'),
        (8.7, 127, 210.2, '#F39C12'),
        (9.9, 128, 121.3, '#27AE60'),
        (11.1, 129, 176.0, '#F39C12'),
        (12.3, 130, 576.6, '#E74C3C'),
    ]

    for x, rid, hi, color in rings:
        circle = plt.Circle((x, 0.25), 0.22, facecolor=color, edgecolor='white',
                           linewidth=1.5, alpha=0.9, zorder=5)
        ax.add_patch(circle)
        ax.text(x, 0.25, f'R{rid}', ha='center', va='center',
               fontsize=9, color='white', fontweight='bold', zorder=6)
        ax.text(x, -0.08, f'{hi:.0f}', ha='center', va='center',
               fontsize=9, color=color, fontweight='bold', zorder=6)

    ax.text(0.8, 0.25, 'HI(R):', ha='center', va='center',
           fontsize=10, color='#2C3E50', fontweight='bold', zorder=6)

    # =====================================================================
    # 保存
    # =====================================================================

    plt.tight_layout(pad=0.5)
    from pathlib import Path
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Figure] Saved to: {save_path}")


if __name__ == '__main__':
    save_path = "C:/NO CLUB/experience/TBM_Cutter_Wear_Project/results/fig1_pipeline_workflow.png"
    generate_figure(save_path)