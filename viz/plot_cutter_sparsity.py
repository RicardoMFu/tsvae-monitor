"""
刀具标定数据稀疏性热力图
=======================
颜色直接由Excel字体colour_idx决定：
  colour_idx=10  → RGB(255,0,0) 纯红  → 标红异常（偏磨/崩断）
  colour_idx=8   → RGB(0,0,0)   纯黑  → 特殊/正常
  colour_idx=32767 → Excel默认色    → 普通磨损
  colour_idx=57  → RGB(51,153,102) 绿  → 特殊/其他
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re, xlrd

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False

# ── Step1：读取Excel和颜色元数据 ───────────────────────────────
wb = xlrd.open_workbook('viz/record_to_2.26.xls', formatting_info=True)
ws = wb.sheet_by_index(0)

ring_ids_raw = [ws.cell(0, j).value for j in range(1, ws.ncols)]
dates_raw    = [ws.cell(1, j).value for j in range(1, ws.ncols)]
cutter_labels = [ws.cell(r, 0).value for r in range(2, ws.nrows)]

cell_color = {}
for row in range(2, ws.nrows):
    for col in range(1, ws.ncols):
        cell = ws.cell(row, col)
        if cell.value and str(cell.value).strip() not in ['', 'nan']:
            xf = wb.xf_list[cell.xf_index]
            font = wb.font_list[xf.font_index]
            cell_color[(row - 2, col - 1)] = font.colour_index

# ── Step2：磨损值计算 ──────────────────────────────────────────
def extract_wear(txt):
    if txt is None: return np.nan
    t = str(txt).strip()
    if '崩断' in t or '断裂' in t or '刀圈崩断' in t: return 0.95
    m = re.search(r'偏磨(\d+)', t)
    if m:
        v = int(m.group(1))
        return min(0.5 + v / 100 * 0.4, 0.9)
    m2 = re.search(r'磨损(\d+)', t)
    if m2:
        v = int(m2.group(1))
        return min(0.2 + v / 100 * 0.3, 0.5)
    if '正常' in t or '换刀' in t or '保留' in t: return 0.15
    return 0.5

# ── Step3：构建数据矩阵 ──────────────────────────────────────
n_rows = ws.nrows - 2
n_cols = ws.ncols - 1
wear_matrix  = np.full((n_rows, n_cols), np.nan)
color_matrix = np.full((n_rows, n_cols), -1, dtype=int)

for r in range(n_rows):
    for c in range(n_cols):
        cell = ws.cell(r + 2, c + 1)
        if cell.value and str(cell.value).strip() not in ['', 'nan']:
            wear_matrix[r, c]  = extract_wear(cell.value)
            color_matrix[r, c] = cell_color.get((r, c), 32767)

# ── Step4：环号去重合并 ──────────────────────────────────────
ring_to_cols = {}
for c, r_raw in enumerate(ring_ids_raw):
    try: k = int(r_raw)
    except:
        m = re.search(r'\d+', str(r_raw))
        k = int(m.group()) if m else c
    if k not in ring_to_cols: ring_to_cols[k] = []
    ring_to_cols[k].append(c)

unique_rings = sorted(ring_to_cols.keys())
print(f"唯一环号: {len(unique_rings)}, 范围: {unique_rings[0]}~{unique_rings[-1]}")

n_cutters = n_rows
n_rings   = len(unique_rings)
merged_wear    = np.full((n_cutters, n_rings), np.nan)
merged_color  = np.full((n_cutters, n_rings), -1, dtype=int)
merged_text   = [['' for _ in range(n_rings)] for __ in range(n_cutters)]

for ci, ring in enumerate(unique_rings):
    for col in ring_to_cols[ring]:
        for ri in range(n_cutters):
            if not np.isnan(wear_matrix[ri, col]):
                cur = merged_wear[ri, ci]
                if np.isnan(cur) or wear_matrix[ri, col] > cur:
                    merged_wear[ri, ci]   = wear_matrix[ri, col]
                    merged_color[ri, ci]  = color_matrix[ri, col]
                    merged_text[ri][ci]    = str(ws.cell(ri + 2, col + 1).value).strip()

clean_labels = []
for c in cutter_labels:
    if isinstance(c, float) and abs(c - 0.333) < 0.01:
        clean_labels.append('0.33')
    else:
        clean_labels.append(str(c).strip() if str(c).strip() else '?')

# ── Step5：日期排序 ─────────────────────────────────────────
def month_key(s):
    m = re.search(r'^(\d+)\.(\d+)', s)
    if m:
        mo, day = int(m.group(1)), int(m.group(2))
        order_map = {11: 0, 12: 1, 1: 2, 2: 3}
        return (order_map.get(mo, 99), day)
    return (99, 99)

date_to_ring = {}
for j, (r_raw, d_raw) in enumerate(zip(ring_ids_raw, dates_raw)):
    try: k = int(r_raw)
    except:
        m = re.search(r'\d+', str(r_raw))
        k = int(m.group()) if m else j
    ds = str(d_raw).strip()
    if ds not in date_to_ring: date_to_ring[ds] = []
    date_to_ring[ds].append((k, j))

unique_dates = sorted(date_to_ring.keys(), key=month_key)
print(f"唯一日期: {len(unique_dates)}")

n_dates = len(unique_dates)
merged_wear_by_date    = np.full((n_cutters, n_dates), np.nan)
merged_color_by_date  = np.full((n_cutters, n_dates), -1, dtype=int)
merged_text_by_date    = [['' for _ in range(n_dates)] for __ in range(n_cutters)]

for ci, date_str in enumerate(unique_dates):
    for (ring_k, col_j) in date_to_ring[date_str]:
        for ri in range(n_cutters):
            if not np.isnan(wear_matrix[ri, col_j]):
                cur = merged_wear_by_date[ri, ci]
                if np.isnan(cur) or wear_matrix[ri, col_j] > cur:
                    merged_wear_by_date[ri, ci]   = wear_matrix[ri, col_j]
                    merged_color_by_date[ri, ci]  = color_matrix[ri, col_j]
                    merged_text_by_date[ri][ci]    = str(ws.cell(ri + 2, col_j + 1).value).strip()

# ── 颜色：由colour_idx决定，配合文字关键词 ───────────────────
def get_cell_color(color_idx, txt=''):
    """colour_idx → RGBA：
      10  = RGB(255,0,0)   纯红 → 标红异常（偏磨/崩断）
      8   = RGB(0,0,0)     纯黑 → 特殊/正常
      32767 = Excel默认色  → 普通磨损（>5mm为主）
      57  = RGB(51,153,102) 绿 → 探头故障/其他特殊
    32767默认色若含崩断/偏磨关键词 → 升级为标红
    """
    if color_idx == 10:                     # 纯红
        return (1.0, 0.0, 0.0, 1.0)
    if color_idx == 8:                       # 纯黑
        return (0.0, 0.0, 0.0, 1.0)
    if color_idx == 57:                      # 绿色
        return (0.2, 0.6, 0.4, 1.0)
    # 32767 默认色，含崩断/偏磨关键词 → 升级为标红
    if txt and ('崩断' in txt or '偏磨' in txt):
        return (1.0, 0.0, 0.0, 1.0)
    # 普通磨损 → 中灰色
    return (0.55, 0.55, 0.55, 1.0)

# ── 绘制函数 ──────────────────────────────────────────────────
def draw_heatmap(ax, merged, labels, x_ticks, title, xlabel, color_mat, text_mat=None,
                 toplabel_right='', n_red=0, n_green=0, n_gray=0, n_black=0):
    n_r, n_c = merged.shape
    if text_mat is None:
        text_mat = [['' for _ in range(n_c)] for __ in range(n_r)]

    # 灰色背景
    bg_data = np.full((n_r, n_c), 0.88)
    ax.imshow(bg_data, cmap='gray', vmin=0, vmax=1, aspect='auto')

    # 绘制数据格子
    for i in range(n_r):
        for j in range(n_c):
            v = merged[i, j]
            if np.isnan(v):
                continue
            ci = color_mat[i, j]
            txt = text_mat[i][j] if text_mat else ''
            rgba = get_cell_color(ci if ci != -1 else 32767, txt)
            rect = plt.Rectangle((j - 0.42, i - 0.42), 0.84, 0.84,
                                  facecolor=rgba[:3], edgecolor='none', alpha=0.9)
            ax.add_patch(rect)
            # 标红 → 深红边框
            if rgba[0] > 0.9 and rgba[1] < 0.1 and rgba[2] < 0.1:
                rect2 = plt.Rectangle((j - 0.42, i - 0.42), 0.84, 0.84,
                                       facecolor='none', edgecolor='darkred', linewidth=2.5)
                ax.add_patch(rect2)

    ax.set_xticks(range(n_c))
    ax.set_xticklabels(x_ticks, rotation=45, ha='right', fontsize=13)
    ax.set_xlim(-0.5, n_c - 0.5)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_ylim(n_r - 0.5, -0.5)
    ax.set_xlabel(xlabel, fontsize=16, labelpad=10)
    ax.set_title(title, fontsize=22, fontweight='bold', pad=22)

    # 右上角：顶部标注（紧邻标题右端）
    if toplabel_right:
        ax.text(1.01, 1.06, toplabel_right,
                transform=ax.transAxes, fontsize=14,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.6', fc='white', ec='#546E7A', alpha=0.92))

    # 左上角：数据密度统计（大字）
    non_null = np.sum(~np.isnan(merged))
    total = merged.size
    ax.text(0.01, 1.06,
            f'数据密度：{non_null}/{total}（{100*(1-non_null/total):.1f}%空白）',
            transform=ax.transAxes, fontsize=14,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.90))

    # 右上角：标红异常记录数（大字）
    if n_red > 0:
        ax.text(0.99, 1.02,
                f'标红异常记录：{n_red}条',
                transform=ax.transAxes, fontsize=14,
                va='top', ha='right', color='#CC0000',
                bbox=dict(boxstyle='round,pad=0.5', fc='#FFF0F0', ec='#CC0000', alpha=0.90))

    # 左上角下方：颜色图例（用小色块标注各颜色含义）
    legend_y = 0.93
    patches = [
        mpatches.Patch(facecolor=(1.0, 0.0, 0.0), edgecolor='darkred', linewidth=1.5,
                       label=f'红 — 标红异常（偏磨/崩断）{n_red}条'),
        mpatches.Patch(facecolor=(0.55, 0.55, 0.55), edgecolor='none',
                       label=f'灰 — 普通磨损（>5mm）{n_gray}条'),
        mpatches.Patch(facecolor=(0.0, 0.0, 0.0), edgecolor='none',
                       label=f'黑 — 特殊/正常（≤5mm）{n_black}条'),
        mpatches.Patch(facecolor=(0.2, 0.6, 0.4), edgecolor='none',
                       label=f'绿 — 探头故障等特殊{n_green}条'),
        mpatches.Patch(facecolor=(0.88, 0.88, 0.88), edgecolor='none',
                       label=f'灰白 — 无记录（空白区域）'),
    ]
    leg = ax.legend(handles=patches, loc='upper left',
                    bbox_to_anchor=(0.01, 0.85),
                    fontsize=13, framealpha=0.92,
                    edgecolor='#546E7A', fancybox=True)
    leg.get_frame().set_linewidth(0.8)


# ── 统计汇总 ──────────────────────────────────────────────────
c10_ring    = int(np.sum(merged_color == 10))
c8_ring     = int(np.sum(merged_color == 8))
c57_ring    = int(np.sum(merged_color == 57))
c32767_ring  = int(np.sum(merged_color == 32767))
c10_date    = int(np.sum(merged_color_by_date == 10))
c8_date     = int(np.sum(merged_color_by_date == 8))
c57_date    = int(np.sum(merged_color_by_date == 57))
c32767_date  = int(np.sum(merged_color_by_date == 32767))

# 实际显示的红色条数（含关键词升级）
def count_red(color_mat, text_mat, n_r, n_c):
    n = 0
    for i in range(n_r):
        for j in range(n_c):
            ci = color_mat[i, j]
            txt = text_mat[i][j] if text_mat else ''
            rgba = get_cell_color(ci if ci != -1 else 32767, txt)
            if rgba[0] > 0.9 and rgba[1] < 0.1 and rgba[2] < 0.1:
                n += 1
    return n

n_red_ring = count_red(merged_color, merged_text, n_cutters, n_rings)
n_red_date = count_red(merged_color_by_date, merged_text_by_date, n_cutters, n_dates)

print(f"\n=== 颜色统计 ===")
print(f"环号版: 标红={n_red_ring}条, 普通灰={c32767_ring}条, 特殊黑={c8_ring}条, 特殊绿={c57_ring}条")
print(f"日期版: 标红={n_red_date}条, 普通灰={c32767_date}条, 特殊黑={c8_date}条, 特殊绿={c57_date}条")


# ── 绘制环号版 ────────────────────────────────────────────────
ring_labels = [str(r) for r in unique_rings]

fig, ax = plt.subplots(figsize=(20, 15), facecolor='white')
draw_heatmap(ax, merged_wear, clean_labels, ring_labels,
            '盾构机刀具标定数据稀疏性热力图（按环号）',
            '环号 (Ring ID)', merged_color, merged_text,
            toplabel_right='TS-VAE健康基线区间：Ring 121~123',
            n_red=n_red_ring, n_green=c57_ring, n_gray=c32767_ring, n_black=c8_ring)

# 健康基线蓝色竖条（不挡数据）
focus_idx = [i for i, r in enumerate(unique_rings) if 121 <= r <= 123]
if focus_idx:
    ax.axvspan(focus_idx[0] - 0.5, focus_idx[-1] + 0.5,
               alpha=0.18, color='#4361EE', zorder=0)

plt.tight_layout()
plt.savefig('viz/svg/cutter_sparsity_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
print("[环号版] 生成")
plt.close()

# ── 绘制日期版 ────────────────────────────────────────────────
short_dates = []
for d in unique_dates:
    m = re.search(r'^(\d+)\.(\d+)', d)
    if m:
        short_dates.append(f"{m.group(1)}.{m.group(2)}")
    else:
        short_dates.append(d[:6])

fig, ax = plt.subplots(figsize=(20, 15), facecolor='white')
draw_heatmap(ax, merged_wear_by_date, clean_labels, short_dates,
            '盾构机刀具标定数据稀疏性热力图（按日期）',
            '日期 (Date)', merged_color_by_date, merged_text_by_date,
            toplabel_right='掘进活跃期：12月~2月',
            n_red=n_red_date, n_green=c57_date, n_gray=c32767_date, n_black=c8_date)

# 掘进活跃期蓝色竖条
dec_idx = [i for i, d in enumerate(unique_dates) if any(d.startswith(p) for p in ['12.', '1.', '2.'])]
if dec_idx:
    ax.axvspan(dec_idx[0] - 0.5, dec_idx[-1] + 0.5,
               alpha=0.18, color='#4361EE', zorder=0)

plt.tight_layout()
plt.savefig('viz/svg/cutter_sparsity_heatmap_by_date.png', dpi=300, bbox_inches='tight', facecolor='white')
print("[日期版] 生成")
plt.close()

# ── 并排合并 ──────────────────────────────────────────────────
from PIL import Image
img1 = Image.open('viz/svg/cutter_sparsity_heatmap.png')
img2 = Image.open('viz/svg/cutter_sparsity_heatmap_by_date.png')
w1, h1 = img1.size
w2, h2 = img2.size
combined = Image.new('RGB', (w1 + w2 + 50, max(h1, h2)), (255, 255, 255))
combined.paste(img1, (0, 0))
combined.paste(img2, (w1 + 50, 0))
combined.save('viz/svg/cutter_sparsity_combined.png', dpi=(300, 300))

import os
for f in ['viz/svg/cutter_sparsity_heatmap.png',
          'viz/svg/cutter_sparsity_heatmap_by_date.png',
          'viz/svg/cutter_sparsity_combined.png']:
    print(f"{f}: {os.path.getsize(f)/1024:.0f} KB")