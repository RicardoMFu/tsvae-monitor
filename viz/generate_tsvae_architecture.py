"""
TS-VAE 顶刊级 3D 架构矢量图 V5
==============================
基于v3（1400×450）画布，修复文字排版拥挤问题：
- Z-Score文字有独立空间
- Reparameterization区域紧凑排版
- MSE/HI间距适当拉开但不浪费
"""

def svg_header(w: int, h: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        '<rect width="100%" height="100%" fill="#FFFFFF"/>\n'
        '<defs>\n'
        '  <marker id="arr" markerWidth="9" markerHeight="7" '
        'refX="8" refY="3.2" orient="auto">\n'
        '    <polygon points="0 0, 9 3.2, 0 7" fill="#37474F"/>\n'
        '  </marker>\n'
        '</defs>\n'
    )

def footer() -> str:
    return '</svg>\n'

def tag(x, y, txt, size=26, color="#B71C1C", bold=True, anchor="middle"):
    w = 'font-weight="bold"' if bold else ''
    return (f'<text x="{x:.1f}" y="{y:.1f}" '
            f'font-family="Microsoft YaHei,sans-serif" '
            f'font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" {w}>{txt}</text>\n')

def tag_sub(x, y, txt, size=20, color="#455A64"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" '
            f'font-family="Microsoft YaHei,sans-serif" '
            f'font-size="{size}" fill="{color}" '
            f'text-anchor="middle">{txt}</text>\n')

def arrow(x1, y1, x2, y2, color="#37474F", lw=3.5):
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
            f'x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{lw}" '
            f'marker-end="url(#arr)"/>\n')

def cuboid(svg, cx, cy, W, H, D, fc, ec="#546E7A", lw=2.0):
    x0, y0 = cx - W/2, cy - H/2
    svg.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{W:.1f}" height="{H:.1f}" '
               f'fill="{fc}" stroke="{ec}" stroke-width="{lw}"/>\n')
    svg.append(f'<polygon points="'
               f'{x0:.1f},{y0:.1f} '
               f'{x0+D:.1f},{y0-D:.1f} '
               f'{x0+W+D:.1f},{y0-D:.1f} '
               f'{x0+W:.1f},{y0:.1f}" '
               f'fill="{fc}" stroke="{ec}" stroke-width="{lw}" opacity="0.85"/>\n')
    svg.append(f'<polygon points="'
               f'{x0+W:.1f},{y0:.1f} '
               f'{x0+W+D:.1f},{y0-D:.1f} '
               f'{x0+W+D:.1f},{y0+H-D:.1f} '
               f'{x0+W:.1f},{y0+H:.1f}" '
               f'fill="{fc}" stroke="{ec}" stroke-width="{lw}" opacity="0.6"/>\n')

def gru_pair(svg, cx1, cx2, Y, H, W, D, fc, label_top, label_dim):
    cuboid(svg, cx1, Y, W, H, D, fc)
    cuboid(svg, cx2, Y, W, H, D, fc)
    cx = (cx1 + cx2) / 2
    svg.append(tag(cx, Y - H/2 - 38, label_top, size=23))
    svg.append(tag_sub(cx, Y + H/2 + 38, label_dim, size=17))

def generate(save_path="viz/svg/tsvae_architecture_v5.svg"):
    W, H = 1400, 450
    Y = H // 2   # Y=225

    svg = []
    svg.append(svg_header(W, H))
    svg.append('<style>text{font-family:"Microsoft YaHei",sans-serif}</style>\n')

    YELLOW = "#FFF9C4"
    BLUE   = "#E3F2FD"
    PURPLE = "#F3E5F5"
    MINT   = "#E0F2F1"
    RED    = "#FFEBEE"

    # Cuboid尺寸（与v3一致，巨型化）
    I_H, I_W, I_D = 200, 80, 40   # 输入/输出
    G_H, G_W, G_D = 160, 100, 80  # Encoder/Decoder
    L_H, L_W, L_D = 100, 30, 20   # Latent（窄瓶颈）
    M_H, M_W, M_D = 55, 44, 18    # μ / logσ²

    # ── 节点X坐标（v3紧凑排列，仅局部微调）─────────────────
    input_x  =  90
    enc1_x   = 228
    enc2_x   = 358
    mu_x     = 515
    sigma2_x = 580
    latent_x  = 740
    dec1_x    = 858
    dec2_x    = 988
    output_x  = 1125
    mse_x     = 1250
    hi_x      = 1340

    # ── 1. 输入张量 ──────────────────────────────────────────
    cuboid(svg, input_x, Y, I_W, I_H, I_D, YELLOW)
    svg.append(tag(input_x, Y - I_H/2 - 38, "输入张量", size=23, color="#B71C1C"))
    svg.append(tag_sub(input_x, Y + I_H/2 + 36, "(B, 60, 10)", size=17))
    svg.append(tag_sub(input_x, Y + I_H/2 + 60, "时序窗口 60s×10维", size=15, color="#90A4AE"))

    # ── 2. Z-Score 过渡 ──────────────────────────────────────
    # 箭头从input右侧到encoder左侧，文字居中放在箭头下方空白处
    svg.append(arrow(input_x + I_W/2 + 4, Y, enc1_x - G_W/2 - 4, Y, "#00ACC1", 3.5))
    # Z-Score文字：向左微调，下方文字放大
    zscore_cx = (input_x + I_W/2 + 4 + enc1_x - G_W/2 - 4) / 2 - 8
    svg.append(tag(zscore_cx, Y + 44, "Z-Score", size=16, color="#00838F", bold=False))
    svg.append(tag_sub(zscore_cx, Y + 67, "归一化", size=17, color="#4DD0E1"))

    # ── 3. Encoder GRU ×2 ─────────────────────────────────────
    gru_pair(svg, enc1_x, enc2_x, Y, G_H, G_W, G_D, BLUE,
             "Encoder GRU (×2)", "hidden=64, layers=2")

    # Encoder → μ/σ²（向上下分开，避免与重参数区域重叠）
    svg.append(arrow(enc2_x + G_W/2, Y - 28, mu_x - M_W/2, Y - M_H/2 - 4, "#1565C0", 3))
    svg.append(arrow(enc2_x + G_W/2, Y + 28, sigma2_x - M_W/2, Y + M_H/2 + 4, "#1565C0", 3))

    # ── 4. μ 和 logσ² ─────────────────────────────────────────
    cuboid(svg, mu_x, Y, M_W, M_H, M_D, BLUE)
    svg.append(tag(mu_x, Y - M_H/2 - 34, "μ", size=22, color="#1565C0"))
    svg.append(tag_sub(mu_x, Y + M_H/2 + 34, "(B, 16)", size=15))

    cuboid(svg, sigma2_x, Y, M_W, M_H, M_D, BLUE)
    svg.append(tag(sigma2_x, Y - M_H/2 - 34, "logσ²", size=22, color="#1565C0"))
    svg.append(tag_sub(sigma2_x, Y + M_H/2 + 34, "(B, 16)", size=15))

    # μ → latent（单一直线，Y=250穿过）
    svg.append(arrow(mu_x + M_W/2, Y, latent_x - L_W/2 - 10, Y, "#5C6BC0", 3.5))
    # σ² → latent（从上方 Y-30 斜接同一终点）
    svg.append(arrow(sigma2_x + M_W/2, Y - 30, latent_x - L_W/2 - 10, Y - 8, "#5C6BC0", 3.5))

    # ── 5. Reparameterization（上下展开，充分利用空白）────────
    # 上公式 → Y-78（靠近μ盒子上缘的留白区）
    # Reparam. → Y（中间，粗体）
    # 下公式 → Y+68（靠近σ²盒子下缘的留白区）
    reparam_x = 653
    svg.append(tag(reparam_x, Y - 108, "z = μ + σ⊙ε", size=20, color="#1a1a2e", bold=True))
    svg.append(tag(reparam_x, Y + 22,  "Reparam.",      size=17, color="#546E7A", bold=True))
    svg.append(tag_sub(reparam_x, Y + 68, "ε ~ N(0, I)", size=17, color="#546E7A"))

    # ── 6. Latent Z（极窄瓶颈）───────────────────────────────

    # ── 6. Latent Z（极窄瓶颈）───────────────────────────────
    cuboid(svg, latent_x, Y, L_W, L_H, L_D, PURPLE, lw=2.5)
    svg.append(tag(latent_x, Y - L_H/2 - 34, "潜在流形 Z", size=21, color="#6A1B9A"))
    svg.append(tag_sub(latent_x, Y + L_H/2 + 34, "dim=16", size=16, color="#6A1B9A"))
    svg.append(tag_sub(latent_x, Y + L_H/2 + 56, "健康流形", size=16, color="#9C27B0"))

    # latent → decoder
    svg.append(arrow(latent_x + L_W/2, Y, dec1_x - G_W/2, Y, "#00695C", 3.5))

    # ── 7. Decoder GRU ×2 ─────────────────────────────────────
    gru_pair(svg, dec1_x, dec2_x, Y, G_H, G_W, G_D, MINT,
             "Decoder GRU (×2)", "hidden=64, layers=2")

    # decoder → output
    svg.append(arrow(dec2_x + G_W/2, Y, output_x - I_W/2, Y, "#00695C", 3.5))

    # ── 8. 重构输出 ───────────────────────────────────────────
    cuboid(svg, output_x, Y, I_W, I_H, I_D, YELLOW)
    svg.append(tag(output_x, Y - I_H/2 - 38, "重构输出 X̂", size=23, color="#2E7D32"))
    svg.append(tag_sub(output_x, Y + I_H/2 + 36, "(B, 60, 10)", size=17))
    # 内嵌波形
    pts = (f"{output_x-28},{Y+5} "
           f"{output_x-8},{Y-13} "
           f"{output_x+12},{Y+9} "
           f"{output_x+28},{Y-6}")
    svg.append(f'<polyline points="{pts}" '
               f'fill="none" stroke="#388E3C" stroke-width="2.2" opacity="0.8"/>\n')

    # output → MSE（直线：output右边缘 → MSE左边缘-箭头offset）
    svg.append(arrow(output_x + I_W/2, Y, mse_x - 52, Y, "#C62828", 3.5))

    # ── 9. MSE ────────────────────────────────────────────────
    cuboid(svg, mse_x, Y, 60, 82, 24, RED)
    svg.append(tag(mse_x, Y - 82/2 - 36, "MSE", size=26, color="#B71C1C"))
    svg.append(tag_sub(mse_x, Y + 82/2 + 38, "重构均方误差", size=16, color="#E53935"))

    # MSE → HI（直线，右箭头：mse右缘 → HI左缘）
    svg.append(arrow(mse_x + 52, Y, hi_x - 56, Y, "#C62828", 3.5))

    # ── 10. HI(R) ─────────────────────────────────────────────
    cuboid(svg, hi_x, Y, 55, 72, 22, "#FFCDD2")
    svg.append(tag(hi_x, Y - 72/2 - 36, "HI(R)", size=26, color="#B71C1C"))
    svg.append(tag_sub(hi_x, Y + 72/2 + 28, "环级健康指标", size=16, color="#E53935"))

    # ── 11. ELBO底部标注 ─────────────────────────────────────
    svg.append(tag(W/2, H - 24,
                   "ELBO = MSE(X, X̂) + β·D_KL(q(z|X) ‖ N(0,I))    β=0.001    "
                   "优化器：AdamW (lr=1e-3, weight_decay=1e-4)",
                   size=23, color="#37474F"))
    svg.append(tag_sub(W/2, H - 2,
                       "参数量：85,162 | 潜在维度：d=16 | GRU隐藏单元：64 | 层数：2",
                       size=19, color="#90A4AE"))

    # ── 写入 ──────────────────────────────────────────────────
    svg.append(footer())
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.writelines(svg)

    print(f"[TS-VAE v5] 生成：{save_path}  画布：{W}×{H}  Y_CENTER={Y}")

if __name__ == "__main__":
    generate()