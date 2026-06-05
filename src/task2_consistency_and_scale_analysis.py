# -*- coding: utf-8 -*-
"""
第二章：运动学参考一致性与操纵性尺度分析最终版代码

主要修改：
1. 保持原期刊绘图格式：宋体、Times New Roman、7.5 号字、600 dpi、线宽等不变；
2. 统一图中文字术语：
   - “AIS位置报告中的对地航速”
   - “AIS位置报告中的动态信息”
   - “相邻位置点计算结果”
3. 修正一致性统计表字段，避免原表中“失配比例/%”重复且阈值不明；
4. 新增“对地航向变化率分布对比图”，用于支撑回转半径偏差分析；
5. 保留速度一致性散点图、互相关滞后图、回转半径分位图和减速距离统计表；
6. 修复原代码中 Dpos50、Dais50 等未定义变量导致表格生成失败的问题。

运行前请确保同目录下存在：
- ais-USV_filled.csv
- simsun.ttc
- simsunb.ttf
- times.ttf
- timesbd.ttf
- timesi.ttf

若文件名带括号，如 ais-USV_filled(7).csv，本代码也会自动尝试匹配。
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredOffsetbox, AnnotationBbox, HPacker, VPacker, TextArea

# =========================
# 0. 路径设置
# =========================
CSV_CANDIDATES = [
    "ais-USV_filled.csv",
    "ais-USV_filled(7).csv",
    "/mnt/data/ais-USV_filled(7).csv",
]

OUT_DIR = Path("task2_consistency_final3")
OUT_DIR.mkdir(exist_ok=True)


def resolve_file(candidates):
    for item in candidates:
        p = Path(item)
        if p.exists():
            return p
    raise FileNotFoundError("未找到 AIS 数据文件，请检查路径：\n" + "\n".join(candidates))


CSV_PATH = resolve_file(CSV_CANDIDATES)

# =========================
# 1. 字体配置：保持原格式
# =========================
FONT_DIR = Path(".")


def font_or_family(filename, family):
    p = FONT_DIR / filename
    if p.exists():
        return FontProperties(fname=str(p))
    return FontProperties(family=family)


FP_SIMSUN = font_or_family("simsun.ttc", "SimSun")
FP_SIMSUN_BOLD = font_or_family("simsunb.ttf", "SimSun")
FP_TIMES = font_or_family("times.ttf", "Times New Roman")
FP_TIMES_BOLD = font_or_family("timesbd.ttf", "Times New Roman")
FP_TIMES_ITALIC = font_or_family("timesi.ttf", "Times New Roman")

for fn in ["simsun.ttc", "simsunb.ttf", "times.ttf", "timesbd.ttf", "timesi.ttf"]:
    p = FONT_DIR / fn
    if p.exists():
        matplotlib.font_manager.fontManager.addfont(str(p))

TIMES_NAME = FP_TIMES.get_name()

rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["font.family"] = TIMES_NAME
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = TIMES_NAME
rcParams["mathtext.it"] = f"{TIMES_NAME}:italic"
rcParams["mathtext.bf"] = f"{TIMES_NAME}:bold"
rcParams["mathtext.default"] = "rm"

plt.rcParams.update({
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "axes.linewidth": 0.8,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "figure.figsize": (3, 2),
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
})

# =========================
# 2. 通用绘图工具函数
# =========================
def beautify_axes(ax, labelpad=5, tickpad=2):
    ax.xaxis.labelpad = labelpad
    ax.yaxis.labelpad = labelpad
    ax.tick_params(axis="both", which="major", direction="in", pad=tickpad, length=4, width=0.8, labelsize=7.0)
    ax.tick_params(axis="both", which="minor", direction="in", pad=tickpad, length=2, width=0.6)
    ax.margins(x=0.03, y=0.03)


def set_ticklabel_font(ax, x_font=FP_TIMES, y_font=FP_TIMES):
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(x_font)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(y_font)
    ax.xaxis.get_offset_text().set_fontproperties(x_font)
    ax.yaxis.get_offset_text().set_fontproperties(y_font)


def _ta(text, fp, fs):
    return TextArea(text, textprops=dict(fontproperties=fp, fontsize=fs))


def add_packed_xlabel(ax, parts, y=-0.18):
    ax.set_xlabel("")
    box = HPacker(children=[_ta(txt, fp, fs) for txt, fp, fs in parts], align="center", pad=0, sep=0)
    anchored = AnchoredOffsetbox(
        loc="lower center",
        child=box,
        frameon=False,
        bbox_to_anchor=(0.5, y),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
        pad=0.0,
    )
    ax.add_artist(anchored)


def add_mixed_ylabel_plain(ax, cn_text, en_text, x=-0.11, fontsize=7.5):
    ax.set_ylabel("")
    ax.text(
        x, 0.64, cn_text,
        transform=ax.transAxes,
        ha="center", va="top",
        rotation=90,
        fontproperties=FP_SIMSUN,
        fontsize=fontsize,
    )
    ax.text(
        x, 0.64, en_text,
        transform=ax.transAxes,
        ha="center", va="bottom",
        rotation=90,
        fontproperties=FP_TIMES,
        fontsize=fontsize,
    )


def add_stats_box_speed(ax, mae, rmse, mismatch_abs, mismatch_rel):
    rows = [
        _ta(f"MAE={mae:.3f}, RMSE={rmse:.3f}", FP_TIMES, 5.5),
        _ta(f"|Δv|>{TAU_ABS:.1f} m/s: {mismatch_abs * 100:.1f}%", FP_TIMES, 5.5),
        _ta(f"|Δv|>{TAU_REL:.1f}×SOG: {mismatch_rel * 100:.1f}%", FP_TIMES, 5.5),
    ]
    vbox = VPacker(children=rows, align="left", pad=0, sep=1)
    anchored = AnchoredOffsetbox(
        loc="upper left",
        child=vbox,
        frameon=True,
        pad=0.2,
        borderpad=0.35,
        bbox_to_anchor=(0.02, 0.98),
        bbox_transform=ax.transAxes,
    )
    anchored.patch.set_boxstyle("round,pad=0.20")
    anchored.patch.set_facecolor("white")
    anchored.patch.set_edgecolor("0.7")
    anchored.patch.set_alpha(0.90)
    ax.add_artist(anchored)


def add_lag_box(ax, peak_corr, corr0, delta_corr):
    parts = [
        _ta("峰值相关=", FP_SIMSUN, 5.0),
        _ta(f"{peak_corr:.3f}", FP_TIMES, 5.0),
        _ta("  零时滞相关=", FP_SIMSUN, 5.0),
        _ta(f"{corr0:.3f}", FP_TIMES, 5.0),
        _ta("  提升=", FP_SIMSUN, 5.0),
        _ta(f"{delta_corr:.3f}", FP_TIMES, 5.0),
    ]
    hbox = HPacker(children=parts, align="center", pad=0, sep=0)
    anchored = AnchoredOffsetbox(
        loc="upper right",
        child=hbox,
        frameon=True,
        pad=0.18,
        borderpad=0.30,
        bbox_to_anchor=(0.98, 0.98),
        bbox_transform=ax.transAxes,
    )
    anchored.patch.set_boxstyle("round,pad=0.18")
    anchored.patch.set_facecolor("white")
    anchored.patch.set_edgecolor("0.7")
    anchored.patch.set_alpha(0.88)
    ax.add_artist(anchored)


def add_two_xticklabels(ax, left_text, right_text, y=-0.08, fs_left=6.5, fs_right=6.5):
    ax.set_xticklabels(["", ""])
    trans = ax.get_xaxis_transform()
    ax.text(0, y, left_text, transform=trans, ha="center", va="top", fontproperties=FP_SIMSUN, fontsize=fs_left)
    ax.text(1, y, right_text, transform=trans, ha="center", va="top", fontproperties=FP_SIMSUN, fontsize=fs_right)


def add_xtick_offsetbox(ax, x, y, box):
    """
    在指定 x 轴刻度位置放置可混排字体的标签。
    x 使用数据坐标，y 使用坐标轴相对坐标，效果与 ax.get_xaxis_transform() 下的 ax.text 基本一致。
    """
    ab = AnnotationBbox(
        box,
        (x, y),
        xycoords=ax.get_xaxis_transform(),
        box_alignment=(0.5, 1.0),
        frameon=False,
        pad=0.0,
        annotation_clip=False,
    )
    ax.add_artist(ab)
    return ab


def add_two_xticklabels_ais_one_line(ax, right_text, y=-0.08, fs_left=6.5, fs_right=6.5):
    """
    左侧标签为“AIS动态信息”：AIS 使用 Times New Roman，中文使用宋体。
    """
    ax.set_xticklabels(["", ""])
    trans = ax.get_xaxis_transform()

    left_box = HPacker(
        children=[
            _ta("AIS", FP_TIMES, fs_left),
            _ta("位置报告动态信息", FP_SIMSUN, fs_left),
        ],
        align="center",
        pad=0,
        sep=0,
    )
    add_xtick_offsetbox(ax, 0, y, left_box)
    ax.text(1, y, right_text, transform=trans, ha="center", va="top", fontproperties=FP_SIMSUN, fontsize=fs_right)


def add_two_xticklabels_ais_two_lines(ax, right_text, y=-0.08, fs_left=6.5, fs_right=6.5):
    """
    左侧标签为两行“AIS位置报告 / 动态信息”：AIS 使用 Times New Roman，中文使用宋体。
    """
    ax.set_xticklabels(["", ""])
    trans = ax.get_xaxis_transform()

    line1 = HPacker(
        children=[
            _ta("AIS", FP_TIMES, fs_left),
            _ta("位置报告", FP_SIMSUN, fs_left),
        ],
        align="center",
        pad=0,
        sep=0,
    )
    line2 = _ta("动态信息", FP_SIMSUN, fs_left)
    left_box = VPacker(children=[line1, line2], align="center", pad=0, sep=0)

    add_xtick_offsetbox(ax, 0, y, left_box)
    ax.text(1, y, right_text, transform=trans, ha="center", va="top", fontproperties=FP_SIMSUN, fontsize=fs_right)


def save_fig(fig, stem):
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.tight_layout(pad=0.6)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)
    return png, pdf


def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def wrap360(deg):
    return deg % 360.0


def quantiles(arr, probs=(0.05, 0.50, 0.90), min_n=20):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < min_n:
        return tuple([np.nan] * len(probs))
    return tuple(np.quantile(arr, probs).tolist())

# =========================
# 3. 参数设置
# =========================
V_MAX = 1.5
TAU_ABS = 0.2
TAU_REL = 0.30
MAX_XCORR_SPAN_S = 2 * 3600
XCORR_STEP_S = 5.0
XCORR_MAX_LAG_S = 600.0
USE_DIFF_DETREND = True
BOUNDARY_EPS = 1e-9
V_THR = 0.05
W_THR = 0.003
A_LIM_Q = 0.02
V_STAR = 0.113
BAND = 0.03
GENERATE_TABLES = True

# =========================
# 4. 加载数据
# =========================
cols = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
usecols = [c for c in ["base_date_time", "latitude", "longitude", "sog", "cog"] if c in cols]
df = pd.read_csv(CSV_PATH, usecols=usecols)

df["base_date_time"] = pd.to_datetime(df["base_date_time"], errors="coerce", utc=True)
df = df.dropna(subset=["base_date_time", "latitude", "longitude"]).sort_values("base_date_time").drop_duplicates("base_date_time").reset_index(drop=True)

for col in ["latitude", "longitude", "sog", "cog"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").interpolate(method="linear").ffill().bfill()

# 时间与经纬度
t = df["base_date_time"].astype("int64").to_numpy() / 1e9
dt = np.diff(t, prepend=np.nan).astype(np.float64)
lat = df["latitude"].to_numpy(np.float64)
lon = df["longitude"].to_numpy(np.float64)

# =========================
# 5. 相邻位置点计算结果计算
# =========================
R_EARTH = 6371000.0
latr = np.deg2rad(lat)
lonr = np.deg2rad(lon)

lat1 = np.roll(latr, 1)
lon1 = np.roll(lonr, 1)
dlat = latr - lat1
dlon = lonr - lon1
dlat[0] = np.nan
dlon[0] = np.nan

latm = 0.5 * (latr + lat1)
dx = (R_EARTH * dlon * np.cos(latm)).astype(np.float64)
dy = (R_EARTH * dlat).astype(np.float64)
disp = np.sqrt(dx * dx + dy * dy).astype(np.float64)
disp[0] = np.nan

ok = np.isfinite(dt) & (dt > 0) & np.isfinite(disp) & (disp >= 1.0)

v_pos = np.full(len(df), np.nan, np.float64)
v_pos[ok] = disp[ok] / dt[ok]

psi_pos = ((np.rad2deg(np.arctan2(dx, dy)) + 360.0) % 360.0).astype(np.float64)
psi_pos[0] = np.nan
psi_u = np.rad2deg(np.unwrap(np.deg2rad(np.nan_to_num(psi_pos, nan=0.0)))).astype(np.float64)
psi_u[~np.isfinite(psi_pos)] = np.nan

omega_pos = np.full(len(df), np.nan, np.float64)
omega_pos[1:] = np.deg2rad(psi_u[1:] - psi_u[:-1]) / dt[1:]
omega_pos[~np.isfinite(omega_pos)] = np.nan

a_pos = np.full(len(df), np.nan, np.float64)
a_pos[1:] = (v_pos[1:] - v_pos[:-1]) / dt[1:]
a_pos[~np.isfinite(a_pos)] = np.nan

# =========================
# 6. AIS 位置报告动态字段运动学参数计算
# =========================
sog_ms = df["sog"].to_numpy(np.float64) * 0.514444 if "sog" in df.columns else None

a_sog = None
if sog_ms is not None:
    a_sog = np.full(len(df), np.nan, np.float64)
    a_sog[1:] = (sog_ms[1:] - sog_ms[:-1]) / dt[1:]
    a_sog[~np.isfinite(a_sog)] = np.nan

omega_cog = None
if "cog" in df.columns:
    cog = wrap360(df["cog"].to_numpy(np.float64))
    dcog = np.full(len(df), np.nan, np.float64)
    dcog[1:] = wrap180(cog[1:] - cog[:-1])
    omega_cog = np.full(len(df), np.nan, np.float64)
    omega_cog[ok] = np.deg2rad(dcog[ok]) / dt[ok]
    omega_cog[~np.isfinite(omega_cog)] = np.nan

# ============================================================
# 图 2：AIS 位置报告中的对地航速与位置差分对地航速一致性散点图
# ============================================================
mae = rmse = mismatch_abs = mismatch_rel = np.nan

if sog_ms is not None:
    m0 = ok & np.isfinite(v_pos) & np.isfinite(sog_ms) & (v_pos >= 0) & (sog_ms >= 0)
    m = m0 & (v_pos <= V_MAX) & (sog_ms <= V_MAX)

    dv = v_pos[m] - sog_ms[m]
    mae = float(np.mean(np.abs(dv)))
    rmse = float(np.sqrt(np.mean(dv * dv)))
    mismatch_abs = float(np.mean(np.abs(dv) > TAU_ABS))
    mismatch_rel = float(np.mean(np.abs(dv) > (TAU_REL * np.maximum(sog_ms[m], 1e-3))))

    idx = np.where(m)[0]
    if idx.size > 3500:
        idx = np.random.default_rng(0).choice(idx, size=3500, replace=False)

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.scatter(sog_ms[idx], v_pos[idx], s=5, alpha=0.45, linewidths=0, color="#1f77b4")
    ax.plot([0, V_MAX], [0, V_MAX], linestyle="--", linewidth=1.0, color="#d62728")
    ax.set_xlim(0, V_MAX)
    ax.set_ylim(0, V_MAX)
    ax.set_xticks(np.linspace(0, V_MAX, 4))
    ax.set_yticks(np.linspace(0, V_MAX, 4))

    add_packed_xlabel(
        ax,
        [
            ("AIS", FP_TIMES, 7.5),
            ("位置报告中的对地航速/", FP_SIMSUN, 7.5),
            ("m·s⁻¹", FP_TIMES, 7.5),
        ],
        y=-0.18,
    )
    add_mixed_ylabel_plain(ax, "位置差分对地航速/", "m·s⁻¹", x=-0.11, fontsize=7.5)
    ax.grid(True, linestyle=":", alpha=0.5, linewidth=0.5)
    add_stats_box_speed(ax, mae, rmse, mismatch_abs, mismatch_rel)
    set_ticklabel_font(ax, x_font=FP_TIMES, y_font=FP_TIMES)
    beautify_axes(ax, labelpad=5, tickpad=2)
    save_fig(fig, "Fig2_speed_consistency")

# ============================================================
# 图 3：对地航向变化率分布对比
# 说明：
# 原来的粗竖条 + 多个标注框视觉负担过重。
# 这里改为“P05~P90 细线 + P50 菱形点”的分位数图，精确数值放入表格。
# 字体、字号、dpi、线宽体系保持原期刊格式。
# ============================================================
omega_stats = {}
if omega_cog is not None:
    omega_mask_ais = ok & np.isfinite(omega_cog) & np.isfinite(sog_ms) & (sog_ms > V_THR)
    omega_mask_pos = ok & np.isfinite(omega_pos) & np.isfinite(v_pos) & (v_pos > V_THR)

    abs_omega_ais = np.abs(omega_cog[omega_mask_ais])
    abs_omega_pos = np.abs(omega_pos[omega_mask_pos])

    # 去掉极小值和离群极大值，避免 log 坐标被无意义点拉伸。
    abs_omega_ais = abs_omega_ais[(abs_omega_ais > 1e-5) & (abs_omega_ais < 1.0)]
    abs_omega_pos = abs_omega_pos[(abs_omega_pos > 1e-5) & (abs_omega_pos < 1.0)]

    a05, a50, a90 = quantiles(abs_omega_ais)
    p05, p50, p90 = quantiles(abs_omega_pos)

    omega_stats = {
        "AIS位置报告中的动态信息_P05": a05,
        "AIS位置报告中的动态信息_P50": a50,
        "AIS位置报告中的动态信息_P90": a90,
        "相邻位置点计算结果_P05": p05,
        "相邻位置点计算结果_P50": p50,
        "相邻位置点计算结果_P90": p90,
        "AIS样本数": int(abs_omega_ais.size),
        "位置计算样本数": int(abs_omega_pos.size),
    }

    fig, ax = plt.subplots(figsize=(3, 2))

    def draw_quantile_line(x, q05, q50, q90, color):
        # P05~P90 范围线
        ax.vlines(x, q05, q90, color=color, linewidth=2.0, zorder=2)
        # 上下端短横线
        ax.hlines([q05, q90], x - 0.055, x + 0.055, color=color, linewidth=1.0, zorder=2)
        # P50 中位数点
        ax.scatter([x], [q50], s=28, marker="D", color=color, zorder=3)

    draw_quantile_line(0, a05, a50, a90, "#1f77b4")
    draw_quantile_line(1, p05, p50, p90, "#ff7f0e")

    ax.set_yscale("log")
    ax.set_xlim(-0.35, 1.35)
    # 留出上方空间，避免 P90 接近图框。
    y_min = max(1e-5, min(a05, p05) * 0.65)
    y_max = max(a90, p90) * 2.2
    ax.set_ylim(y_min, y_max)

    ax.set_xticks([0, 1])
    add_two_xticklabels_ais_one_line(ax, "相邻位置点计算结果", y=-0.08, fs_left=6.5, fs_right=6.5)

    ax.set_yticks([1e-4, 1e-3, 1e-2])
    ax.set_yticklabels(["10⁻⁴", "10⁻³", "10⁻²"], fontproperties=FP_TIMES, fontsize=6.5)
    add_mixed_ylabel_plain(ax, "对地航向变化率绝对值/", "rad·s⁻¹", x=-0.12, fontsize=7.5)

    ax.grid(True, linestyle=":", alpha=0.5, linewidth=0.5, which="both", axis="y")

    # 只标注 P50，P05/P90 精确值放入表格，避免图面变成贴满创可贴的事故现场。
    bbox_kw = dict(boxstyle="round,pad=0.10", fc="white", ec="0.7", alpha=0.90)
    ax.text(0.07, a50, f"P50={a50:.4f}", ha="left", va="center", bbox=bbox_kw, fontsize=5.0, fontproperties=FP_TIMES)
    ax.text(1.07, p50, f"P50={p50:.4f}", ha="left", va="center", bbox=bbox_kw, fontsize=5.0, fontproperties=FP_TIMES)

    set_ticklabel_font(ax, x_font=FP_TIMES, y_font=FP_TIMES)
    beautify_axes(ax, labelpad=5, tickpad=2)
    save_fig(fig, "Fig3_omega_distribution_compare")

# ============================================================
# 图 4：互相关时滞分析
# ============================================================
lag_pk = np.nan
peak_corr = np.nan
corr0 = np.nan
delta_corr = np.nan
boundary_hit = False

if sog_ms is not None:
    m = ok & np.isfinite(v_pos) & np.isfinite(sog_ms)
    ts_all = t[m].astype(np.float64)
    s_all = sog_ms[m].astype(np.float64)
    v_all = v_pos[m].astype(np.float64)

    if ts_all.size >= 400:
        t_mid = 0.5 * (ts_all.min() + ts_all.max())
        sel = (ts_all >= t_mid - MAX_XCORR_SPAN_S / 2) & (ts_all <= t_mid + MAX_XCORR_SPAN_S / 2)

        if np.sum(sel) < 250:
            mid_idx = ts_all.size // 2
            lo = max(0, mid_idx - 2500)
            hi = min(ts_all.size, mid_idx + 2500)
            ts = ts_all[lo:hi]
            s = s_all[lo:hi]
            v = v_all[lo:hi]
        else:
            ts = ts_all[sel]
            s = s_all[sel]
            v = v_all[sel]

        grid = np.arange(ts.min(), ts.max(), XCORR_STEP_S)
        if grid.size >= 250:
            s_i = np.interp(grid, ts, s).astype(np.float64)
            v_i = np.interp(grid, ts, v).astype(np.float64)

            if USE_DIFF_DETREND:
                s_i = np.diff(s_i)
                v_i = np.diff(v_i)

            s_i = s_i - np.mean(s_i)
            v_i = v_i - np.mean(v_i)

            max_k = int(XCORR_MAX_LAG_S / XCORR_STEP_S)
            corr_full = np.correlate(s_i, v_i, mode="full")
            mid = corr_full.size // 2
            seg = corr_full[mid - max_k:mid + max_k + 1]
            denom = (np.std(s_i) * np.std(v_i) * len(v_i) + 1e-12)
            corr = seg / denom
            lags = np.arange(-max_k, max_k + 1) * XCORR_STEP_S

            kpk = int(np.argmax(corr))
            lag_pk = float(lags[kpk])
            peak_corr = float(corr[kpk])
            k0 = int(np.where(lags == 0.0)[0][0])
            corr0 = float(corr[k0])
            delta_corr = float(peak_corr - corr0)

            if abs(lag_pk - lags[0]) < BOUNDARY_EPS or abs(lag_pk - lags[-1]) < BOUNDARY_EPS:
                boundary_hit = True

            fig, ax = plt.subplots(figsize=(3, 2))
            ax.plot(lags, corr, linewidth=1.2, color="#1f77b4")
            ax.axvline(0.0, linestyle=":", linewidth=1.0, color="#2ca02c")
            ax.axvline(lag_pk, linestyle="--", linewidth=1.0, color="#d62728")
            ax.set_xticks(np.linspace(lags.min(), lags.max(), 5))

            add_packed_xlabel(ax, [("时滞/", FP_SIMSUN, 7.5), ("s", FP_TIMES, 7.5)], y=-0.18)
            ax.set_ylabel("相关系数", fontproperties=FP_SIMSUN, fontsize=7.5)
            ax.grid(True, linestyle=":", alpha=0.5, linewidth=0.5)
            add_lag_box(ax, peak_corr, corr0, delta_corr)
            set_ticklabel_font(ax, x_font=FP_TIMES, y_font=FP_TIMES)
            beautify_axes(ax, labelpad=5, tickpad=2)
            save_fig(fig, "Fig4_lag_xcorr")

# ============================================================
# 图 5：回转半径分布对比
# ============================================================
turn_pos = ok & np.isfinite(v_pos) & np.isfinite(omega_pos) & (v_pos > V_THR) & (np.abs(omega_pos) > W_THR)
R_pos = v_pos[turn_pos] / np.abs(omega_pos[turn_pos])
R_pos = R_pos[np.isfinite(R_pos)]
R_pos = R_pos[(R_pos > 0) & (R_pos < 5000)]
Rpos05, Rpos50, Rpos90 = quantiles(R_pos, min_n=20)

R_ais = np.array([], dtype=np.float64)
turn_ais_n = np.nan
if (sog_ms is not None) and (omega_cog is not None):
    turn_ais = ok & np.isfinite(sog_ms) & np.isfinite(omega_cog) & (sog_ms > V_THR) & (np.abs(omega_cog) > W_THR)
    turn_ais_n = int(np.sum(turn_ais))
    R_ais = sog_ms[turn_ais] / np.abs(omega_cog[turn_ais])
    R_ais = R_ais[np.isfinite(R_ais)]
    R_ais = R_ais[(R_ais > 0) & (R_ais < 5000)]
Rais05, Rais50, Rais90 = quantiles(R_ais, min_n=20)

fig, ax = plt.subplots(figsize=(3, 2))


def draw_quantile_radius(x, p05, p50, p90, color):
    ax.plot([x, x], [p05, p90], linewidth=5, solid_capstyle="butt", zorder=2, color=color)
    ax.scatter([x], [p50], s=50, marker="D", zorder=3, color=color)


draw_quantile_radius(0, Rais05, Rais50, Rais90, "#1f77b4")
draw_quantile_radius(1, Rpos05, Rpos50, Rpos90, "#ff7f0e")

ax.set_yscale("log")
ax.set_xlim(-0.25, 1.25)
ax.set_xticks([0, 1])
add_two_xticklabels_ais_one_line(ax, "相邻位置点计算结果", y=-0.08, fs_left=6.5, fs_right=6.5)
ax.set_yticks([10, 100, 1000])
ax.set_yticklabels(["10", "100", "1000"], fontproperties=FP_TIMES, fontsize=6.5)
add_mixed_ylabel_plain(ax, "回转半径/", "m", x=-0.11, fontsize=7.5)
ax.grid(True, linestyle=":", alpha=0.5, linewidth=0.5, which="both", axis="y")

bbox_kw = dict(boxstyle="round,pad=0.10", fc="white", ec="0.7", alpha=0.90)
for x0, vals, ha_x in [(0, (Rais05, Rais50, Rais90), -0.22), (1, (Rpos05, Rpos50, Rpos90), 0.92)]:
    q05, q50, q90 = vals
    ax.text(ha_x, q90, f"P90={q90:.1f} m", ha="left", va="center", bbox=bbox_kw, fontsize=5.0, fontproperties=FP_TIMES)
    ax.text(ha_x, q50, f"P50={q50:.1f} m", ha="left", va="center", bbox=bbox_kw, fontsize=5.0, fontproperties=FP_TIMES)
    ax.text(ha_x, q05, f"P05={q05:.1f} m", ha="left", va="center", bbox=bbox_kw, fontsize=5.0, fontproperties=FP_TIMES)

set_ticklabel_font(ax, x_font=FP_TIMES, y_font=FP_TIMES)
beautify_axes(ax, labelpad=5, tickpad=2)
save_fig(fig, "Fig5_turning_radius_quantiles")

# ============================================================
# 表格输出：一致性、航向变化率、回转半径、减速距离
# ============================================================
if GENERATE_TABLES:
    table_consistency = pd.DataFrame([
        {"类别": "速度误差", "指标": "MAE /（m·s⁻¹）", "数值": mae},
        {"类别": "速度误差", "指标": "RMSE /（m·s⁻¹）", "数值": rmse},
        {"类别": "失配比例", "指标": f"|Δv|>{TAU_ABS:.1f} m·s⁻¹ / %", "数值": mismatch_abs * 100},
        {"类别": "失配比例", "指标": f"|Δv|>{TAU_REL:.1f}×SOG / %", "数值": mismatch_rel * 100},
        {"类别": "滞后检验", "指标": "峰值时滞 / s", "数值": lag_pk},
        {"类别": "滞后检验", "指标": "峰值相关系数", "数值": peak_corr},
        {"类别": "滞后检验", "指标": "零时滞相关系数", "数值": corr0},
        {"类别": "滞后检验", "指标": "峰值相关增量", "数值": delta_corr},
        {"类别": "滞后检验", "指标": "峰值是否位于搜索边界", "数值": int(boundary_hit)},
    ])
    table_consistency.to_csv(OUT_DIR / "Table1_speed_consistency_summary.csv", index=False, encoding="utf-8-sig")

    table_omega = pd.DataFrame([
        {"方法": "AIS位置报告中的动态信息", "样本数": omega_stats.get("AIS样本数", np.nan), "P05/(rad·s⁻¹)": omega_stats.get("AIS位置报告中的动态信息_P05", np.nan), "P50/(rad·s⁻¹)": omega_stats.get("AIS位置报告中的动态信息_P50", np.nan), "P90/(rad·s⁻¹)": omega_stats.get("AIS位置报告中的动态信息_P90", np.nan)},
        {"方法": "相邻位置点计算结果", "样本数": omega_stats.get("位置差分样本数", np.nan), "P05/(rad·s⁻¹)": omega_stats.get("相邻位置点计算结果_P05", np.nan), "P50/(rad·s⁻¹)": omega_stats.get("相邻位置点计算结果_P50", np.nan), "P90/(rad·s⁻¹)": omega_stats.get("相邻位置点计算结果_P90", np.nan)},
    ])
    table_omega.to_csv(OUT_DIR / "Table2_omega_quantiles.csv", index=False, encoding="utf-8-sig")

    table_radius = pd.DataFrame([
        {"方法": "AIS位置报告中的动态信息", "样本数": int(turn_ais_n) if np.isfinite(turn_ais_n) else np.nan, "P05/m": Rais05, "P50/m": Rais50, "P90/m": Rais90},
        {"方法": "相邻位置点计算结果", "样本数": int(np.sum(turn_pos)), "P05/m": Rpos05, "P50/m": Rpos50, "P90/m": Rpos90},
    ])
    table_radius.to_csv(OUT_DIR / "Table3_turning_radius_quantiles.csv", index=False, encoding="utf-8-sig")

    def robust_decel_limit(a_series):
        a = a_series[ok & np.isfinite(a_series)].astype(np.float64)
        neg = a[a < 0]
        if neg.size < 50:
            return np.nan
        return float(np.quantile(neg, A_LIM_Q))

    def braking_row(v_series, a_series, name):
        a_lim = robust_decel_limit(a_series)
        m = ok & np.isfinite(v_series) & (v_series >= 0)
        v = v_series[m].astype(np.float64)
        sel = (v >= max(0, V_STAR - BAND)) & (v <= V_STAR + BAND)
        if np.sum(sel) < 30:
            k = min(max(200, 30), v.size)
            idx = np.argsort(np.abs(v - V_STAR))[:k]
            v_band = v[idx]
        else:
            v_band = v[sel]

        if not np.isfinite(a_lim) or a_lim >= 0:
            d = np.full_like(v_band, np.nan, dtype=np.float64)
        else:
            d = (v_band ** 2) / (2.0 * abs(a_lim))

        return {
            "方法": name,
            f"减速度Q{A_LIM_Q:.2f}分位值/(m·s⁻²)": a_lim,
            "速度带/(m·s⁻¹)": f"{V_STAR:.3f}±{BAND:.3f}",
            "样本数": int(v_band.size),
            "P50/m": float(np.nanquantile(d, 0.50)),
            "P90/m": float(np.nanquantile(d, 0.90)),
        }

    braking_rows = [braking_row(v_pos, a_pos, "相邻位置点计算结果")]
    if (sog_ms is not None) and (a_sog is not None):
        braking_rows.append(braking_row(sog_ms, a_sog, "AIS位置报告中的动态信息"))
    table_braking = pd.DataFrame(braking_rows)
    table_braking.to_csv(OUT_DIR / "Table4_deceleration_distance_compare.csv", index=False, encoding="utf-8-sig")

    print("\n表格已生成：")
    print("  Table1_speed_consistency_summary.csv")
    print("  Table2_omega_quantiles.csv")
    print("  Table3_turning_radius_quantiles.csv")
    print("  Table4_deceleration_distance_compare.csv")

print("\n处理完成。输出文件夹:", OUT_DIR)
print("=" * 60)
print("字体文件使用确认：")
print("  simsun.ttc   -> 中文标签（宋体）")
print("  simsunb.ttf  -> 中文粗体（备用）")
print("  times.ttf    -> 西文/数字（Times New Roman）")
print("  timesbd.ttf  -> 西文粗体（备用）")
print("  timesi.ttf   -> 西文斜体（备用）")
print("=" * 60)
