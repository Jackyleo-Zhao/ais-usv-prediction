# -*- coding: utf-8 -*-
"""
第一章最终版绘图与数据统计代码

修改目标：
1. 不再使用经纬度直接绘制轨迹，统一转换至局部 ENU 坐标系；
2. 数据集 A 与数据集 B 分别单独成图，不合并为子图；
3. 删除图内标题，不再显示“（a）数据集 A /（b）数据集 B”；
4. 纵轴标签左移，避免与 y 轴刻度数字过近；
5. 缩小图例占用空间；
6. 保留原期刊绘图格式：宋体、Times New Roman、7.5 号字、600 dpi、线宽等不变；
7. 输出简洁版数据集概况表，正文仅保留必要信息。

运行前请确保同目录下存在：
- ais-USV_filled.csv
- UsvState_turning_experiment.xlsx
- simsun.ttc
- times.ttf

若文件名带括号，如 ais-USV_filled(7).csv，本代码也会自动尝试匹配。
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator

# =========================
# 0. 文件路径设置
# =========================
AIS_CANDIDATES = "ais-USV_filled.csv"


B_CANDIDATES = "UsvState_turning_experiment.xlsx"


OUT_DIR = Path("fig_task1_final")
OUT_DIR.mkdir(exist_ok=True)


def resolve_file(candidates):
    """依次查找候选路径，避免本地文件名和上传文件名不一致导致报错。"""
    for item in candidates:
        p = Path(item)
        if p.exists():
            return p
    raise FileNotFoundError("未找到文件，请检查路径：\n" + "\n".join(candidates))


AIS_FILE = resolve_file(AIS_CANDIDATES)
B_FILE = resolve_file(B_CANDIDATES)

# =========================
# 1. 加载自定义字体文件
# =========================
font_paths = [
    "simsun.ttc", "simsunb.ttf", "SimsunExtG.ttf",
    "times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"
]
for fp in font_paths:
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)

zh_font = FontProperties(fname="simsun.ttc") if Path("simsun.ttc").exists() else FontProperties(family="SimSun")
en_font = FontProperties(fname="times.ttf") if Path("times.ttf").exists() else FontProperties(family="Times New Roman")

# =========================
# 2. 全局绘图设置：保持原格式
# =========================
plt.rcParams["font.family"] = en_font.get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = en_font.get_name()
plt.rcParams["mathtext.it"] = f"{en_font.get_name()}:italic"
plt.rcParams["mathtext.bf"] = f"{en_font.get_name()}:bold"
plt.rcParams["mathtext.default"] = "rm"

# =========================
# 3. 通用函数
# =========================
def apply_pub_style(ax, xlabel=None, ylabel=None):
    """中文标签用宋体，刻度数字用 Times New Roman。字号保持 7.5。"""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=7.5, fontproperties=zh_font)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=7.5, fontproperties=zh_font)

    for tick in ax.get_xticklabels():
        tick.set_fontproperties(en_font)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(en_font)

    ax.xaxis.get_offset_text().set_fontproperties(en_font)
    ax.yaxis.get_offset_text().set_fontproperties(en_font)

    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontproperties(zh_font)


def add_mixed_xlabel(ax, cn_text, en_text, y=-0.18, fontsize=7.5):
    """横坐标：中文宋体 + 英文/单位 Times New Roman。"""
    ax.set_xlabel("")
    ax.text(
        0.5, y, cn_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontproperties=zh_font,
        fontsize=fontsize,
    )
    ax.text(
        0.5, y, en_text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontproperties=en_font,
        fontsize=fontsize,
    )


def add_mixed_ylabel(ax, cn_text, en_text, x=-0.20, fontsize=7.5):
    """纵坐标：中文宋体 + 英文/单位 Times New Roman。x 更小表示更靠左。"""
    ax.set_ylabel("")
    ax.text(
        x, 0.5, cn_text,
        transform=ax.transAxes,
        ha="center", va="top",
        rotation=90,
        fontproperties=zh_font,
        fontsize=fontsize,
    )
    ax.text(
        x, 0.5, en_text,
        transform=ax.transAxes,
        ha="center", va="bottom",
        rotation=90,
        fontproperties=en_font,
        fontsize=fontsize,
    )


def wrap_360(deg):
    """将航向角统一限制到 0°~360°。"""
    return np.mod(deg, 360.0)


def lonlat_to_local_enu(lon, lat, lon0=None, lat0=None):
    """经纬度转局部 ENU 平面坐标。默认以本数据集首个有效点为局部原点。"""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    if lon0 is None:
        lon0 = lon[0]
    if lat0 is None:
        lat0 = lat[0]

    R = 6371000.0
    east = R * np.deg2rad(lon - lon0) * np.cos(np.deg2rad(lat0))
    north = R * np.deg2rad(lat - lat0)
    return east, north


def displacement_stats(east, north):
    """相邻位置点位移统计。"""
    d = np.sqrt(np.diff(east) ** 2 + np.diff(north) ** 2)
    return {
        "相邻位移中位数/m": np.nanmedian(d),
        "相邻位移均值/m": np.nanmean(d),
        "相邻位移95%分位值/m": np.nanpercentile(d, 95),
        "零位移比例/%": np.mean(d < 1e-9) * 100,
        "相邻步数": len(d),
    }

# =========================
# 4. 数据读取与预处理
# =========================
def load_dataset_a(path):
    """数据集 A：低频 AIS 历史位置报告数据。"""
    df = pd.read_csv(path)
    df["base_date_time"] = pd.to_datetime(df["base_date_time"], errors="coerce")

    for col in ["longitude", "latitude", "sog", "cog"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["base_date_time", "longitude", "latitude"])
    df = df.sort_values("base_date_time")
    df = df.drop_duplicates(subset=["base_date_time"], keep="first")

    for col in ["longitude", "latitude", "sog", "cog"]:
        df[col] = df[col].interpolate(method="linear").ffill().bfill()

    # 第一章保留 AIS 原始 SOG 单位 kn；后续计算统一换算为 m/s。
    df["sog_kn"] = df["sog"]
    df["sog_ms"] = df["sog"] * 0.514444
    df["cog"] = wrap_360(df["cog"])

    df["dt_s"] = df["base_date_time"].diff().dt.total_seconds()
    df["east_m"], df["north_m"] = lonlat_to_local_enu(
        df["longitude"].values,
        df["latitude"].values,
    )
    return df


def load_dataset_b(path):
    """数据集 B：USV 转向实验状态数据。按秒聚合，避免同一秒多条记录影响图形和统计。"""
    raw = pd.read_excel(path, sheet_name=0)

    useful_cols = ["经度", "纬度", "航速", "航向", "舵角", "期望舵角", "油门控制量", "方向控制量", "控制模式"]
    for col in useful_cols:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw["_time"] = pd.to_datetime("2025-01-01 " + raw["时间"].astype(str), errors="coerce")
    raw = raw.dropna(subset=["_time", "经度", "纬度"])

    numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
    df = raw.groupby("_time", as_index=False)[numeric_cols].mean()
    df = df.sort_values("_time")

    for col in ["经度", "纬度", "航速", "航向", "舵角"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear").ffill().bfill()

    if "航向" in df.columns:
        df["航向"] = wrap_360(df["航向"])

    df["elapsed_s"] = (df["_time"] - df["_time"].iloc[0]).dt.total_seconds()
    df["dt_s"] = df["_time"].diff().dt.total_seconds()
    df["east_m"], df["north_m"] = lonlat_to_local_enu(
        df["经度"].values,
        df["纬度"].values,
    )
    return raw, df


A = load_dataset_a(AIS_FILE)
B_raw, B = load_dataset_b(B_FILE)

# =========================
# 5. 简洁统计表输出
# =========================
A_disp = displacement_stats(A["east_m"].values, A["north_m"].values)
B_disp = displacement_stats(B["east_m"].values, B["north_m"].values)

summary_short = pd.DataFrame([
    {
        "数据集": "A",
        "类型": "低频 AIS 轨迹",
        "规模": f"{len(A)} 条",
        "采样间隔": f"中位值 {A['dt_s'].dropna().median():.0f} s",
        "用途": "主实验",
    },
    {
        "数据集": "B",
        "类型": "转向实验状态数据",
        "规模": f"{len(B)} 点",
        "采样间隔": f"{B['dt_s'].dropna().median():.0f} s",
        "用途": "机动补充分析",
    },
])

summary_full = pd.DataFrame([
    {
        "数据集": "A",
        "原始样本数": len(A),
        "时间跨度/min": (A["base_date_time"].iloc[-1] - A["base_date_time"].iloc[0]).total_seconds() / 60,
        "采样间隔中位数/s": A["dt_s"].dropna().median(),
        "采样间隔95%分位值/s": A["dt_s"].dropna().quantile(0.95),
        "采样间隔最大值/s": A["dt_s"].dropna().max(),
        "SOG中位数/kn": A["sog_kn"].median(),
        "SOG95%分位值/kn": A["sog_kn"].quantile(0.95),
        "COG范围/(°)": f"{A['cog'].min():.1f}~{A['cog'].max():.1f}",
        **A_disp,
    },
    {
        "数据集": "B",
        "原始样本数": len(B_raw),
        "按秒聚合后样本数": len(B),
        "时间跨度/min": (B["_time"].iloc[-1] - B["_time"].iloc[0]).total_seconds() / 60,
        "采样间隔中位数/s": B["dt_s"].dropna().median(),
        "采样间隔95%分位值/s": B["dt_s"].dropna().quantile(0.95),
        "采样间隔最大值/s": B["dt_s"].dropna().max(),
        "航向范围/(°)": f"{B['航向'].min():.1f}~{B['航向'].max():.1f}" if "航向" in B.columns else "-",
        "舵角范围/(°)": f"{B['舵角'].min():.1f}~{B['舵角'].max():.1f}" if "舵角" in B.columns else "-",
        **B_disp,
    },
])

summary_short.to_csv(OUT_DIR / "table1_dataset_overview_short.csv", index=False, encoding="utf-8-sig")
summary_full.to_csv(OUT_DIR / "dataset_statistics_full_for_text.csv", index=False, encoding="utf-8-sig")

print("\n[INFO] 使用文件：")
print(f"  数据集 A: {AIS_FILE}")
print(f"  数据集 B: {B_FILE}")
print("\n[INFO] 表 1：数据集概况（正文使用）：")
print(summary_short.to_string(index=False))
print("\n[INFO] 详细统计（供正文取数，不建议整表放入论文）：")
print(summary_full.to_string(index=False))

# =========================
# 6. 单图绘制函数：无标题、图例缩紧、纵轴标签左移
# =========================
def plot_trajectory_single(ax, df, legend_loc="best"):
    ax.plot(
        df["east_m"], df["north_m"],
        label="轨迹", color="blue", linewidth=0.5, alpha=0.85,
    )
    ax.scatter(
        df["east_m"].iloc[0], df["north_m"].iloc[0],
        color="green", label="起点", zorder=5, s=50, alpha=0.5,
    )
    ax.scatter(
        df["east_m"].iloc[-1], df["north_m"].iloc[-1],
        color="red", label="终点", zorder=5, s=50, alpha=0.5,
    )

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", direction="in", labelsize=7.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_aspect("equal", adjustable="datalim")

    leg = ax.legend(
        loc=legend_loc,
        fontsize=7.5,
        frameon=True,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.8,
        handletextpad=0.45,
        markerscale=0.80,
    )
    for txt in leg.get_texts():
        txt.set_fontproperties(zh_font)

    add_mixed_xlabel(ax, "东向位置/", "m", y=-0.18, fontsize=7.5)
    add_mixed_ylabel(ax, "北向位置/", "m", x=-0.20, fontsize=7.5)
    apply_pub_style(ax)

# =========================
# 7. 图 1：数据集 A 轨迹，单独成图，无图内标题
# =========================
fig, ax = plt.subplots(figsize=(3, 2), dpi=600)
plot_trajectory_single(ax, A, legend_loc="lower right")
fig.subplots_adjust(left=0.23, bottom=0.25, right=0.98, top=0.98)
fig.savefig(OUT_DIR / "fig1_dataset_A_trajectory.tif", dpi=600)
fig.savefig(OUT_DIR / "fig1_dataset_A_trajectory.png", dpi=600)
plt.show()

# =========================
# 8. 图 2：数据集 B 轨迹，单独成图，无图内标题
# =========================
fig, ax = plt.subplots(figsize=(3, 2), dpi=600)
plot_trajectory_single(ax, B, legend_loc="upper left")
fig.subplots_adjust(left=0.23, bottom=0.25, right=0.98, top=0.98)
fig.savefig(OUT_DIR / "fig2_dataset_B_trajectory.tif", dpi=600)
fig.savefig(OUT_DIR / "fig2_dataset_B_trajectory.png", dpi=600)
plt.show()

# =========================
# 9. 可选图：数据集 B 航向/舵角检查图
#    正文第一章不建议放；如后文写持续机动分析，可打开。
# =========================
PLOT_B_TURNING_CHECK = False

if PLOT_B_TURNING_CHECK:
    fig, ax1 = plt.subplots(figsize=(3, 2), dpi=600)
    ax1.plot(
        B["elapsed_s"] / 60,
        B["航向"],
        color="lightcoral",
        linewidth=0.5,
        alpha=0.8,
        label="航向",
    )
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.tick_params(axis="both", direction="in", labelsize=7.5)
    apply_pub_style(ax1, xlabel="时间/min", ylabel=None)
    add_mixed_ylabel(ax1, "航向/", "°", x=-0.18, fontsize=7.5)

    ax2 = ax1.twinx()
    ax2.plot(
        B["elapsed_s"] / 60,
        B["舵角"],
        color="gray",
        linewidth=0.5,
        alpha=0.8,
        label="舵角",
    )
    ax2.tick_params(axis="y", direction="in", labelsize=7.5)
    for tick in ax2.get_yticklabels():
        tick.set_fontproperties(en_font)
    add_mixed_ylabel(ax2, "舵角/", "°", x=1.13, fontsize=7.5)

    fig.subplots_adjust(left=0.22, bottom=0.23, right=0.82, top=0.96)
    fig.savefig(OUT_DIR / "supp_dataset_B_heading_rudder_check.tif", dpi=600)
    fig.savefig(OUT_DIR / "supp_dataset_B_heading_rudder_check.png", dpi=600)
    plt.show()
