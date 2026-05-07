import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.font_manager import FontProperties
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, DrawingArea
from matplotlib.lines import Line2D

# 加载自定义字体文件 (中文宋体字体 和 英文字体 Times New Roman)
font_paths = [
    "simsun.ttc", "simsunb.ttf", "SimsunExtG.ttf",
    "times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"
]
for fp in font_paths:
    font_manager.fontManager.addfont(fp)

# 字体对象
zh_font = FontProperties(fname="simsun.ttc")   # 中文：宋体
en_font = FontProperties(fname="times.ttf")    # 西文：Times New Roman

# 全局设置：默认西文/数字 = Times New Roman
plt.rcParams["font.family"] = en_font.get_name()
plt.rcParams["axes.unicode_minus"] = False

# 数学文本尽量走 Times New Roman
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = en_font.get_name()
plt.rcParams["mathtext.it"] = f"{en_font.get_name()}:italic"
plt.rcParams["mathtext.bf"] = f"{en_font.get_name()}:bold"
plt.rcParams["mathtext.default"] = "rm"

# =========================
# 通用函数
# =========================
def apply_pub_style(ax, xlabel=None, ylabel=None):
    """中文标签用宋体，刻度数字用 Times New Roman"""
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

def add_mixed_ylabel(ax, cn_text, en_text, x=-0.11, fontsize=7.5):
    """纵坐标：中文宋体 + 英文/单位 Times New Roman"""
    ax.set_ylabel("")

    # 中文部分放下半段
    ax.text(
        x, 0.5, cn_text,
        transform=ax.transAxes,
        ha="center", va="top",
        rotation=90,
        fontproperties=zh_font,
        fontsize=fontsize
    )

    # 单位部分放上半段
    ax.text(
        x, 0.5, en_text,
        transform=ax.transAxes,
        ha="center", va="bottom",
        rotation=90,
        fontproperties=en_font,
        fontsize=fontsize
    )

def add_hist_xlabel(ax):
    """直方图横坐标：地面速率 (SOG) [节]，中英分字体"""
    ax.set_xlabel("")

    txt_cn1 = TextArea("对地航速 (", textprops=dict(fontproperties=zh_font, fontsize=7.5))
    txt_en = TextArea("SOG", textprops=dict(fontproperties=en_font, fontsize=7.5))
    txt_cn2 = TextArea(") [节]", textprops=dict(fontproperties=zh_font, fontsize=7.5))

    box = HPacker(children=[txt_cn1, txt_en, txt_cn2], align="center", pad=0, sep=0)

    anchored_box = AnchoredOffsetbox(
        loc='lower center',
        child=box,
        frameon=False,
        bbox_to_anchor=(0.5, -0.22),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
        pad=0.0
    )
    ax.add_artist(anchored_box)

def add_mean_box(ax, mean_value):
    """右上角自定义图例：均值 SOG: xx.xx 节"""
    da = DrawingArea(26, 10, 0, 0)
    line = Line2D([0, 26], [5, 5], color='red', linestyle='--', linewidth=1.5)
    da.add_artist(line)

    txt_cn1 = TextArea("均值 ", textprops=dict(fontproperties=zh_font, fontsize=7.5))
    txt_en = TextArea(f"SOG: {mean_value:.2f} ", textprops=dict(fontproperties=en_font, fontsize=7.5))
    txt_cn2 = TextArea("节", textprops=dict(fontproperties=zh_font, fontsize=7.5))

    box = HPacker(children=[da, txt_cn1, txt_en, txt_cn2], align="center", pad=0, sep=2)

    anchored_box = AnchoredOffsetbox(
        loc='upper right',
        child=box,
        frameon=True,
        pad=0.2,
        borderpad=0.35
    )

    anchored_box.patch.set_boxstyle("round,pad=0.25")
    anchored_box.patch.set_facecolor("white")
    anchored_box.patch.set_edgecolor("0.7")
    anchored_box.patch.set_alpha(0.9)

    ax.add_artist(anchored_box)

# 加载数据
file_path = "ais-USV_filled.csv"
data = pd.read_csv(file_path)

# 将 base_date_time 转换为日期时间格式
data['base_date_time'] = pd.to_datetime(data['base_date_time'])

# 计算SOG和COG的误差（标准差）以及置信区间
sog_std = data['sog'].std()
cog_std = data['cog'].std()

# 计算SOG的置信区间（95%）
sog_ci = stats.norm.interval(0.95, loc=data['sog'].mean(), scale=sog_std / np.sqrt(len(data)))

# 计算COG的置信区间（95%）
cog_ci = stats.norm.interval(0.95, loc=data['cog'].mean(), scale=cog_std / np.sqrt(len(data)))

# =========================
# 1. 绘制船舶历史轨迹 (经度 vs 纬度)，并标记开始点和停止点
# =========================
fig, ax = plt.subplots(figsize=(3, 2), dpi=600)
ax.plot(data['longitude'], data['latitude'], label="船舶轨迹", color='blue', linewidth=0.5)
ax.scatter(data['longitude'].iloc[0], data['latitude'].iloc[0], color='green', label='起点', zorder=5, s=50)
ax.scatter(data['longitude'].iloc[-1], data['latitude'].iloc[-1], color='red', label='终点', zorder=5, s=50)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', fontsize=7.5)
ax.tick_params(axis='both', direction='in', labelsize=7.5)

apply_pub_style(ax, xlabel="经度", ylabel="纬度")
plt.tight_layout()
plt.show()

# =========================
# 2. 绘制SOG图（线性）
# =========================
fig, ax = plt.subplots(figsize=(3, 2), dpi=600)
ax.plot(data['base_date_time'], data['sog'], color='lightblue', label='SOG', linewidth=0.5, alpha=0.6)
ax.grid(True, linestyle='--', alpha=0.5)
ax.tick_params(axis='both', direction='in', labelsize=7.5)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# 这里只改纵轴单位字体，其余不变
apply_pub_style(ax, xlabel="时间", ylabel=None)
add_mixed_ylabel(ax, "对地航速/", "kn", x=-0.11, fontsize=7.5)

plt.tight_layout()
plt.show()

# =========================
# 3. 绘制COG图（线性）
# =========================
fig, ax = plt.subplots(figsize=(3, 2), dpi=600)
ax.plot(data['base_date_time'], data['cog'], color='lightcoral', label='COG', linewidth=0.5, alpha=0.6)
ax.grid(True, linestyle='--', alpha=0.5)
ax.tick_params(axis='both', direction='in', labelsize=7.5)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# 这里只改纵轴单位字体，其余不变
apply_pub_style(ax, xlabel="时间", ylabel=None)
add_mixed_ylabel(ax, "对地航向/", "°", x=-0.13, fontsize=7.5)

plt.tight_layout()
plt.show()

# =========================
# 4. 绘制速度分布直方图 (SOG)，并加上KDE曲线和均值红线
# =========================
fig, ax = plt.subplots(figsize=(3, 2), dpi=600)
sns.histplot(
    data['sog'],
    kde=True,
    color='green',
    bins=30,
    stat='density',
    alpha=0.5,
    kde_kws={"bw_adjust": 1.5},
    ax=ax
)

mean_sog = data['sog'].mean()
ax.axvline(mean_sog, color='red', linestyle='--', linewidth=1.5)

ax.grid(True, linestyle='--', alpha=0.5)
ax.tick_params(axis='both', direction='in', labelsize=7.5)

apply_pub_style(ax, xlabel=None, ylabel="密度")
add_hist_xlabel(ax)
add_mean_box(ax, mean_sog)

fig.subplots_adjust(left=0.16, bottom=0.24, right=0.98, top=0.98)
plt.show()