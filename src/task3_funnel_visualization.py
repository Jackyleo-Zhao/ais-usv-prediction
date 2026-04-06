# -*- coding: utf-8 -*-
"""
终稿（中文期刊排版增强版，已修复中文标题字体警告）
适用于 ais-USV_filled.csv

功能：
- 多预测时长：2 / 10 / 20 min
- 真实轨迹（实线） vs 预测轨迹（虚线） + 扇形置信域（90%）
- P50 典型片段 + P90 困难片段（优先选取转向困难片段）
- 中文：宋体；数字/英文/单位：Times New Roman
- 同时输出 PNG 和 PDF
- 所有输出统一保存到一个总文件夹
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Patch
from matplotlib.font_manager import FontProperties

EARTH_R = 6371000.0

# =========================
# 输出目录
# =========================
OUT_DIR = "任务3_中文期刊输出2"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 字体配置
# 请确保以下字体文件与脚本位于同一目录
# =========================
FONT_DIR = "."
FP_SIMSUN = FontProperties(fname=os.path.join(FONT_DIR, "simsun.ttc"))
FP_SIMSUN_BOLD = FontProperties(fname=os.path.join(FONT_DIR, "simsunb.ttf"))
FP_TIMES = FontProperties(fname=os.path.join(FONT_DIR, "times.ttf"))
FP_TIMES_BOLD = FontProperties(fname=os.path.join(FONT_DIR, "timesbd.ttf"))
FP_TIMES_ITALIC = FontProperties(fname=os.path.join(FONT_DIR, "timesi.ttf"))

# =========================
# 参数设置
# =========================
HORIZONS_MIN = (2, 10, 20)
K_CONF = 1.64

DS_EPS = 1.0
V_MOVE_MIN = 0.05
OMEGA_MIN = 0.003

BACKTEST_STEP = 20
SIM_DT = 1.0

FUNNEL_STRIDE = 20
FUNNEL_ALPHA = 0.035

W_GROW_P = 1.10
FUNNEL_BETA = 0.55
W_SHORT_CAP = {2: 6.0, 10: 45.0, 20: 120.0}

OMEGA_WIN_K = 9
OMEGA_MIN_VALID = 3
OMEGA_MIN_TRAVEL = 6.0
OMEGA_LOW_SPEED = 0.30
OMEGA_TURN_THRESH = 0.003
OMEGA_SHRINK = 0.60


def set_paper_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 600,
        "savefig.dpi": 600,
    })


def apply_tick_font(ax, xfont=FP_TIMES, yfont=FP_TIMES):
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(xfont)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(yfont)


def apply_axis_label_font(ax, xlabel_font=FP_SIMSUN, ylabel_font=FP_SIMSUN, title_font=FP_SIMSUN):
    ax.xaxis.label.set_fontproperties(xlabel_font)
    ax.yaxis.label.set_fontproperties(ylabel_font)
    ax.title.set_fontproperties(title_font)


def apply_legend_font(legend_obj, font=FP_SIMSUN):
    if legend_obj is not None:
        for text in legend_obj.get_texts():
            text.set_fontproperties(font)


def save_fig_dual(fig, stem):
    pdf_path = os.path.join(OUT_DIR, f"{stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{stem}.png")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=600)
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return pdf_path, png_path


def latlon_to_enu(lat_deg, lon_deg, lat0_deg, lon0_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)
    east = (lon - lon0) * np.cos(lat0) * EARTH_R
    north = (lat - lat0) * EARTH_R
    return east, north


def set_square_view(ax, xs, ys, pad_ratio=0.12):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    xmin, xmax = np.nanmin(xs), np.nanmax(xs)
    ymin, ymax = np.nanmin(ys), np.nanmax(ys)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin)
    span = max(span, 1e-6)
    half = span * (0.5 + pad_ratio)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_box_aspect(1)


def compute_pos_kinematics(e, n, t_sec, ds_eps=DS_EPS):
    de = np.r_[0.0, np.diff(e)]
    dn = np.r_[0.0, np.diff(n)]
    dt = np.r_[np.nan, np.diff(t_sec)]
    ds = np.sqrt(de**2 + dn**2)

    v = np.full_like(t_sec, np.nan, dtype=float)
    v[1:] = np.divide(ds[1:], dt[1:], out=np.full(len(t_sec) - 1, np.nan), where=dt[1:] > 0)

    theta = np.full_like(t_sec, np.nan, dtype=float)
    theta[1:] = np.arctan2(de[1:], dn[1:])
    theta = np.unwrap(pd.Series(theta).ffill().bfill().to_numpy())

    omega = np.full_like(t_sec, np.nan, dtype=float)
    omega[1:] = np.divide(np.diff(theta), dt[1:], out=np.full(len(t_sec) - 1, np.nan), where=dt[1:] > 0)
    omega[ds < ds_eps] = np.nan

    v_fill = pd.Series(v).ffill().bfill().to_numpy()
    a = np.full_like(t_sec, np.nan, dtype=float)
    a[1:] = np.divide(np.diff(v_fill), dt[1:], out=np.full(len(t_sec) - 1, np.nan), where=dt[1:] > 0)

    return v, theta, omega, a


def interp_at_time(t_sec, y, t_query):
    if t_query < t_sec[0] or t_query > t_sec[-1]:
        return None
    return float(np.interp(t_query, t_sec, y))


def compute_caps_from_ais(df, t_sec):
    sog = df["sog"].to_numpy(dtype=float) * 0.514444
    v_cap = float(np.nanquantile(sog, 0.99))

    cog = np.deg2rad(df["cog"].to_numpy(dtype=float))
    cog = np.unwrap(cog)
    dt = np.r_[np.nan, np.diff(t_sec)]
    omega_cog = np.r_[np.nan, np.diff(cog) / dt[1:]]
    omega_cap = float(np.nanquantile(np.abs(omega_cog), 0.99))
    return v_cap, omega_cap


def robust_omega0(i0, e, n, omega, v0):
    start = max(0, i0 - (OMEGA_WIN_K - 1))
    om_hist = omega[start:i0 + 1]
    om_valid = om_hist[np.isfinite(om_hist)]
    if len(om_valid) == 0:
        return 0.0

    omega0 = float(np.median(om_valid))

    if len(om_valid) < OMEGA_MIN_VALID:
        omega0 = 0.0

    dwin = float(np.hypot(e[i0] - e[start], n[i0] - n[start]))
    if dwin < OMEGA_MIN_TRAVEL:
        omega0 = 0.0

    if (v0 < OMEGA_LOW_SPEED) and (abs(omega0) > OMEGA_TURN_THRESH):
        omega0 = 0.0

    omega0 *= OMEGA_SHRINK
    return omega0


def simulate_cv_ct(e0, n0, v0, th0, omega0, H_sec, dt_sim=SIM_DT, v_cap=None, omega_cap=None):
    steps = int(np.ceil(H_sec / dt_sim))
    ts = np.arange(steps + 1) * dt_sim
    ts[-1] = H_sec

    if omega_cap is not None:
        omega0 = float(np.clip(omega0, -omega_cap, omega_cap))
    if v_cap is not None:
        v0 = float(np.clip(v0, 0.0, v_cap))

    e = np.zeros_like(ts, dtype=float)
    n = np.zeros_like(ts, dtype=float)
    th = np.zeros_like(ts, dtype=float)
    e[0], n[0], th[0] = e0, n0, th0

    for k in range(1, len(ts)):
        dt = ts[k] - ts[k - 1]
        th[k] = th[k - 1] + omega0 * dt
        e[k] = e[k - 1] + v0 * np.sin(th[k]) * dt
        n[k] = n[k - 1] + v0 * np.cos(th[k]) * dt

    return ts, e, n, th


def add_funnel_cloud(ax, x, y, w, stride=FUNNEL_STRIDE, alpha=FUNNEL_ALPHA):
    w = np.asarray(w).copy()
    w[0] = 0.0
    patches = [Circle((float(x[i]), float(y[i])), radius=float(w[i]))
               for i in range(1, len(x), stride)]
    pc = PatchCollection(patches, edgecolor="none", alpha=alpha, zorder=0)
    ax.add_collection(pc)
    return pc


def backtest(e, n, t_sec, v, theta, omega, v_cap=None, omega_cap=None):
    rows = []
    for i0 in range(0, len(t_sec), BACKTEST_STEP):
        if not np.isfinite(v[i0]) or v[i0] < V_MOVE_MIN:
            continue

        v0 = float(v[i0])
        th0 = float(theta[i0])
        omega0 = robust_omega0(i0, e, n, omega, v0)
        state = "转向" if abs(omega0) > OMEGA_MIN else "直航"

        for Hm in HORIZONS_MIN:
            Hs = float(Hm) * 60.0
            t_end = t_sec[i0] + Hs
            e_gt = interp_at_time(t_sec, e, t_end)
            n_gt = interp_at_time(t_sec, n, t_end)
            if e_gt is None:
                continue

            ts, e_pr, n_pr, _ = simulate_cv_ct(
                e[i0], n[i0], v0, th0, omega0, Hs,
                dt_sim=SIM_DT, v_cap=v_cap, omega_cap=omega_cap
            )
            end_err = float(np.hypot(e_gt - e_pr[-1], n_gt - n_pr[-1]))
            rows.append((i0, Hm, state, v0, omega0, end_err))

    return pd.DataFrame(rows, columns=[
        "起始索引", "预测时长_min", "运动状态", "初始速度_mps", "初始角速度_rps", "终点误差_m"
    ])


def add_speed_bins(bt, q=(0.33, 0.66)):
    q1, q2 = bt["初始速度_mps"].quantile(q[0]), bt["初始速度_mps"].quantile(q[1])

    def bin_v(x):
        if x <= q1:
            return "低速"
        if x <= q2:
            return "中速"
        return "高速"

    bt["速度分层"] = bt["初始速度_mps"].map(bin_v)
    bt.attrs["q1"] = float(q1)
    bt.attrs["q2"] = float(q2)
    return bt


def calibrate_sigma(bt):
    return (
        bt.groupby(["预测时长_min", "运动状态", "速度分层"])
        .agg(
            样本数=("终点误差_m", "size"),
            典型误差尺度_m=("终点误差_m", "median")
        )
        .reset_index()
    )


def get_sigma(sig_tbl, Hm, state, speed_bin):
    hit = sig_tbl[
        (sig_tbl["预测时长_min"] == Hm) &
        (sig_tbl["运动状态"] == state) &
        (sig_tbl["速度分层"] == speed_bin)
    ]
    if len(hit) > 0:
        return float(hit["典型误差尺度_m"].iloc[0])

    h = sig_tbl[sig_tbl["预测时长_min"] == Hm]
    if len(h) > 0:
        return float(h["典型误差尺度_m"].median())

    return float(sig_tbl["典型误差尺度_m"].median())


def pick_start_by_quantile(bt, H_pick=10, q=0.50):
    sub = bt[bt["预测时长_min"] == H_pick].copy()
    target = sub["终点误差_m"].quantile(q)
    sub["距离目标差值"] = (sub["终点误差_m"] - target).abs()
    return int(sub.sort_values("距离目标差值").iloc[0]["起始索引"])


def choose_p50_p90(bt):
    i_typ = pick_start_by_quantile(bt, H_pick=10, q=0.50)

    bt_turn = bt[bt["运动状态"] == "转向"]
    if len(bt_turn) >= 30:
        i_hard = pick_start_by_quantile(bt_turn, H_pick=10, q=0.90)
    else:
        i_hard = pick_start_by_quantile(bt, H_pick=10, q=0.90)
    return i_typ, i_hard


def plot_task3(e, n, t_sec, v, theta, omega, sig_tbl, q1, q2,
               i0, tag="P50", stem="任务3",
               v_cap=None, omega_cap=None):
    set_paper_style()

    fig = plt.figure(figsize=(10.5, 6.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.55], hspace=0.35, wspace=0.22)
    axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    ax_err = fig.add_subplot(gs[1, :])

    v0 = float(v[i0])
    th0 = float(theta[i0])
    omega0 = robust_omega0(i0, e, n, omega, v0)
    state = "转向" if abs(omega0) > OMEGA_MIN else "直航"

    def bin_v0(x):
        if x <= q1:
            return "低速"
        if x <= q2:
            return "中速"
        return "高速"

    speed_bin = bin_v0(v0)

    errs = []
    ln_gt = ln_pr = None

    for ax, Hm, lab in zip(axes, HORIZONS_MIN, ["(a)", "(b)", "(c)"]):
        Hs = float(Hm) * 60.0

        ts, e_pr, n_pr, _ = simulate_cv_ct(
            e[i0], n[i0], v0, th0, omega0, Hs,
            dt_sim=SIM_DT, v_cap=v_cap, omega_cap=omega_cap
        )

        t_query = t_sec[i0] + ts
        e_gt = np.interp(t_query, t_sec, e)
        n_gt = np.interp(t_query, t_sec, n)

        e_gt_end = interp_at_time(t_sec, e, t_sec[i0] + Hs)
        n_gt_end = interp_at_time(t_sec, n, t_sec[i0] + Hs)
        if e_gt_end is None:
            ax.set_axis_off()
            continue

        sigma = get_sigma(sig_tbl, Hm, state, speed_bin)

        r = np.clip(ts / Hs, 0.0, 1.0) ** W_GROW_P
        w = K_CONF * sigma * r
        dist = np.maximum(v0 * ts, 1e-3)
        w = np.minimum(w, FUNNEL_BETA * dist)
        w = np.minimum(w, W_SHORT_CAP[int(Hm)])

        ln_gt, = ax.plot(e_gt, n_gt, "-", linewidth=2.2, zorder=4, label="真实轨迹")
        ln_pr, = ax.plot(e_pr, n_pr, "--", linewidth=1.6, zorder=3, label="预测轨迹")

        add_funnel_cloud(ax, e_pr, n_pr, w, stride=FUNNEL_STRIDE, alpha=FUNNEL_ALPHA)

        ax.scatter([e[i0]], [n[i0]], s=26, c="k", zorder=5)
        ax.scatter([e_gt_end], [n_gt_end], s=40, c="k", marker="^", zorder=5)

        # 关键修复：标题含中文，因此标题字体必须用宋体
        ax.set_title(f"{lab} {tag}，预测时长={Hm} min", pad=4)
        ax.set_xlabel("东向坐标/m")
        ax.set_ylabel("北向坐标/m", labelpad=0.2)

        set_square_view(ax, np.r_[e_gt, e_pr], np.r_[n_gt, n_pr], pad_ratio=0.12)
        apply_axis_label_font(ax, xlabel_font=FP_SIMSUN, ylabel_font=FP_SIMSUN, title_font=FP_SIMSUN)
        apply_tick_font(ax, xfont=FP_TIMES, yfont=FP_TIMES)

        end_err = float(np.hypot(e_gt_end - e_pr[-1], n_gt_end - n_pr[-1]))
        rmse = float(np.sqrt(np.mean((e_gt - e_pr) ** 2 + (n_gt - n_pr) ** 2)))
        errs.append((Hm, end_err, rmse))

    if errs:
        hs = [x[0] for x in errs]
        ee = [x[1] for x in errs]
        rm = [x[2] for x in errs]
        ax_err.plot(hs, ee, marker="o", linewidth=1.6, label="终点误差")
        ax_err.plot(hs, rm, marker="s", linewidth=1.6, label="均方根误差")
        ax_err.set_xlabel("预测时长/min")
        ax_err.set_ylabel("误差/m")
        ax_err.set_xticks(list(HORIZONS_MIN))
        ax_err.set_xlim(min(HORIZONS_MIN) - 0.5, max(HORIZONS_MIN) + 0.5)
        ax_err.grid(True, alpha=0.25)

        lg_err = ax_err.legend(loc="upper left", frameon=True)
        apply_legend_font(lg_err, font=FP_SIMSUN)
        apply_axis_label_font(ax_err, xlabel_font=FP_SIMSUN, ylabel_font=FP_SIMSUN, title_font=FP_SIMSUN)
        apply_tick_font(ax_err, xfont=FP_TIMES, yfont=FP_TIMES)

    funnel_proxy = Patch(alpha=FUNNEL_ALPHA, label="扇形置信域（90%）")
    lg = fig.legend(
        [ln_gt, ln_pr, funnel_proxy],
        ["真实轨迹", "预测轨迹", "扇形置信域（90%）"],
        loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.98)
    )
    apply_legend_font(lg, font=FP_SIMSUN)

    save_fig_dual(fig, stem)


def main():
    csv_path = "ais-USV_filled.csv"
    df = pd.read_csv(csv_path)

    df["t"] = pd.to_datetime(df["base_date_time"])
    df = df.sort_values("t").drop_duplicates("t", keep="last").reset_index(drop=True)

    lat0, lon0 = float(df.loc[0, "latitude"]), float(df.loc[0, "longitude"])
    e, n = latlon_to_enu(
        df["latitude"].to_numpy(float),
        df["longitude"].to_numpy(float),
        lat0, lon0
    )

    t = df["t"].to_numpy()
    t_sec = ((t - t[0]) / np.timedelta64(1, "s")).astype(float)

    v_pos, th_pos, om_pos, _ = compute_pos_kinematics(e, n, t_sec, ds_eps=DS_EPS)
    v_cap, omega_cap = compute_caps_from_ais(df, t_sec)

    bt = backtest(e, n, t_sec, v_pos, th_pos, om_pos, v_cap=v_cap, omega_cap=omega_cap)
    if len(bt) == 0:
        raise RuntimeError("回测样本数为0，请适当降低 V_MOVE_MIN 或 BACKTEST_STEP。")

    bt = add_speed_bins(bt)
    q1, q2 = bt.attrs["q1"], bt.attrs["q2"]
    sig_tbl = calibrate_sigma(bt)

    bt.to_csv(os.path.join(OUT_DIR, "任务3_回测误差表.csv"), index=False, encoding="utf-8-sig")
    sig_tbl.to_csv(os.path.join(OUT_DIR, "任务3_置信域参数表.csv"), index=False, encoding="utf-8-sig")

    i_typ, i_hard = choose_p50_p90(bt)

    plot_task3(
        e, n, t_sec, v_pos, th_pos, om_pos, sig_tbl, q1, q2,
        i_typ, tag="P50", stem="任务3_P50_典型片段",
        v_cap=v_cap, omega_cap=omega_cap
    )

    plot_task3(
        e, n, t_sec, v_pos, th_pos, om_pos, sig_tbl, q1, q2,
        i_hard, tag="P90", stem="任务3_P90_困难片段",
        v_cap=v_cap, omega_cap=omega_cap
    )

    print("处理完成。")
    print("输出目录：", OUT_DIR)
    print("输出文件：")
    print("  - 任务3_P50_典型片段.pdf")
    print("  - 任务3_P50_典型片段.png")
    print("  - 任务3_P90_困难片段.pdf")
    print("  - 任务3_P90_困难片段.png")
    print("  - 任务3_回测误差表.csv")
    print("  - 任务3_置信域参数表.csv")
    print(f"AIS截断上限：速度上限={v_cap:.3f} m/s，角速度上限={omega_cap:.6f} rad/s")
    print(f"置信域参数：alpha={FUNNEL_ALPHA}, stride={FUNNEL_STRIDE}, beta={FUNNEL_BETA}, "
          f"W_SHORT_CAP={W_SHORT_CAP}, grow_p={W_GROW_P}")


if __name__ == "__main__":
    main()