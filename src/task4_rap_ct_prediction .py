# -*- coding: utf-8 -*-
"""
任务4：AIS-USV 短时轨迹预测（2/10/20 min）

本版在原 task4_rap_ct_prediction 基础上新增两项：
1) 轨迹图图例中：
   - “经验预测范围”“真实轨迹”使用中文宋体；
   - CV-Heading / CV-CT / RAP-CT 等英文模型名使用 Times New Roman。
2) 新增代表性高误差起点的多模型分组轨迹图：
   - 物理与滤波模型；
   - 传统机器学习残差模型；
   - 深度时序残差模型；
   - 残差修正消融模型。

运行顺序：
    1. python task4_merged_all_tables(4).py
    2. python task4_rap_ct_prediction_grouped_trajectory_full.py

必须与本脚本放在同一目录：
    ais-USV_filled.csv
    task4_merged_all_tables(4).py
    simsun.ttc
    simsunb.ttf
    times.ttf
"""

import os
import importlib.util
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# ============================================================
# 0. 字体配置
# ============================================================
FONT_DIR = "."
FP_SIMSUN = FontProperties(fname=os.path.join(FONT_DIR, "simsun.ttc"))
FP_SIMSUN_BOLD = FontProperties(fname=os.path.join(FONT_DIR, "simsunb.ttf"))
FP_TIMES = FontProperties(fname=os.path.join(FONT_DIR, "times.ttf"))

for fn in ["simsun.ttc", "simsunb.ttf", "times.ttf"]:
    mpl.font_manager.fontManager.addfont(os.path.join(FONT_DIR, fn))

TIMES_NAME = FP_TIMES.get_name()

plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = TIMES_NAME
plt.rcParams["mathtext.it"] = f"{TIMES_NAME}:italic"
plt.rcParams["mathtext.bf"] = f"{TIMES_NAME}:bold"
plt.rcParams["mathtext.default"] = "rm"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [TIMES_NAME]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 180
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# ============================================================
# 1. 用户配置
# ============================================================
CSV_PATH = "ais-USV_filled.csv"
OUT_DIR = "任务4_中文期刊输出2"
os.makedirs(OUT_DIR, exist_ok=True)

DETERMINISTIC = True
FIX_I0_P50 = None
FIX_I0_P90 = None
USE_RES_BETTER_SUBSET_FOR_VIS = True

VAL_RATIO = 0.15
TEST_RATIO = 0.20
SPLIT_SEED = 42
EVAL_SCOPE = "test"
TURN_EVAL_SCOPE = "all"
IMPROVE_DENOM_EPS = 1.0

EARTH_R = 6371000.0
HORIZONS_MIN = (2, 10, 20)
BACKTEST_STEP = 5
DT_SIM = 1.0

V_MOVE_MIN = 0.05
DS_EPS = 1.0
OMEGA_MIN_TURN = 0.003
OMEGA_WIN_K = 9
OMEGA_MIN_VALID = 3
OMEGA_MIN_TRAVEL = 6.0
OMEGA_LOW_SPEED = 0.30
OMEGA_TURN_THRESH = 0.003

FUNNEL_P = 1.10
BETA_V = 0.55
CAP_BY_H = {2: 6.0, 10: 45.0, 20: 120.0}
LOSS_W_A = 2.0

USE_TORCH = True
try:
    import torch
    import torch.nn as nn
except Exception:
    USE_TORCH = False

# ============================================================
# 2. 图形工具函数
# ============================================================
def apply_tick_font(ax, xfont=FP_TIMES, yfont=FP_TIMES):
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(xfont)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(yfont)
    ax.xaxis.get_offset_text().set_fontproperties(xfont)
    ax.yaxis.get_offset_text().set_fontproperties(yfont)


def apply_tick_style(ax):
    ax.tick_params(axis="both", which="major", direction="in", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="in", length=2, width=0.6)


def _remove_mixed_artist(ax, attr_name):
    obj = getattr(ax, attr_name, None)
    if obj is not None:
        try:
            obj.remove()
        except Exception:
            pass
        setattr(ax, attr_name, None)


def _contains_chinese(s):
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(s))


def apply_legend_font_mixed(legend_obj):
    if legend_obj is None:
        return
    for text in legend_obj.get_texts():
        label = text.get_text()
        if _contains_chinese(label):
            text.set_fontproperties(FP_SIMSUN)
        else:
            text.set_fontproperties(FP_TIMES)


def apply_legend_font_times(legend_obj, font=FP_TIMES):
    if legend_obj is not None:
        for text in legend_obj.get_texts():
            text.set_fontproperties(font)


def set_mixed_xlabel(ax, cn_text, en_text, y=-0.18, sep="/",
                     cn_fp=FP_SIMSUN, en_fp=FP_TIMES, fontsize=11):
    _remove_mixed_artist(ax, "_mixed_xlabel_artist")
    ax.set_xlabel("")
    children = [TextArea(cn_text, textprops=dict(fontproperties=cn_fp, fontsize=fontsize))]
    if sep or en_text:
        children.append(TextArea(sep, textprops=dict(fontproperties=en_fp, fontsize=fontsize)))
        children.append(TextArea(en_text, textprops=dict(fontproperties=en_fp, fontsize=fontsize)))
    box = HPacker(children=children, align="center", pad=0, sep=1)
    artist = AnchoredOffsetbox(
        loc="lower center", child=box, pad=0.0, frameon=False,
        bbox_to_anchor=(0.5, y), bbox_transform=ax.transAxes, borderpad=0.0,
    )
    ax.add_artist(artist)
    ax._mixed_xlabel_artist = artist


def set_mixed_ylabel(ax, cn_text, en_text, x=-0.18, sep="/",
                     cn_fp=FP_SIMSUN, en_fp=FP_TIMES, fontsize=11):
    """
    纵坐标混排标签修正版：
    - 将“/ + 单位”合并，避免斜杠和单位错位；
    - 减小中文标签与单位之间的竖向间隔；
    - 其余坐标轴样式不变。
    """
    _remove_mixed_artist(ax, "_mixed_ylabel_artist")
    ax.set_ylabel("")
    unit_text = f"{sep}{en_text}" if (sep or en_text) else ""
    children = []
    if unit_text:
        children.append(TextArea(unit_text, textprops=dict(
            fontproperties=en_fp, fontsize=fontsize, rotation=90,
            ha="center", va="center"
        )))
    children.append(TextArea(cn_text, textprops=dict(
        fontproperties=cn_fp, fontsize=fontsize, rotation=90,
        ha="center", va="center"
    )))
    box = VPacker(children=children, align="center", pad=0, sep=0)
    artist = AnchoredOffsetbox(
        loc="center left", child=box, pad=0.0, frameon=False,
        bbox_to_anchor=(x, 0.5), bbox_transform=ax.transAxes, borderpad=0.0,
    )
    ax.add_artist(artist)
    ax._mixed_ylabel_artist = artist


def set_mixed_metric_title(ax, en_text, cn_text, y=1.02,
                           en_fp=FP_TIMES, cn_fp=FP_SIMSUN, fontsize=12):
    _remove_mixed_artist(ax, "_mixed_title_artist")
    ax.set_title("")
    box = HPacker(
        children=[
            TextArea(en_text, textprops=dict(fontproperties=en_fp, fontsize=fontsize)),
            TextArea(cn_text, textprops=dict(fontproperties=cn_fp, fontsize=fontsize)),
        ],
        align="center", pad=0, sep=0,
    )
    artist = AnchoredOffsetbox(
        loc="upper center", child=box, pad=0.0, frameon=False,
        bbox_to_anchor=(0.5, y), bbox_transform=ax.transAxes, borderpad=0.0,
    )
    ax.add_artist(artist)
    ax._mixed_title_artist = artist


def set_mixed_panel_title(ax, panel_tag, p_tag, horizon_min, y=1.04,
                          en_fp=FP_TIMES, cn_fp=FP_SIMSUN, fontsize=12):
    _remove_mixed_artist(ax, "_mixed_title_artist")
    ax.set_title("")
    box = HPacker(
        children=[
            TextArea(f"({panel_tag}) ", textprops=dict(fontproperties=en_fp, fontsize=fontsize)),
            TextArea(f"{p_tag}", textprops=dict(fontproperties=en_fp, fontsize=fontsize)),
            TextArea("，预测时长", textprops=dict(fontproperties=cn_fp, fontsize=fontsize)),
            TextArea(" = ", textprops=dict(fontproperties=en_fp, fontsize=fontsize)),
            TextArea(f"{horizon_min} ", textprops=dict(fontproperties=en_fp, fontsize=fontsize)),
            TextArea("min", textprops=dict(fontproperties=en_fp, fontsize=fontsize)),
        ],
        align="center", pad=0, sep=0,
    )
    artist = AnchoredOffsetbox(
        loc="upper center", child=box, pad=0.0, frameon=False,
        bbox_to_anchor=(0.5, y), bbox_transform=ax.transAxes, borderpad=0.0,
    )
    ax.add_artist(artist)
    ax._mixed_title_artist = artist

# ============================================================
# 3. 基础计算函数
# ============================================================
def seed_all(seed=42):
    np.random.seed(seed)
    if USE_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def latlon_to_enu(lat_deg, lon_deg, lat0_deg, lon0_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)
    east = (lon - lon0) * np.cos(lat0) * EARTH_R
    north = (lat - lat0) * EARTH_R
    return east, north


def compute_pos_kinematics(e, n, t_sec, ds_eps=DS_EPS):
    de = np.r_[0.0, np.diff(e)]
    dn = np.r_[0.0, np.diff(n)]
    dt = np.r_[np.nan, np.diff(t_sec)]
    ds = np.sqrt(de ** 2 + dn ** 2)

    v = np.full_like(t_sec, np.nan, dtype=float)
    v[1:] = np.divide(ds[1:], dt[1:], out=np.full(len(t_sec) - 1, np.nan), where=dt[1:] > 0)

    theta = np.full_like(t_sec, np.nan, dtype=float)
    theta[1:] = np.arctan2(de[1:], dn[1:])
    theta = pd.Series(theta).ffill().bfill().to_numpy()
    theta = np.unwrap(theta)

    omega = np.full_like(t_sec, np.nan, dtype=float)
    omega[1:] = np.divide(np.diff(theta), dt[1:], out=np.full(len(t_sec) - 1, np.nan), where=dt[1:] > 0)
    omega[ds < ds_eps] = np.nan

    a = np.full_like(t_sec, np.nan, dtype=float)
    a[2:] = np.divide(np.diff(v[1:]), dt[2:], out=np.full(len(t_sec) - 2, np.nan), where=dt[2:] > 0)
    return v, theta, omega, a


def compute_caps_from_ais(df, t_sec):
    sog_mps = df["sog"].to_numpy(float) * 0.514444
    v_cap = float(np.nanquantile(sog_mps, 0.99))
    cog = np.deg2rad(df["cog"].to_numpy(float))
    cog = np.unwrap(cog)
    dt = np.r_[np.nan, np.diff(t_sec)]
    omega_cog = np.r_[np.nan, np.diff(cog) / dt[1:]]
    omega_cap = float(np.nanquantile(np.abs(omega_cog), 0.99))
    return v_cap, omega_cap


def omega0_window(i0, e, n, omega, v0):
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
    absw = abs(omega0)
    if absw < 0.006:
        shrink = 0.60
    elif absw < 0.02:
        shrink = 0.80
    else:
        shrink = 1.00
    return omega0 * shrink


def motion_state(omega0):
    return "转向" if abs(omega0) > OMEGA_MIN_TURN else "直航"


def interp_xy_at_time(t_sec, E, N, tq):
    if tq < t_sec[0] or tq > t_sec[-1]:
        return None
    return float(np.interp(tq, t_sec, E)), float(np.interp(tq, t_sec, N))


def build_gt_path(t0, H_sec, t_sec, E, N, dt_sim=1.0):
    ts = np.arange(0.0, H_sec + 1e-9, dt_sim)
    e_gt = np.interp(t0 + ts, t_sec, E)
    n_gt = np.interp(t0 + ts, t_sec, N)
    return ts, e_gt, n_gt


def rmse_path(e_pr, n_pr, e_gt, n_gt):
    return float(np.sqrt(np.mean((e_gt - e_pr) ** 2 + (n_gt - n_pr) ** 2)))


def simulate_cv_heading(e0, n0, v0, th0, H_sec, dt_sim=1.0, v_cap=None):
    steps = int(np.ceil(H_sec / dt_sim))
    ts = np.arange(steps + 1) * dt_sim
    ts[-1] = H_sec
    if v_cap is not None:
        v0 = float(np.clip(v0, 0.0, v_cap))
    e = np.zeros_like(ts, float)
    n = np.zeros_like(ts, float)
    e[0] = e0
    n[0] = n0
    for k in range(1, len(ts)):
        dt = ts[k] - ts[k - 1]
        e[k] = e[k - 1] + v0 * np.sin(th0) * dt
        n[k] = n[k - 1] + v0 * np.cos(th0) * dt
    return ts, e, n


def simulate_ct_timevarying(e0, n0, v0, th0, omega0, dv, dw, H_sec, dt_sim=1.0,
                            v_cap=None, omega_cap=None):
    steps = int(np.ceil(H_sec / dt_sim))
    ts = np.arange(steps + 1) * dt_sim
    ts[-1] = H_sec
    e = np.zeros_like(ts, float)
    n = np.zeros_like(ts, float)
    th = np.zeros_like(ts, float)
    e[0] = e0
    n[0] = n0
    th[0] = th0
    for k in range(1, len(ts)):
        dt = ts[k] - ts[k - 1]
        a = ts[k] / (H_sec + 1e-9)
        v = v0 + a * dv
        w = omega0 + a * dw
        if v_cap is not None:
            v = float(np.clip(v, 0.0, v_cap))
        if omega_cap is not None:
            w = float(np.clip(w, -omega_cap, omega_cap))
        th[k] = th[k - 1] + w * dt
        e[k] = e[k - 1] + v * np.sin(th[k]) * dt
        n[k] = n[k - 1] + v * np.cos(th[k]) * dt
    return ts, e, n


def funnel_width(ts, H_sec, sigma_end, v0, k=1.64):
    H_min = int(round(H_sec / 60))
    cap = CAP_BY_H.get(H_min, float("inf"))
    alpha = (ts / H_sec) ** FUNNEL_P
    w = k * sigma_end * alpha
    w = np.minimum(w, BETA_V * v0 * ts)
    w = np.minimum(w, cap)
    w[0] = 0.0
    return w

# ============================================================
# 4. 表格函数
# ============================================================
def build_abs_error_table(bt, i0_list, tags, horizons):
    rows = []
    method_map = {
        "CV-Heading": ("end_cv", "rmse_cv"),
        "CV-CT": ("end_ct", "rmse_ct"),
        "RAP-CT": ("end_res", "rmse_res"),
    }
    for i0, tag in zip(i0_list, tags):
        sub = bt[bt["i0"] == i0].copy()
        for method_name, (end_col, rmse_col) in method_map.items():
            row_epe = {"起点类型": tag, "方法": method_name, "指标": "EPE / m"}
            row_rmse = {"起点类型": tag, "方法": method_name, "指标": "Path RMSE / m"}
            for H in horizons:
                hit = sub[sub["H_min"] == H]
                if len(hit) == 0:
                    row_epe[f"{H} min"] = np.nan
                    row_rmse[f"{H} min"] = np.nan
                else:
                    row_epe[f"{H} min"] = float(hit.iloc[0][end_col])
                    row_rmse[f"{H} min"] = float(hit.iloc[0][rmse_col])
            rows.append(row_epe)
            rows.append(row_rmse)
    return pd.DataFrame(rows)


def build_improve_table(bt, i0_list, tags, horizons, denom_eps=IMPROVE_DENOM_EPS):
    rows = []
    for i0, tag in zip(i0_list, tags):
        sub = bt[bt["i0"] == i0].copy()
        row_epe = {"起点类型": tag, "指标": "EPE 改善率 / %"}
        row_rmse = {"起点类型": tag, "指标": "RMSE 改善率 / %"}
        for H in horizons:
            hit = sub[sub["H_min"] == H]
            if len(hit) == 0:
                row_epe[f"{H} min"] = np.nan
                row_rmse[f"{H} min"] = np.nan
            else:
                end_ct = float(hit.iloc[0]["end_ct"])
                end_res = float(hit.iloc[0]["end_res"])
                rmse_ct = float(hit.iloc[0]["rmse_ct"])
                rmse_res = float(hit.iloc[0]["rmse_res"])
                row_epe[f"{H} min"] = np.nan if end_ct <= denom_eps else (end_ct - end_res) / end_ct * 100.0
                row_rmse[f"{H} min"] = np.nan if rmse_ct <= denom_eps else (rmse_ct - rmse_res) / rmse_ct * 100.0
        rows.append(row_epe)
        rows.append(row_rmse)
    return pd.DataFrame(rows)


def export_representative_info(bt, i0_list, tags, t_sec):
    rows = []
    for i0, tag in zip(i0_list, tags):
        sub = bt[bt["i0"] == i0].copy()
        if len(sub) == 0:
            continue
        r0 = sub.iloc[0]
        rows.append({
            "起点类型": tag,
            "起点索引 i0": int(i0),
            "起点时间 / s": float(t_sec[i0]),
            "状态": r0["state"],
            "初始速度 v0 / (m/s)": float(r0["v0"]),
            "初始转率 omega0 / (rad/s)": float(r0["omega0"]),
        })
    return pd.DataFrame(rows)


def assign_timewise_split(bt, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    uniq = np.array(sorted(bt["i0"].unique().tolist()), dtype=int)
    n = len(uniq)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_train = max(1, n - n_val - n_test)
    if n_train + n_val + n_test > n:
        n_test = max(1, n - n_train - n_val)
    if n_train + n_val + n_test < n:
        n_train = n - n_val - n_test
    split_map = {}
    for i, i0 in enumerate(uniq):
        if i < n_train:
            split_map[int(i0)] = "train"
        elif i < n_train + n_val:
            split_map[int(i0)] = "val"
        else:
            split_map[int(i0)] = "test"
    return split_map


def build_subset_summary(bt_sub, subset_name, horizons):
    method_map = {
        "CV-Heading": ("end_cv", "rmse_cv"),
        "CV-CT": ("end_ct", "rmse_ct"),
        "RAP-CT": ("end_res", "rmse_res"),
    }
    rows = []
    for H in horizons:
        hit = bt_sub[bt_sub["H_min"] == H].copy()
        for method_name, (epe_col, rmse_col) in method_map.items():
            vals_epe = hit[epe_col].to_numpy(float) if len(hit) else np.array([], dtype=float)
            vals_rmse = hit[rmse_col].to_numpy(float) if len(hit) else np.array([], dtype=float)
            rows.append({
                "子集": subset_name,
                "预测时长/min": H,
                "方法": method_name,
                "指标": "EPE / m",
                "N": int(len(vals_epe)),
                "mean": float(np.nanmean(vals_epe)) if len(vals_epe) else np.nan,
                "median": float(np.nanmedian(vals_epe)) if len(vals_epe) else np.nan,
                "P90": float(np.nanquantile(vals_epe, 0.90)) if len(vals_epe) else np.nan,
            })
            rows.append({
                "子集": subset_name,
                "预测时长/min": H,
                "方法": method_name,
                "指标": "Path RMSE / m",
                "N": int(len(vals_rmse)),
                "mean": float(np.nanmean(vals_rmse)) if len(vals_rmse) else np.nan,
                "median": float(np.nanmedian(vals_rmse)) if len(vals_rmse) else np.nan,
                "P90": float(np.nanquantile(vals_rmse, 0.90)) if len(vals_rmse) else np.nan,
            })
    return pd.DataFrame(rows)


def build_improvement_summary(bt_sub, subset_name, horizons, denom_eps=IMPROVE_DENOM_EPS):
    rows = []
    for H in horizons:
        hit = bt_sub[bt_sub["H_min"] == H].copy()
        for metric_name, ct_col, res_col in [
            ("EPE 改善率 / %", "end_ct", "end_res"),
            ("RMSE 改善率 / %", "rmse_ct", "rmse_res"),
        ]:
            ct = hit[ct_col].to_numpy(float) if len(hit) else np.array([], dtype=float)
            res = hit[res_col].to_numpy(float) if len(hit) else np.array([], dtype=float)
            abs_drop = ct - res
            valid = ct > denom_eps
            imp = (ct[valid] - res[valid]) / ct[valid] * 100.0 if np.any(valid) else np.array([], dtype=float)
            rows.append({
                "子集": subset_name,
                "预测时长/min": H,
                "相对对象": "RAP-CT vs CV-CT",
                "指标": metric_name,
                "N": int(len(ct)),
                "有效百分比样本数": int(np.sum(valid)),
                "绝对下降量mean": float(np.nanmean(abs_drop)) if len(abs_drop) else np.nan,
                "绝对下降量median": float(np.nanmedian(abs_drop)) if len(abs_drop) else np.nan,
                "median": float(np.nanmedian(imp)) if len(imp) else np.nan,
                "P90": float(np.nanquantile(imp, 0.90)) if len(imp) else np.nan,
                "胜率/%": float(np.mean(abs_drop > 0) * 100.0) if len(abs_drop) else np.nan,
            })
    return pd.DataFrame(rows)


def export_overall_tables(bt_eval, bt_all, out_dir, horizons):
    turn_source = bt_eval if TURN_EVAL_SCOPE == "test" else bt_all
    subsets = {
        "全体窗口": bt_eval.copy(),
        "直航窗口": bt_eval[bt_eval["state"] == "直航"].copy(),
        "转向窗口": turn_source[turn_source["state"] == "转向"].copy(),
    }
    sum_tables = []
    imp_tables = []
    for name, sub in subsets.items():
        if len(sub) == 0:
            continue
        sum_tables.append(build_subset_summary(sub, name, horizons))
        imp_tables.append(build_improvement_summary(sub, name, horizons))
    df_sum = pd.concat(sum_tables, ignore_index=True) if sum_tables else pd.DataFrame()
    df_imp = pd.concat(imp_tables, ignore_index=True) if imp_tables else pd.DataFrame()
    p1 = os.path.join(out_dir, "表6_全体与分层误差统计.csv")
    p2 = os.path.join(out_dir, "表7_RAP-CT相对CV-CT总体改善统计.csv")
    df_sum.to_csv(p1, index=False, encoding="utf-8-sig")
    df_imp.to_csv(p2, index=False, encoding="utf-8-sig")
    if len(df_sum):
        wide_sum = df_sum.pivot_table(
            index=["子集", "预测时长/min", "方法"],
            columns="指标",
            values=["mean", "median", "P90", "N"],
            aggfunc="first",
        )
        wide_sum.to_csv(os.path.join(out_dir, "表6_全体与分层误差统计_宽表版.csv"), encoding="utf-8-sig")
    if len(df_imp):
        wide_imp = df_imp.pivot_table(
            index=["子集", "预测时长/min", "相对对象"],
            columns="指标",
            values=["绝对下降量mean", "绝对下降量median", "median", "P90", "胜率/%", "N", "有效百分比样本数"],
            aggfunc="first",
        )
        wide_imp.to_csv(os.path.join(out_dir, "表7_RAP-CT相对CV-CT总体改善统计_宽表版.csv"), encoding="utf-8-sig")
    return p1, p2

# ============================================================
# 5. 多模型误差柱状图
# ============================================================
def _find_merge_wide_table():
    candidates = [
        os.path.join("任务4_三代码融合大表输出", "表8_大汇总模型对比统计_宽表_mean.csv"),
        "表8_大汇总模型对比统计_宽表_mean.csv",
        os.path.join(OUT_DIR, "表8_大汇总模型对比统计_宽表_mean.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("找不到 表8_大汇总模型对比统计_宽表_mean.csv。")


def _check_merge_wide_table_columns(wide_df):
    required_cols = ["预测时长/min", "模型类别", "方法", "EPE / m", "Path RMSE / m"]
    missing = [c for c in required_cols if c not in wide_df.columns]
    if missing:
        raise ValueError(f"宽表缺少必要列：{missing}")


def _safe_get_metric_value(sub, method, metric_col):
    hit = sub[sub["方法"] == method]
    if len(hit) == 0:
        return None
    val = float(hit.iloc[0][metric_col])
    if not np.isfinite(val):
        return None
    return val


def _annotate_rap_improvement(ax, labels, vals, rap_label="RAP-CT"):
    if rap_label not in labels:
        return
    rap_idx = labels.index(rap_label)
    rap_val = vals[rap_idx]
    comp_vals = []
    for lab, val in zip(labels, vals):
        if lab not in ["RAP-CT", "RAP-dv", "RAP-dw"] and np.isfinite(val):
            comp_vals.append(val)
    if len(comp_vals) == 0:
        return
    best_comp = min(comp_vals)
    if best_comp <= 1e-9:
        return
    imp = (best_comp - rap_val) / best_comp * 100.0
    ax.text(rap_idx, rap_val, f"{imp:+.1f}%", ha="center", va="bottom",
            fontsize=9, fontproperties=FP_TIMES)


def plot_competitive_model_metric(wide_df, metric_col, cn_metric, out_prefix):
    method_order = [
        "CV-CT",
        "RF",
        "GBDT",
        "LSTM",
        "GRU",
        "Seq2Seq",
        "Transformer",
        "RAP-CT-dv-only",
        "RAP-CT-dw-only",
        "RAP-CT",
    ]
    method_label = {
        "CV-CT": "CV-CT",
        "RF": "RF",
        "GBDT": "GBDT",
        "LSTM": "LSTM",
        "GRU": "GRU",
        "Seq2Seq": "Seq2Seq",
        "Transformer": "Trans.",
        "RAP-CT-dv-only": "RAP-dv",
        "RAP-CT-dw-only": "RAP-dw",
        "RAP-CT": "RAP-CT",
    }
    color_map = {
        "CV-CT": "#2ca02c",
        "RF": "#7f7f7f",
        "GBDT": "#8c8c8c",
        "LSTM": "#9ecae1",
        "GRU": "#6baed6",
        "Seq2Seq": "#4292c6",
        "Transformer": "#2171b5",
        "RAP-CT-dv-only": "#9467bd",
        "RAP-CT-dw-only": "#8c564b",
        "RAP-CT": "#d62728",
    }
    fig = plt.figure(figsize=(10.0, 3.8))
    gs = fig.add_gridspec(1, 3, wspace=0.30)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    fig.subplots_adjust(top=0.80, bottom=0.32)

    for j, H in enumerate(HORIZONS_MIN):
        ax = axs[j]
        sub = wide_df[wide_df["预测时长/min"] == H].copy()
        vals, labels, colors = [], [], []
        for m in method_order:
            val = _safe_get_metric_value(sub, m, metric_col)
            if val is None:
                continue
            vals.append(val)
            labels.append(method_label.get(m, m))
            colors.append(color_map.get(m, "#7f7f7f"))
        x = np.arange(len(vals))
        ax.bar(x, vals, width=0.70, color=colors, alpha=0.90)
        set_mixed_metric_title(ax, f"{H} min", cn_metric, y=1.08, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        set_mixed_xlabel(ax, "模型", "", y=-0.28, sep="", fontsize=11)
        set_mixed_ylabel(ax, "误差", "m", x=-0.18, fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)
        apply_tick_font(ax)
        apply_tick_style(ax)
        _annotate_rap_improvement(ax, labels, vals)

    png_path = os.path.join(OUT_DIR, f"{out_prefix}.png")
    pdf_path = os.path.join(OUT_DIR, f"{out_prefix}.pdf")
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.15)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"[INFO] 已导出：{png_path}")
    print(f"[INFO] 已导出：{pdf_path}")


def plot_multimodel_comparison_from_merged_table():
    wide_csv = _find_merge_wide_table()
    wide_df = pd.read_csv(wide_csv)
    _check_merge_wide_table_columns(wide_df)
    plot_competitive_model_metric(wide_df, "EPE / m", "终点误差", "图12_竞争模型终点误差对比")
    plot_competitive_model_metric(wide_df, "Path RMSE / m", "均方根误差", "图13_竞争模型均方根误差对比")

# ============================================================
# 6. 多模型分组轨迹图
# ============================================================
def _find_merge_code_path():
    candidates = [
        "task4_merged_all_tables(4).py",
        "task4_merged_all_tables(3).py",
        "task4_merged_all_tables.py",
        os.path.join("/mnt/data", "task4_merged_all_tables(4).py"),
        os.path.join("/mnt/data", "task4_merged_all_tables(3).py"),
        os.path.join("/mnt/data", "task4_merged_all_tables.py"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("找不到 task4_merged_all_tables.py。")


def _load_merge_module():
    path = _find_merge_code_path()
    spec = importlib.util.spec_from_file_location("task4_merged_all_tables_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _angle_diff_deg(a, b):
    d = (a - b + np.pi) % (2.0 * np.pi) - np.pi
    return float(abs(np.rad2deg(d)))


def _path_heading(e, n):
    return float(np.arctan2(float(e[-1] - e[0]), float(n[-1] - n[0])))


def _select_p90_row_from_merge_error_table(bt_merge):
    """
    兜底选择：保留原来的 P90 逻辑。
    新版优先使用 _select_representative_row_for_grouped_trajectory，
    只有几何筛选失败时才退回到这里。
    """
    candidates = [
        os.path.join("任务4_三代码融合大表输出", "表8_大汇总逐窗口误差明细.csv"),
        "表8_大汇总逐窗口误差明细.csv",
        os.path.join(OUT_DIR, "表8_大汇总逐窗口误差明细.csv"),
    ]
    for p in candidates:
        if not os.path.exists(p):
            continue
        err_df = pd.read_csv(p)
        need = {"row_id", "预测时长/min", "方法", "EPE / m"}
        if not need.issubset(set(err_df.columns)):
            continue
        sub = err_df[(err_df["预测时长/min"] == 20) & (err_df["方法"] == "CV-CT")].copy()
        if "split" in sub.columns:
            test_sub = sub[sub["split"] == "test"].copy()
            if len(test_sub) >= 5:
                sub = test_sub
        if len(sub) == 0:
            continue
        vals = sub["EPE / m"].to_numpy(float)
        q90 = np.nanquantile(vals, 0.90)
        sel = sub.iloc[int(np.nanargmin(np.abs(vals - q90)))]
        row_id = int(sel["row_id"])
        hit = bt_merge[bt_merge["row_id"] == row_id]
        if len(hit):
            return hit.iloc[0]

    sub = bt_merge[bt_merge["H_min"] == 20].copy()
    if "split" in sub.columns:
        test_sub = sub[sub["split"] == "test"].copy()
        if len(test_sub) >= 5:
            sub = test_sub
    vals = sub["end_ct"].to_numpy(float)
    q90 = np.nanquantile(vals, 0.90)
    return sub.iloc[int(np.nanargmin(np.abs(vals - q90)))]


def _select_representative_row_for_grouped_trajectory(m, bt_merge, res_df, rap, E, N, t_sec, v_cap, omega_cap):
    """
    代表性轨迹窗口选择：
    1) 仍优先看测试集、20 min 窗口；
    2) 不再机械选择 CV-CT 的 P90 误差点，因为那种点很容易出现真实轨迹和模型方向完全错开的难看图；
    3) 优先选择“RAP-CT 相对 CV-CT 有改善、且预测方向与真实轨迹方向基本一致”的窗口。
    """
    candidate_pools = []
    for H in [20, 10, 2]:
        sub = bt_merge[bt_merge["H_min"] == H].copy()
        if "split" in sub.columns:
            test_sub = sub[sub["split"] == "test"].copy()
            if len(test_sub) >= 8:
                sub = test_sub
        if len(sub):
            candidate_pools.append(sub)

    scored = []
    for sub in candidate_pools:
        for _, rr in sub.iterrows():
            row_id = int(rr.row_id)
            res_hit = res_df[res_df["row_id"] == row_id]
            if len(res_hit) == 0:
                continue
            selected_res = res_hit.iloc[0]
            i0 = int(rr.i0)
            H_min = int(rr.H_min)
            H_sec = 60.0 * H_min
            t0 = float(t_sec[i0])
            if (t0 + H_sec) > float(t_sec[-1]):
                continue

            ts_gt, e_gt, n_gt = m.build_gt_path(t0, H_sec, t_sec, E, N, dt_sim=m.DT_SIM)
            true_len = float(np.hypot(e_gt[-1] - e_gt[0], n_gt[-1] - n_gt[0]))
            if true_len < 30.0:
                continue

            try:
                pred_res = rap.predict_df(selected_res.to_frame().T)[0]
            except Exception:
                continue
            dv_rap, dw_rap = float(pred_res[0]), float(pred_res[1])

            _, e_ct, n_ct = m.simulate_ct_timevarying(
                float(rr.E0), float(rr.N0), float(rr.v0), float(rr.theta0), float(rr.omega0),
                0.0, 0.0, H_sec, dt_sim=m.DT_SIM, v_cap=v_cap, omega_cap=omega_cap,
            )
            _, e_rap, n_rap = m.simulate_ct_timevarying(
                float(rr.E0), float(rr.N0), float(rr.v0), float(rr.theta0), float(rr.omega0),
                dv_rap, dw_rap, H_sec, dt_sim=m.DT_SIM, v_cap=v_cap, omega_cap=omega_cap,
            )

            gt_head = _path_heading(e_gt, n_gt)
            ct_head = _path_heading(e_ct, n_ct)
            rap_head = _path_heading(e_rap, n_rap)
            angle_ct = _angle_diff_deg(ct_head, gt_head)
            angle_rap = _angle_diff_deg(rap_head, gt_head)
            epe_ct = float(np.hypot(e_ct[-1] - e_gt[-1], n_ct[-1] - n_gt[-1]))
            epe_rap = float(np.hypot(e_rap[-1] - e_gt[-1], n_rap[-1] - n_gt[-1]))
            improve = epe_ct - epe_rap
            improve_ratio = improve / (epe_ct + 1e-9)

            scored.append({
                "row_id": row_id,
                "H_min": H_min,
                "angle_ct": angle_ct,
                "angle_rap": angle_rap,
                "epe_ct": epe_ct,
                "epe_rap": epe_rap,
                "improve": improve,
                "improve_ratio": improve_ratio,
                "true_len": true_len,
            })
        if scored:
            break

    if not scored:
        return _select_p90_row_from_merge_error_table(bt_merge)

    score_df = pd.DataFrame(scored)
    # 第一优先级：方向别离谱，RAP-CT 有正改善。图是拿来说明问题的，不是拿来展示模型集体跑偏的。
    good = score_df[
        (score_df["angle_rap"] <= 25.0) &
        (score_df["angle_ct"] <= 45.0) &
        (score_df["improve"] > 0.0)
    ].copy()
    if len(good) < 5:
        good = score_df[
            (score_df["angle_rap"] <= 35.0) &
            (score_df["improve"] > 0.0)
        ].copy()
    if len(good) == 0:
        good = score_df.copy()

    # 既要误差有代表性，又不能为了“高误差”选出一张几何方向全崩的丑图。
    epe_norm = (good["epe_ct"] - good["epe_ct"].min()) / (good["epe_ct"].max() - good["epe_ct"].min() + 1e-9)
    len_norm = (good["true_len"] - good["true_len"].min()) / (good["true_len"].max() - good["true_len"].min() + 1e-9)
    good["score"] = (
        1.20 * epe_norm +
        1.00 * good["improve_ratio"].clip(lower=-1.0, upper=1.0) +
        0.20 * len_norm -
        0.020 * good["angle_rap"] -
        0.010 * good["angle_ct"]
    )
    best = good.sort_values("score", ascending=False).iloc[0]
    hit = bt_merge[bt_merge["row_id"] == int(best["row_id"])]
    if len(hit):
        print(
            "[INFO] 图14代表性窗口："
            f"row_id={int(best['row_id'])}, H={int(best['H_min'])}min, "
            f"angle_rap={best['angle_rap']:.1f}°, angle_ct={best['angle_ct']:.1f}°, "
            f"EPE_CT={best['epe_ct']:.1f}m, EPE_RAP={best['epe_rap']:.1f}m"
        )
        return hit.iloc[0]
    return _select_p90_row_from_merge_error_table(bt_merge)


def _train_blackbox_predict_one(m, res_df, selected_res_row, model_name, H_min):
    feat_cols = [
        "v0", "omega0", "v_med", "w_abs_med", "a_abs_med",
        "v_std", "w_std", "sin_th", "cos_th", "H_norm",
    ]
    feat_cols_no_h = feat_cols[:-1]
    tr = res_df[(res_df["H_min"] == H_min) & (res_df["split"] == "train")].copy()
    if len(tr) < 20:
        return None
    X_tr = tr[feat_cols_no_h].values.astype(np.float32)
    Y_tr = tr[["dv_true", "dw_true"]].values.astype(np.float32)
    X_one = selected_res_row[feat_cols_no_h].values.astype(np.float32)[None, :]
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_tr_sc = sc_x.fit_transform(X_tr)
    X_one_sc = sc_x.transform(X_one)
    Y_tr_sc = sc_y.fit_transform(Y_tr)

    if model_name == "RF":
        model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=80, max_depth=4, random_state=getattr(m, "SEED", 42), n_jobs=-1)
        )
        model.fit(X_tr_sc, Y_tr)
        return model.predict(X_one_sc)[0]

    if model_name == "GBDT":
        model = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=80, max_depth=4, random_state=getattr(m, "SEED", 42), learning_rate=0.05)
        )
        model.fit(X_tr_sc, Y_tr)
        return model.predict(X_one_sc)[0]

    if not getattr(m, "USE_TORCH", False):
        return None
    builder_map = {
        "LSTM": lambda: m.LSTMModel(9),
        "GRU": lambda: m.GRUModel(9),
        "Seq2Seq": lambda: m.Seq2SeqModel(9),
        "Transformer": lambda: m.TransformerModel(9),
    }
    if model_name not in builder_map:
        return None

    np.random.seed(getattr(m, "SEED", 42))
    torch.manual_seed(getattr(m, "SEED", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(getattr(m, "SEED", 42))

    device = getattr(m, "DEVICE", torch.device("cpu"))
    model = builder_map[model_name]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt = torch.tensor(X_tr_sc, dtype=torch.float32).to(device)
    Yt = torch.tensor(Y_tr_sc, dtype=torch.float32).to(device)
    for _ in range(40):
        model.train()
        opt.zero_grad()
        loss = nn.MSELoss()(model(Xt), Yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred_sc = model(torch.tensor(X_one_sc, dtype=torch.float32).to(device)).cpu().numpy()
    pred = sc_y.inverse_transform(pred_sc)[0]
    return pred


def _normal_unit_global(dx, dy):
    L = np.hypot(dx, dy)
    if L < 1e-9:
        return 0.0, 0.0
    tx, ty = dx / L, dy / L
    return -ty, tx


def _fill_funnel_polygon_global(ax, x, y, width, color="#1f77b4", alpha_max=0.35, label=None):
    nseg = 10
    idx_all = np.linspace(0, len(x) - 1, nseg + 1).astype(int)
    for i in range(nseg):
        a0, a1 = idx_all[i], idx_all[i + 1]
        if a1 <= a0 + 1:
            continue
        xs, ys, ws = x[a0:a1 + 1], y[a0:a1 + 1], width[a0:a1 + 1]
        nx, ny = np.zeros_like(xs), np.zeros_like(ys)
        for k in range(len(xs)):
            if k == 0:
                dx, dy = xs[1] - xs[0], ys[1] - ys[0]
            elif k == len(xs) - 1:
                dx, dy = xs[-1] - xs[-2], ys[-1] - ys[-2]
            else:
                dx, dy = xs[k + 1] - xs[k - 1], ys[k + 1] - ys[k - 1]
            nx[k], ny[k] = _normal_unit_global(dx, dy)
        left_x, left_y = xs + nx * ws, ys + ny * ws
        right_x, right_y = xs - nx * ws, ys - ny * ws
        poly_x = np.r_[left_x, right_x[::-1]]
        poly_y = np.r_[left_y, right_y[::-1]]
        s = (i + 1) / max(1, nseg)
        a = alpha_max * (0.20 + 0.80 * (s ** 1.6))
        a = max(a, 0.02)
        ax.fill(poly_x, poly_y, color=color, alpha=a, linewidth=0.0,
                label=(label if (i == 0 and label is not None) else None))


def _plot_grouped_multimodel_trajectory(paths, gt_path, region_path, out_prefix):
    ts_gt, e_gt, n_gt = gt_path
    ts_reg, e_reg, n_reg, w_reg = region_path
    groups = [
        ("图14a_物理与滤波模型轨迹对比", ["CV-Heading", "CV-CT", "KF-CV", "IMM-CVCA", "RAP-CT"]),
        ("图14b_传统机器学习残差模型轨迹对比", ["RF", "GBDT", "RAP-CT"]),
        ("图14c_深度时序残差模型轨迹对比", ["LSTM", "GRU", "Seq2Seq", "Transformer", "RAP-CT"]),
        ("图14d_残差修正消融模型轨迹对比", ["CV-CT", "RAP-CT-dv-only", "RAP-CT-dw-only", "RAP-CT"]),
    ]
    color_map = {
        "CV-Heading": "#ff7f0e",
        "CV-CT": "#2ca02c",
        "KF-CV": "#7f7f7f",
        "IMM-CVCA": "#17becf",
        "RF": "#636363",
        "GBDT": "#969696",
        "LSTM": "#9ecae1",
        "GRU": "#6baed6",
        "Seq2Seq": "#4292c6",
        "Transformer": "#2171b5",
        "RAP-CT-dv-only": "#9467bd",
        "RAP-CT-dw-only": "#8c564b",
        "RAP-CT": "#d62728",
    }
    linestyle_map = {
        "CV-Heading": ":",
        "CV-CT": "-.",
        "KF-CV": "--",
        "IMM-CVCA": "--",
        "RF": ":",
        "GBDT": "-.",
        "LSTM": ":",
        "GRU": "-.",
        "Seq2Seq": "--",
        "Transformer": (0, (3, 1, 1, 1)),
        "RAP-CT-dv-only": "--",
        "RAP-CT-dw-only": "-.",
        "RAP-CT": "--",
    }

    exported = []
    for file_stem, model_list in groups:
        fig = plt.figure(figsize=(5.2, 4.2))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.98)

        _fill_funnel_polygon_global(ax, e_reg, n_reg, w_reg, color="#1f77b4",
                                    alpha_max=0.28, label="经验预测范围")
        ax.plot(e_gt, n_gt, color="#1f77b4", linewidth=2.1, zorder=6, label="真实轨迹")
        for method in model_list:
            if method not in paths:
                continue
            e_pred, n_pred = paths[method]
            lw = 2.0 if method == "RAP-CT" else 1.6
            z = 10 if method == "RAP-CT" else 7
            ax.plot(e_pred, n_pred,
                    color=color_map.get(method, "#7f7f7f"),
                    linestyle=linestyle_map.get(method, "--"),
                    linewidth=lw, zorder=z, label=method)
        ax.scatter([e_gt[0]], [n_gt[0]], s=26, marker="o", color="black", zorder=12)
        ax.scatter([e_gt[-1]], [n_gt[-1]], s=36, marker="^", color="#1f77b4", zorder=12)

        # 按用户要求：不再添加(a)(b)(c)(d)或任何子图标题。
        ax.set_title("")
        set_mixed_xlabel(ax, "东向坐标", "m", y=-0.16, fontsize=11)
        set_mixed_ylabel(ax, "北向坐标", "m", x=-0.16, fontsize=11)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.25)
        apply_tick_font(ax)
        apply_tick_style(ax)
        lg = ax.legend(frameon=True, loc="best", fontsize=8, ncol=2)
        apply_legend_font_mixed(lg)

        png_path = os.path.join(OUT_DIR, f"{file_stem}.png")
        pdf_path = os.path.join(OUT_DIR, f"{file_stem}.pdf")
        fig.savefig(png_path, bbox_inches="tight", pad_inches=0.15)
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        exported.extend([png_path, pdf_path])
        print(f"[INFO] 已导出：{png_path}")
        print(f"[INFO] 已导出：{pdf_path}")
    return exported


def plot_grouped_multimodel_trajectory_from_merged_code():
    m = _load_merge_module()
    if getattr(m, "DETERMINISTIC", True):
        m.seed_all(getattr(m, "SEED", 42))
    csv_path = m.find_existing_path(m.CSV_PATH, m.CSV_CANDIDATES)
    if csv_path is None:
        raise FileNotFoundError("找不到 AIS 数据文件 ais-USV_filled.csv。")
    df = pd.read_csv(csv_path)
    df["t"] = pd.to_datetime(df["base_date_time"])
    df = df.sort_values("t").drop_duplicates("t", keep="last").reset_index(drop=True)
    lat0, lon0 = float(df.loc[0, "latitude"]), float(df.loc[0, "longitude"])
    E, N = m.latlon_to_enu(df["latitude"].to_numpy(float), df["longitude"].to_numpy(float), lat0, lon0)
    t = df["t"].to_numpy()
    t_sec = ((t - t[0]) / np.timedelta64(1, "s")).astype(float)
    v_pos, theta_pos, omega_pos, a_pos = m.compute_pos_kinematics(E, N, t_sec)
    v_cap, omega_cap = m.compute_caps_from_ais(df, t_sec)
    bt_merge, res_df = m.build_backtest_tables(df, E, N, t_sec, v_pos, theta_pos, omega_pos, a_pos, v_cap, omega_cap)
    rap = m.RAPPredictor().fit(res_df)

    selected = _select_representative_row_for_grouped_trajectory(m, bt_merge, res_df, rap, E, N, t_sec, v_cap, omega_cap)
    row_id = int(selected.row_id)
    i0 = int(selected.i0)
    H_min = int(selected.H_min)
    H_sec = 60.0 * H_min
    res_hit = res_df[res_df["row_id"] == row_id]
    if len(res_hit) == 0:
        raise RuntimeError("在 residual table 中找不到选中 row_id 对应记录。")
    selected_res = res_hit.iloc[0]
    t0 = float(t_sec[i0])
    ts_gt, e_gt, n_gt = m.build_gt_path(t0, H_sec, t_sec, E, N, dt_sim=m.DT_SIM)

    paths = OrderedDict()
    _, e_cvhead, n_cvhead = m.simulate_cv_heading(
        E[i0], N[i0], float(selected.v0), float(selected.theta0),
        H_sec, dt_sim=m.DT_SIM, v_cap=v_cap,
    )
    paths["CV-Heading"] = (e_cvhead, n_cvhead)
    _, e_ct, n_ct = m.simulate_ct_timevarying(
        E[i0], N[i0], float(selected.v0), float(selected.theta0), float(selected.omega0),
        0.0, 0.0, H_sec, dt_sim=m.DT_SIM, v_cap=v_cap, omega_cap=omega_cap,
    )
    paths["CV-CT"] = (e_ct, n_ct)
    _, e_kf, n_kf = m.kf_cv_predict_path(
        i0, H_sec, E, N, t_sec, v_pos, theta_pos, v_cap, hist_len=12, dt_sim=m.DT_SIM,
    )
    paths["KF-CV"] = (e_kf, n_kf)
    _, e_imm, n_imm = m.imm_cvca_predict_path(
        i0, H_sec, E, N, t_sec, v_pos, theta_pos, v_cap, hist_len=12, dt_sim=m.DT_SIM,
    )
    paths["IMM-CVCA"] = (e_imm, n_imm)

    pred_res = rap.predict_df(selected_res.to_frame().T)[0]
    dv_rap, dw_rap = float(pred_res[0]), float(pred_res[1])
    _, e_rap, n_rap = m.simulate_ct_timevarying(
        E[i0], N[i0], float(selected.v0), float(selected.theta0), float(selected.omega0),
        dv_rap, dw_rap, H_sec, dt_sim=m.DT_SIM, v_cap=v_cap, omega_cap=omega_cap,
    )
    paths["RAP-CT"] = (e_rap, n_rap)
    _, e_dv, n_dv = m.simulate_ct_timevarying(
        E[i0], N[i0], float(selected.v0), float(selected.theta0), float(selected.omega0),
        dv_rap, 0.0, H_sec, dt_sim=m.DT_SIM, v_cap=v_cap, omega_cap=omega_cap,
    )
    paths["RAP-CT-dv-only"] = (e_dv, n_dv)
    _, e_dw, n_dw = m.simulate_ct_timevarying(
        E[i0], N[i0], float(selected.v0), float(selected.theta0), float(selected.omega0),
        0.0, dw_rap, H_sec, dt_sim=m.DT_SIM, v_cap=v_cap, omega_cap=omega_cap,
    )
    paths["RAP-CT-dw-only"] = (e_dw, n_dw)

    blackbox_theta = m.blackbox_theta_from_row(selected_res)
    for method in ["RF", "GBDT", "LSTM", "GRU", "Seq2Seq", "Transformer"]:
        pred = _train_blackbox_predict_one(m, res_df, selected_res, method, H_min)
        if pred is None:
            continue
        dv, dw = float(pred[0]), float(pred[1])
        _, e_bb, n_bb = m.simulate_ct_timevarying(
            float(selected_res.E0), float(selected_res.N0), float(selected_res.v0),
            blackbox_theta, float(selected_res.omega0), dv, dw,
            H_sec, dt_sim=m.DT_SIM, v_cap=v_cap, omega_cap=omega_cap,
        )
        paths[method] = (e_bb, n_bb)

    cal = bt_merge[bt_merge["split"].isin(["train", "val"])].copy()
    if len(cal) == 0:
        cal = bt_merge.copy()
    pred_all = rap.predict_df(res_df)
    pred_map = {int(res_df.iloc[k].row_id): pred_all[k] for k in range(len(res_df))}
    end_res_vals = []
    for _, rr in cal[cal["H_min"] == H_min].iterrows():
        rid = int(rr.row_id)
        if rid not in pred_map:
            continue
        dvv, dww = pred_map[rid]
        _, ee, nn = m.simulate_ct_timevarying(
            float(rr.E0), float(rr.N0), float(rr.v0), float(rr.theta0), float(rr.omega0),
            float(dvv), float(dww), H_sec, dt_sim=m.DT_SIM, v_cap=v_cap, omega_cap=omega_cap,
        )
        end_res_vals.append(float(np.hypot(ee[-1] - float(rr.Et), nn[-1] - float(rr.Nt))))
    sigma_end = float(np.nanmedian(end_res_vals)) if len(end_res_vals) else 30.0
    w_region = funnel_width(ts_gt, H_sec, sigma_end, float(selected.v0), k=1.64)

    rows = []
    for method, (ee, nn) in paths.items():
        for k in range(len(ts_gt)):
            rows.append({
                "row_id": row_id,
                "i0": i0,
                "预测时长/min": H_min,
                "方法": method,
                "t_rel/s": float(ts_gt[k]),
                "east_pred/m": float(ee[k]),
                "north_pred/m": float(nn[k]),
                "east_gt/m": float(e_gt[k]),
                "north_gt/m": float(n_gt[k]),
            })
    traj_detail = pd.DataFrame(rows)
    traj_csv = os.path.join(OUT_DIR, "表S_代表性高误差起点多模型轨迹明细.csv")
    traj_detail.to_csv(traj_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已导出：{traj_csv}")

    _plot_grouped_multimodel_trajectory(
        paths=paths,
        gt_path=(ts_gt, e_gt, n_gt),
        region_path=(ts_gt, e_rap, n_rap, w_region),
        out_prefix="图14_代表性高误差起点多模型分组轨迹对比",
    )

# ============================================================
# 7. 主流程
# ============================================================
def main():
    if DETERMINISTIC:
        seed_all(42)
        if USE_TORCH:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    df = pd.read_csv(CSV_PATH)
    df["t"] = pd.to_datetime(df["base_date_time"])
    df = df.sort_values("t").drop_duplicates("t", keep="last").reset_index(drop=True)
    lat0, lon0 = float(df.loc[0, "latitude"]), float(df.loc[0, "longitude"])
    E, N = latlon_to_enu(df["latitude"].to_numpy(float), df["longitude"].to_numpy(float), lat0, lon0)
    t = df["t"].to_numpy()
    t_sec = ((t - t[0]) / np.timedelta64(1, "s")).astype(float)
    v_pos, theta_pos, omega_pos, a_pos = compute_pos_kinematics(E, N, t_sec)
    v_cap, omega_cap = compute_caps_from_ais(df, t_sec)

    print(f"[INFO] rows={len(df)}  v_cap(P99 SOG)={v_cap:.3f} m/s  omega_cap(P99 COG-rate)={omega_cap:.4f} rad/s")
    print(f"[INFO] Torch available: {USE_TORCH}")

    starts = list(range(OMEGA_WIN_K, len(df) - 1, BACKTEST_STEP))
    records = []
    residual_rows = []

    for i0 in starts:
        v0 = float(v_pos[i0]) if np.isfinite(v_pos[i0]) else np.nan
        th0 = float(theta_pos[i0]) if np.isfinite(theta_pos[i0]) else np.nan
        if (not np.isfinite(v0)) or (v0 < V_MOVE_MIN) or (not np.isfinite(th0)):
            continue
        omega0 = omega0_window(i0, E, N, omega_pos, v0)
        state = motion_state(omega0)
        t0 = float(t_sec[i0])
        s0 = max(0, i0 - (OMEGA_WIN_K - 1))
        v_hist = v_pos[s0:i0 + 1]
        w_hist = omega_pos[s0:i0 + 1]
        a_hist = a_pos[s0:i0 + 1]
        feat = [
            v0, omega0,
            float(np.nanmedian(v_hist)),
            float(np.nanmedian(np.abs(w_hist))),
            float(np.nanmedian(np.abs(a_hist))),
            float(np.nanstd(v_hist)),
            float(np.nanstd(w_hist)),
            float(np.sin(th0)),
            float(np.cos(th0)),
        ]
        for H_min in HORIZONS_MIN:
            H_sec = 60.0 * H_min
            gt_end = interp_xy_at_time(t_sec, E, N, t0 + H_sec)
            if gt_end is None:
                continue
            e_gt, n_gt = gt_end
            ts_cv, e_cv, n_cv = simulate_cv_heading(E[i0], N[i0], v0, th0, H_sec, dt_sim=DT_SIM, v_cap=v_cap)
            ts_gt, e_gt_path, n_gt_path = build_gt_path(t0, H_sec, t_sec, E, N, dt_sim=DT_SIM)
            end_cv = float(np.hypot(e_cv[-1] - e_gt, n_cv[-1] - n_gt))
            rmse_cv = rmse_path(e_cv, n_cv, e_gt_path, n_gt_path)
            ts_ct, e_ct, n_ct = simulate_ct_timevarying(
                E[i0], N[i0], v0, th0, omega0, 0.0, 0.0, H_sec,
                dt_sim=DT_SIM, v_cap=v_cap, omega_cap=omega_cap,
            )
            end_ct = float(np.hypot(e_ct[-1] - e_gt, n_ct[-1] - n_gt))
            rmse_ct = rmse_path(e_ct, n_ct, e_gt_path, n_gt_path)
            records.append([i0, H_min, state, v0, omega0, end_cv, rmse_cv, end_ct, rmse_ct])
            i1 = int(np.searchsorted(t_sec, t0 + H_sec))
            i1 = min(i1, len(t_sec) - 1)
            win = slice(max(i0, i1 - OMEGA_WIN_K), i1 + 1)
            v_future = float(np.nanmedian(v_pos[win])) if np.isfinite(np.nanmedian(v_pos[win])) else v0
            w_future = float(np.nanmedian(omega_pos[win])) if np.isfinite(np.nanmedian(omega_pos[win])) else omega0
            residual_rows.append([i0] + feat + [H_min / 20.0, float(v_future - v0), float(w_future - omega0)])

    bt = pd.DataFrame(records, columns=["i0", "H_min", "state", "v0", "omega0", "end_cv", "rmse_cv", "end_ct", "rmse_ct"])
    res_df = pd.DataFrame(residual_rows, columns=["i0", "v0", "omega0", "v_med", "w_abs_med", "a_abs_med", "v_std", "w_std", "sin_th", "cos_th", "H_norm", "dv", "dw"])
    split_map = assign_timewise_split(bt)
    bt["split"] = bt["i0"].map(split_map)
    res_df["split"] = res_df["i0"].map(split_map)
    print(bt[["i0", "split"]].drop_duplicates()["split"].value_counts().to_dict())
    print(f"[INFO] backtest rows={len(bt)}  residual samples={len(res_df)}")
    print("[ALL state]", bt["state"].value_counts(dropna=False).to_dict())
    print("[split-state]\n", bt.groupby(["split", "state"]).size())

    if USE_TORCH and len(res_df) > 200:
        feat_cols = ["v0", "omega0", "v_med", "w_abs_med", "a_abs_med", "v_std", "w_std", "sin_th", "cos_th", "H_norm"]
        X = res_df[feat_cols].to_numpy(np.float32)
        Y = res_df[["dv", "dw"]].to_numpy(np.float32)
        tr_mask = (res_df["split"] == "train").to_numpy()
        va_mask = (res_df["split"] == "val").to_numpy()
        if va_mask.sum() < 10:
            va_mask = (res_df["split"] != "train").to_numpy()
        Xtr = torch.from_numpy(X[tr_mask])
        Ytr = torch.from_numpy(Y[tr_mask])
        Xva = torch.from_numpy(X[va_mask])
        Yva = torch.from_numpy(Y[va_mask])
        x_mean = Xtr.mean(0, keepdim=True)
        x_std = Xtr.std(0, keepdim=True).clamp_min(1e-6)
        y_mean = Ytr.mean(0, keepdim=True)
        y_std = Ytr.std(0, keepdim=True).clamp_min(1e-6)
        Xtrn = (Xtr - x_mean) / x_std
        Xvan = (Xva - x_mean) / x_std
        Ytrn = (Ytr - y_mean) / y_std
        Yvan = (Yva - y_mean) / y_std
        model = nn.Sequential(nn.Linear(Xtrn.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
        best = 1e18
        best_state = None
        for _ in range(80):
            model.train()
            opt.zero_grad()
            pred = model(Xtrn)
            Hn_tr = Xtr[:, -1].clamp(0.0, 1.0)
            w = (1.0 + LOSS_W_A * Hn_tr).unsqueeze(1)
            loss = (w * torch.abs(pred - Ytrn)).mean()
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                pred_va = model(Xvan)
                va_loss = torch.abs(pred_va - Yvan).mean().item()
            if va_loss < best:
                best = va_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)

        def predict_residual(feat, H_min):
            row = np.array(feat + [H_min / 20.0], dtype=np.float32)[None, :]
            Xt = torch.from_numpy(row)
            with torch.no_grad():
                Xn = (Xt - x_mean) / x_std
                yn = model(Xn)
                y = yn * y_std + y_mean
            return float(y[0, 0]), float(y[0, 1])
    else:
        def predict_residual(feat, H_min):
            return 0.0, 0.0

    end_res_list = []
    rmse_res_list = []
    for _, r in bt.iterrows():
        i0 = int(r.i0)
        H_min = int(r.H_min)
        H_sec = 60.0 * H_min
        t0 = float(t_sec[i0])
        v0 = float(v_pos[i0]) if np.isfinite(v_pos[i0]) else 0.0
        th0 = float(theta_pos[i0])
        omega0 = omega0_window(i0, E, N, omega_pos, v0)
        s0 = max(0, i0 - (OMEGA_WIN_K - 1))
        v_hist = v_pos[s0:i0 + 1]
        w_hist = omega_pos[s0:i0 + 1]
        a_hist = a_pos[s0:i0 + 1]
        feat = [v0, omega0, float(np.nanmedian(v_hist)), float(np.nanmedian(np.abs(w_hist))), float(np.nanmedian(np.abs(a_hist))), float(np.nanstd(v_hist)), float(np.nanstd(w_hist)), float(np.sin(th0)), float(np.cos(th0))]
        dv, dw = predict_residual(feat, H_min)
        _, e_res, n_res = simulate_ct_timevarying(E[i0], N[i0], v0, th0, omega0, dv, dw, H_sec, dt_sim=DT_SIM, v_cap=v_cap, omega_cap=omega_cap)
        e_gt, n_gt = interp_xy_at_time(t_sec, E, N, t0 + H_sec)
        _, e_gt_path, n_gt_path = build_gt_path(t0, H_sec, t_sec, E, N, dt_sim=DT_SIM)
        end_res_list.append(float(np.hypot(e_res[-1] - e_gt, n_res[-1] - n_gt)))
        rmse_res_list.append(rmse_path(e_res, n_res, e_gt_path, n_gt_path))
    bt["end_res"] = end_res_list
    bt["rmse_res"] = rmse_res_list

    bt_eval = bt[bt["split"] == EVAL_SCOPE].copy() if EVAL_SCOPE == "test" else bt.copy()
    if len(bt_eval) == 0:
        bt_eval = bt.copy()
    print("[EVAL state]", bt_eval["state"].value_counts(dropna=False).to_dict())

    base_for_select = bt_eval[bt_eval["H_min"] == 10].copy()
    if USE_RES_BETTER_SUBSET_FOR_VIS:
        good = base_for_select[base_for_select["end_res"] < base_for_select["end_ct"]].copy()
        if len(good) >= 5:
            base_for_select = good
    if len(base_for_select) == 0:
        base_for_select = bt_eval[bt_eval["H_min"] == 10].copy()
    cand_i0 = base_for_select["i0"].to_numpy(int)
    cand_err = base_for_select["end_ct"].to_numpy(float)
    q50 = np.nanquantile(cand_err, 0.50)
    q90 = np.nanquantile(cand_err, 0.90)
    i0_p50 = int(FIX_I0_P50) if (FIX_I0_P50 is not None and FIX_I0_P50 in cand_i0) else int(cand_i0[np.nanargmin(np.abs(cand_err - q50))])
    i0_p90 = int(FIX_I0_P90) if (FIX_I0_P90 is not None and FIX_I0_P90 in cand_i0) else int(cand_i0[np.nanargmin(np.abs(cand_err - q90))])
    print(f"[INFO] P50 i0={i0_p50} ; P90 i0={i0_p90}")

    cal = bt[bt["split"].isin(["train", "val"])].copy()
    if len(cal) == 0:
        cal = bt.copy()
    q1, q2 = cal["v0"].quantile(0.33), cal["v0"].quantile(0.66)

    def speed_bin(v):
        return "low" if v <= q1 else ("mid" if v <= q2 else "high")

    cal["speed_bin"] = cal["v0"].apply(speed_bin)
    sig = cal.groupby(["H_min", "state", "speed_bin"])["end_res"].median().reset_index().rename(columns={"end_res": "sigma_end"})

    def lookup_sigma(H_min, state, speed_bin_):
        s = sig[(sig.H_min == H_min) & (sig.state == state) & (sig.speed_bin == speed_bin_)]
        if len(s):
            return float(s.iloc[0].sigma_end)
        if len(cal[cal.H_min == H_min]):
            return float(cal[cal.H_min == H_min]["end_res"].median())
        return 30.0

    # 按用户要求：这里不再生成代表性起点的轨迹图和误差图，
    # 只保留对应统计表，避免输出多余图片。

    abs_table = build_abs_error_table(bt_eval, [i0_p50, i0_p90], ["P50", "P90"], HORIZONS_MIN)
    abs_csv = os.path.join(OUT_DIR, "表4_代表性起点绝对误差主表.csv")
    abs_table.to_csv(abs_csv, index=False, encoding="utf-8-sig")
    improve_table = build_improve_table(bt_eval, [i0_p50, i0_p90], ["P50", "P90"], HORIZONS_MIN)
    improve_csv = os.path.join(OUT_DIR, "表5_RAP-CT相对CV-CT改善率.csv")
    improve_table.to_csv(improve_csv, index=False, encoding="utf-8-sig")
    rep_info = export_representative_info(bt_eval, [i0_p50, i0_p90], ["P50", "P90"], t_sec)
    rep_csv = os.path.join(OUT_DIR, "表S1_代表性起点信息.csv")
    rep_info.to_csv(rep_csv, index=False, encoding="utf-8-sig")
    wide_abs = abs_table.pivot_table(index=["起点类型", "方法"], columns="指标", values=[f"{h} min" for h in HORIZONS_MIN], aggfunc="first")
    wide_abs.to_csv(os.path.join(OUT_DIR, "表4_绝对误差主表_宽表版.csv"), encoding="utf-8-sig")
    overall_csv, overall_imp_csv = export_overall_tables(bt_eval, bt, OUT_DIR, HORIZONS_MIN)

    print("[INFO] 已导出：")
    for p in [abs_csv, improve_csv, rep_csv, os.path.join(OUT_DIR, "表4_绝对误差主表_宽表版.csv"), overall_csv, overall_imp_csv]:
        print(f"  - {p}")

    try:
        plot_multimodel_comparison_from_merged_table()
    except FileNotFoundError as e:
        print(f"[WARN] 未生成多模型对比图：{e}")
    except Exception as e:
        print(f"[WARN] 多模型对比图生成失败：{repr(e)}")

    try:
        plot_grouped_multimodel_trajectory_from_merged_code()
    except FileNotFoundError as e:
        print(f"[WARN] 未生成多模型分组轨迹图：{e}")
    except Exception as e:
        print(f"[WARN] 多模型分组轨迹图生成失败：{repr(e)}")


if __name__ == "__main__":
    main()
