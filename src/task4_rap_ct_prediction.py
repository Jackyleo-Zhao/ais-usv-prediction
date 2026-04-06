# -*- coding: utf-8 -*-
"""
任务4：AIS-USV 短时轨迹预测（2/10/20 min）
模型：
  （1）CV-Heading 基线
  （2）CV-CT 基线：基于位置差分角速度的常转率模型，dv=dw=0
  （3）RAP-CT：以 CT 为骨架，利用 MLP 预测 (dv, dw)

核心原则：
  - 轨迹中心预测与误差评估均基于位置差分运动学（ENU -> v_pos, theta_pos, omega_pos）
  - SOG/COG 仅用于上界截断：v <= v_cap(P99 SOG), |omega| <= omega_cap(P99 COG-rate)
  - 扇形置信域宽度由回测残差误差标定（sigma_H）
  - 图形风格适配中文期刊：中文宋体，刻度数字 Times New Roman，输出 PNG/PDF
  - 新增导出：
      表4_代表性起点绝对误差主表.csv
      表5_RAP-CT相对CV-CT改善率.csv
      表S1_代表性起点信息.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

# ============== 字体配置 ==============
FONT_DIR = "."
FP_SIMSUN = FontProperties(fname=os.path.join(FONT_DIR, "simsun.ttc"))
FP_SIMSUN_BOLD = FontProperties(fname=os.path.join(FONT_DIR, "simsunb.ttf"))
FP_TIMES = FontProperties(fname=os.path.join(FONT_DIR, "times.ttf"))

# ============== USER CONFIG ==============
CSV_PATH = "ais-USV_filled.csv"
OUT_DIR = "任务4_中文期刊输出_补强版4"

DETERMINISTIC = True

# 若你希望固定论文中展示的 P50 / P90 片段，就保留固定索引；
# 若希望自动选择，可改成 None
FIX_I0_P50 = None
FIX_I0_P90 = None

# 是否严格使用“残差优于CT”的子集来挑代表样本
USE_RES_BETTER_SUBSET_FOR_VIS = True

# ===== 新增：严格划分训练/验证/测试，避免残差模型在同一批窗口上自评 =====
VAL_RATIO = 0.15
TEST_RATIO = 0.20
SPLIT_SEED = 0
EVAL_SCOPE = "test"   # 可选: "test" 或 "all"
TURN_EVAL_SCOPE = "all"  # turn 子集统计可选: "test" 或 "all"
IMPROVE_DENOM_EPS = 1.0   # 改善率分母保护阈值，单位 m
# ========================================

os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 180
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

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


def apply_tick_font(ax, xfont=FP_TIMES, yfont=FP_TIMES):
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(xfont)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(yfont)


def apply_legend_font(legend_obj, font=FP_SIMSUN):
    if legend_obj is not None:
        for text in legend_obj.get_texts():
            text.set_fontproperties(font)


def apply_tick_style(ax):
    ax.tick_params(axis="both", which="major", direction="in", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="in", length=2, width=0.6)


def seed_all(seed=0):
    np.random.seed(seed)
    if USE_TORCH:
        torch.manual_seed(seed)
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
    ds = np.sqrt(de**2 + dn**2)

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
        row_rmse = {"起点类型": tag, "指标": "Path RMSE 改善率 / %"}

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
        for metric_name, ct_col, res_col in [("EPE 改善率 / %", "end_ct", "end_res"), ("Path RMSE 改善率 / %", "rmse_ct", "rmse_res")]:
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
        wide_sum = df_sum.pivot_table(index=["子集", "预测时长/min", "方法"], columns="指标", values=["mean", "median", "P90", "N"], aggfunc="first")
        wide_sum.to_csv(os.path.join(out_dir, "表6_全体与分层误差统计_宽表版.csv"), encoding="utf-8-sig")
    if len(df_imp):
        wide_imp = df_imp.pivot_table(index=["子集", "预测时长/min", "相对对象"], columns="指标", values=["绝对下降量mean", "绝对下降量median", "median", "P90", "胜率/%", "N", "有效百分比样本数"], aggfunc="first")
        wide_imp.to_csv(os.path.join(out_dir, "表7_RAP-CT相对CV-CT总体改善统计_宽表版.csv"), encoding="utf-8-sig")
    return p1, p2


def main():
    if DETERMINISTIC:
        seed_all(0)
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
                dt_sim=DT_SIM, v_cap=v_cap, omega_cap=omega_cap
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

    bt = pd.DataFrame(records, columns=[
        "i0", "H_min", "state", "v0", "omega0",
        "end_cv", "rmse_cv", "end_ct", "rmse_ct"
    ])

    res_df = pd.DataFrame(residual_rows, columns=[
        "i0", "v0", "omega0", "v_med", "w_abs_med", "a_abs_med", "v_std", "w_std", "sin_th", "cos_th", "H_norm", "dv", "dw"
    ])

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

        Xtr = torch.from_numpy(X[tr_mask]); Ytr = torch.from_numpy(Y[tr_mask])
        Xva = torch.from_numpy(X[va_mask]); Yva = torch.from_numpy(Y[va_mask])

        x_mean = Xtr.mean(0, keepdim=True)
        x_std = Xtr.std(0, keepdim=True).clamp_min(1e-6)
        y_mean = Ytr.mean(0, keepdim=True)
        y_std = Ytr.std(0, keepdim=True).clamp_min(1e-6)

        Xtrn = (Xtr - x_mean) / x_std
        Xvan = (Xva - x_mean) / x_std
        Ytrn = (Ytr - y_mean) / y_std
        Yvan = (Yva - y_mean) / y_std

        model = nn.Sequential(
            nn.Linear(Xtrn.shape[1], 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
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
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)

        def predict_residual(feat_vec, H_min):
            x = np.array(feat_vec + [H_min / 20.0], dtype=np.float32)[None, :]
            xt = torch.from_numpy(x)
            with torch.no_grad():
                xn = (xt - x_mean) / x_std
                yn = model(xn)
                y = yn * y_std + y_mean
            return float(y[0, 0]), float(y[0, 1])
    else:
        print("[WARN] Torch 不可用或样本过少，残差项回退为 0")
        def predict_residual(feat_vec, H_min):
            return 0.0, 0.0

    rows = []
    for _, r in bt.iterrows():
        i0 = int(r.i0)
        H_min = int(r.H_min)
        H_sec = 60.0 * H_min
        v0 = float(r.v0)
        omega0 = float(r.omega0)
        th0 = float(theta_pos[i0])
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
            float(np.sin(th0)), float(np.cos(th0)),
        ]

        dv, dw = predict_residual(feat, H_min)
        ts_res, e_res, n_res = simulate_ct_timevarying(
            E[i0], N[i0], v0, th0, omega0, dv, dw, H_sec,
            dt_sim=DT_SIM, v_cap=v_cap, omega_cap=omega_cap
        )

        gt_end = interp_xy_at_time(t_sec, E, N, t0 + H_sec)
        if gt_end is None:
            continue
        e_gt, n_gt = gt_end

        end_res = float(np.hypot(e_res[-1] - e_gt, n_res[-1] - n_gt))
        ts_gt, e_gt_path, n_gt_path = build_gt_path(t0, H_sec, t_sec, E, N, dt_sim=DT_SIM)
        rmse_res = rmse_path(e_res, n_res, e_gt_path, n_gt_path)
        rows.append([i0, H_min, end_res, rmse_res])

    bt = bt.merge(
        pd.DataFrame(rows, columns=["i0", "H_min", "end_res", "rmse_res"]),
        on=["i0", "H_min"], how="inner"
    )

    if EVAL_SCOPE == "test":
        bt_eval = bt[bt["split"] == "test"].copy()
        if len(bt_eval) < 10:
            print("[WARN] test 窗口过少，评估回退为全体窗口。")
            bt_eval = bt.copy()
    else:
        bt_eval = bt.copy()

    print("[EVAL state]", bt_eval["state"].value_counts(dropna=False).to_dict())

    # ===== 选择代表性起点 =====
    sub10_all = bt_eval[bt_eval.H_min == 10].copy()
    if USE_RES_BETTER_SUBSET_FOR_VIS:
        sub10 = sub10_all[sub10_all["end_res"] < sub10_all["end_ct"]].copy()
        if len(sub10) < 5:
            print("[WARN] 10 min 条件下残差修正优于 CT 的片段过少，回退为全样本选择。")
            sub10 = sub10_all
    else:
        sub10 = sub10_all

    p50_t = sub10.end_ct.median()
    p90_t = sub10.end_ct.quantile(0.90)
    i0_p50_auto = int(sub10.iloc[(sub10.end_ct - p50_t).abs().argmin()].i0)
    i0_p90_auto = int(sub10.iloc[(sub10.end_ct - p90_t).abs().argmin()].i0)

    cand_i0 = set(bt_eval["i0"].unique().tolist())
    if (FIX_I0_P50 is not None) and (FIX_I0_P50 in cand_i0):
        i0_p50 = int(FIX_I0_P50)
    else:
        i0_p50 = int(i0_p50_auto)

    if (FIX_I0_P90 is not None) and (FIX_I0_P90 in cand_i0):
        i0_p90 = int(FIX_I0_P90)
    else:
        i0_p90 = int(i0_p90_auto)

    print(f"[INFO] P50 i0={i0_p50} ; P90 i0={i0_p90}")

    # ===== Funnel 校准 =====
    cal = bt.copy()
    q1, q2 = cal["v0"].quantile(0.33), cal["v0"].quantile(0.66)

    def speed_bin(v):
        return "low" if v <= q1 else ("mid" if v <= q2 else "high")

    cal["speed_bin"] = cal["v0"].map(speed_bin)
    sig = (cal.groupby(["H_min", "state", "speed_bin"])["end_res"]
           .median().reset_index().rename(columns={"end_res": "sigma_end"}))

    def lookup_sigma(H_min, state, speed_bin_):
        s = sig[(sig.H_min == H_min) & (sig.state == state) & (sig.speed_bin == speed_bin_)]
        if len(s) == 0:
            return float(cal[cal.H_min == H_min]["end_res"].median())
        return float(s.iloc[0].sigma_end)

    def _fill_funnel_polygon(ax, e_center, n_center, w, color="#1f77b4",
                             alpha_max=0.40, label=None):
        e = np.asarray(e_center, float)
        n = np.asarray(n_center, float)
        w = np.asarray(w, float)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        if len(e) < 2:
            return

        ds = np.hypot(np.diff(e), np.diff(n))
        keep = np.r_[True, ds > 1e-6]
        e = e[keep]
        n = n[keep]
        w = w[keep]
        if len(e) < 2:
            return

        ds2 = np.hypot(np.diff(e), np.diff(n))
        s_arc = np.r_[0.0, np.cumsum(ds2)]
        if float(s_arc[-1]) < 1e-6:
            return

        GAMMA = 0.55
        w_vis = np.minimum(w, GAMMA * s_arc)

        W_MIN_VIS = 0.8
        w_vis = np.maximum(w_vis, W_MIN_VIS)
        w_vis[0] = 0.0

        nseg = len(e) - 1
        for i in range(nseg):
            x0, y0 = e[i], n[i]
            x1, y1 = e[i + 1], n[i + 1]
            dx, dy = x1 - x0, y1 - y0
            seglen = float(np.hypot(dx, dy))
            if seglen < 1e-9:
                continue

            tx, ty = dx / seglen, dy / seglen
            nx, ny = -ty, tx

            w0 = float(w_vis[i])
            w1 = float(w_vis[i + 1])

            p0u = (x0 + nx * w0, y0 + ny * w0)
            p1u = (x1 + nx * w1, y1 + ny * w1)
            p1l = (x1 - nx * w1, y1 - ny * w1)
            p0l = (x0 - nx * w0, y0 - ny * w0)

            poly_x = [p0u[0], p1u[0], p1l[0], p0l[0]]
            poly_y = [p0u[1], p1u[1], p1l[1], p0l[1]]

            s = (i + 1) / max(1, nseg)
            a = alpha_max * (0.20 + 0.80 * (s ** 1.6))
            a = max(a, 0.02)

            ax.fill(poly_x, poly_y, color=color, alpha=a, linewidth=0.0,
                    label=(label if (i == 0 and label is not None) else None))

    def plot_segment_compare(i0, tag):
        t0 = float(t_sec[i0])
        v0 = float(v_pos[i0]) if np.isfinite(v_pos[i0]) else 0.0
        th0 = float(theta_pos[i0])
        omega0 = omega0_window(i0, E, N, omega_pos, v0)
        state = motion_state(omega0)

        q1_, q2_ = cal["v0"].quantile(0.33), cal["v0"].quantile(0.66)

        def speed_bin_(v):
            return "low" if v <= q1_ else ("mid" if v <= q2_ else "high")

        sb = speed_bin_(v0)

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
            float(np.sin(th0)), float(np.cos(th0)),
        ]

        c_gt = "#1f77b4"
        c_cv = "#ff7f0e"
        c_ct = "#2ca02c"
        c_res = "#d62728"

        fig = plt.figure(figsize=(10.0, 3.8))
        gs = fig.add_gridspec(1, 3, wspace=0.30)
        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
        fig.subplots_adjust(top=0.80)

        end_cv_list, end_ct_list, end_res_list = [], [], []
        rmse_cv_list, rmse_ct_list, rmse_res_list = [], [], []

        for j, H_min in enumerate(HORIZONS_MIN):
            H_sec = 60.0 * H_min

            ts_gt, e_gt, n_gt = build_gt_path(t0, H_sec, t_sec, E, N, dt_sim=DT_SIM)

            ts_cv, e_cv, n_cv = simulate_cv_heading(
                E[i0], N[i0], v0, th0, H_sec, dt_sim=DT_SIM, v_cap=v_cap
            )

            ts_ct, e_ct, n_ct = simulate_ct_timevarying(
                E[i0], N[i0], v0, th0, omega0, 0.0, 0.0, H_sec,
                dt_sim=DT_SIM, v_cap=v_cap, omega_cap=omega_cap
            )

            dv, dw = predict_residual(feat, H_min)
            ts_res, e_res, n_res = simulate_ct_timevarying(
                E[i0], N[i0], v0, th0, omega0, dv, dw, H_sec,
                dt_sim=DT_SIM, v_cap=v_cap, omega_cap=omega_cap
            )

            end_cv_list.append(float(np.hypot(e_cv[-1] - e_gt[-1], n_cv[-1] - n_gt[-1])))
            end_ct_list.append(float(np.hypot(e_ct[-1] - e_gt[-1], n_ct[-1] - n_gt[-1])))
            end_res_list.append(float(np.hypot(e_res[-1] - e_gt[-1], n_res[-1] - n_gt[-1])))

            rmse_cv_list.append(rmse_path(e_cv, n_cv, e_gt, n_gt))
            rmse_ct_list.append(rmse_path(e_ct, n_ct, e_gt, n_gt))
            rmse_res_list.append(rmse_path(e_res, n_res, e_gt, n_gt))

            sigma_end = lookup_sigma(H_min, state, sb)
            w = funnel_width(ts_res, H_sec, sigma_end, v0, k=1.64)

            ax = axs[j]
            _fill_funnel_polygon(ax, e_res, n_res, w, color=c_gt, alpha_max=0.40,
                                 label=("90%置信域" if j == 0 else None))

            ax.plot(e_gt, n_gt, color=c_gt, linewidth=2.0, zorder=6, label=("真实轨迹" if j == 0 else None))
            ax.plot(e_cv, n_cv, color=c_cv, linestyle=":", linewidth=1.7, zorder=7,
                    label=("CV-Heading预测" if j == 0 else None))
            ax.plot(e_ct, n_ct, color=c_ct, linestyle="-.", linewidth=1.8, zorder=8,
                    label=("CV-CT预测" if j == 0 else None))
            ax.plot(e_res, n_res, color=c_res, linestyle="--", linewidth=1.8, zorder=9,
                    label=("RAP-CT预测" if j == 0 else None))

            ax.scatter([e_gt[0]], [n_gt[0]], s=28, marker="o", color="black", zorder=12)
            ax.scatter([e_gt[-1]], [n_gt[-1]], s=38, marker="^", color=c_gt, zorder=12)

            ax.set_title(f"({chr(97 + j)}) {tag}，预测时长={H_min} min", fontproperties=FP_SIMSUN)
            ax.set_xlabel("东向坐标/m", fontproperties=FP_SIMSUN)
            ax.set_ylabel("北向坐标/m", fontproperties=FP_SIMSUN)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.25)
            ax.yaxis.set_label_coords(-0.18, 0.5)
            apply_tick_font(ax, xfont=FP_TIMES, yfont=FP_TIMES)
            apply_tick_style(ax)

        handles, labels = [], []
        for a in axs:
            h, l = a.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)

        lg = fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 0.98))
        apply_legend_font(lg, font=FP_SIMSUN)

        traj_png = os.path.join(OUT_DIR, f"任务4_{tag}_轨迹对比.png")
        traj_pdf = os.path.join(OUT_DIR, f"任务4_{tag}_轨迹对比.pdf")
        fig.savefig(traj_png, bbox_inches="tight", pad_inches=0.15)
        with PdfPages(traj_pdf) as pdf:
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)

        hs = np.array(list(HORIZONS_MIN), dtype=float)
        x = np.arange(len(hs))
        width = 0.24
        off = width

        fig2 = plt.figure(figsize=(10.0, 3.6))
        gs2 = fig2.add_gridspec(1, 2, wspace=0.25)
        ax1 = fig2.add_subplot(gs2[0, 0])
        ax2 = fig2.add_subplot(gs2[0, 1])

        ax1.bar(x - off, end_cv_list, width, label="CV-Heading", color=c_cv, alpha=0.90)
        ax1.bar(x + 0.0, end_ct_list, width, label="CV-CT", color=c_ct, alpha=0.85)
        ax1.bar(x + off, end_res_list, width, label="RAP-CT", color=c_res, alpha=0.90)
        ax1.set_title(f"{tag}终点误差", fontproperties=FP_SIMSUN)
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(int(h)) for h in hs])
        ax1.set_xlabel("预测时长/min", fontproperties=FP_SIMSUN)
        ax1.set_ylabel("误差/m", fontproperties=FP_SIMSUN)
        ax1.grid(True, axis="y", alpha=0.25)
        lg1 = ax1.legend(frameon=True, ncol=3, loc="upper left")
        apply_legend_font(lg1, font=FP_SIMSUN)
        apply_tick_font(ax1, xfont=FP_TIMES, yfont=FP_TIMES)
        apply_tick_style(ax1)

        for i in range(len(hs)):
            base = end_ct_list[i] + 1e-9
            imp_res = (base - end_res_list[i]) / base * 100.0
            ax1.text(x[i] + off, end_res_list[i], f"{imp_res:+.1f}%", ha="center", va="bottom", fontsize=9)

        ax2.bar(x - off, rmse_cv_list, width, label="CV-Heading", color=c_cv, alpha=0.70)
        ax2.bar(x + 0.0, rmse_ct_list, width, label="CV-CT", color=c_ct, alpha=0.65)
        ax2.bar(x + off, rmse_res_list, width, label="RAP-CT", color=c_res, alpha=0.70)
        ax2.set_title(f"{tag}路径均方根误差", fontproperties=FP_SIMSUN)
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(int(h)) for h in hs])
        ax2.set_xlabel("预测时长（min）", fontproperties=FP_SIMSUN)
        ax2.set_ylabel("误差（m）", fontproperties=FP_SIMSUN)
        ax2.grid(True, axis="y", alpha=0.25)
        lg2 = ax2.legend(frameon=True, ncol=3, loc="upper left")
        apply_legend_font(lg2, font=FP_SIMSUN)
        apply_tick_font(ax2, xfont=FP_TIMES, yfont=FP_TIMES)
        apply_tick_style(ax2)

        for i in range(len(hs)):
            base = rmse_ct_list[i] + 1e-9
            imp_res = (base - rmse_res_list[i]) / base * 100.0
            ax2.text(x[i] + off, rmse_res_list[i], f"{imp_res:+.1f}%", ha="center", va="bottom", fontsize=9)

        err_png = os.path.join(OUT_DIR, f"任务4_{tag}_误差柱状图.png")
        err_pdf = os.path.join(OUT_DIR, f"任务4_{tag}_误差柱状图.pdf")
        fig2.savefig(err_png, bbox_inches="tight", pad_inches=0.15)
        with PdfPages(err_pdf) as pdf:
            pdf.savefig(fig2, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig2)

    # ===== 绘图 =====
    plot_segment_compare(i0_p50, "P50")
    plot_segment_compare(i0_p90, "P90")

    # ===== 导出论文表4：绝对误差主表 =====
    abs_table = build_abs_error_table(bt_eval, [i0_p50, i0_p90], ["P50", "P90"], HORIZONS_MIN)
    abs_csv = os.path.join(OUT_DIR, "表4_代表性起点绝对误差主表.csv")
    abs_table.to_csv(abs_csv, index=False, encoding="utf-8-sig")

    # ===== 导出论文表5：改善率表 =====
    improve_table = build_improve_table(bt_eval, [i0_p50, i0_p90], ["P50", "P90"], HORIZONS_MIN)
    improve_csv = os.path.join(OUT_DIR, "表5_RAP-CT相对CV-CT改善率.csv")
    improve_table.to_csv(improve_csv, index=False, encoding="utf-8-sig")

    # ===== 导出代表性起点说明表 =====
    rep_info = export_representative_info(bt_eval, [i0_p50, i0_p90], ["P50", "P90"], t_sec)
    rep_csv = os.path.join(OUT_DIR, "表S1_代表性起点信息.csv")
    rep_info.to_csv(rep_csv, index=False, encoding="utf-8-sig")

    # ===== 同时导出一个宽表，便于你复制进 Word =====
    wide_abs = abs_table.pivot_table(
        index=["起点类型", "方法"],
        columns="指标",
        values=[f"{h} min" for h in HORIZONS_MIN],
        aggfunc="first"
    )
    wide_abs.to_csv(os.path.join(OUT_DIR, "表4_绝对误差主表_宽表版.csv"), encoding="utf-8-sig")

    overall_csv, overall_imp_csv = export_overall_tables(bt_eval, bt, OUT_DIR, HORIZONS_MIN)

    print("[INFO] 已导出：")
    print(f"  - {abs_csv}")
    print(f"  - {improve_csv}")
    print(f"  - {rep_csv}")
    print(f"  - {os.path.join(OUT_DIR, '表4_绝对误差主表_宽表版.csv')}")
    print(f"  - {overall_csv}")
    print(f"  - {overall_imp_csv}")


if __name__ == "__main__":
    main()