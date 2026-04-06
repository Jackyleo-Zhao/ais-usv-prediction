# -*- coding: utf-8 -*-
"""
horn_paper_full_v1.py

单文件完整版：AIS 数据驱动 + 喇叭口动态预测 + 2/10/20min 多尺度对比 + 误差统计 + 船舶操纵特性约束 + 论文静态导出(P50/P90)

依赖：
- Python 3.8+
- numpy, pandas, matplotlib
- tkinter (标准库)
"""

import os
import math
import time
import json
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================
# Matplotlib 中文显示
# =========================
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 工具函数：角度、地理坐标
# =========================
def wrap_deg_360(x: float) -> float:
    x = float(x)
    x = x % 360.0
    if x < 0:
        x += 360.0
    return x


def wrap_rad_pi(x: float) -> float:
    """wrap 到 [-pi, pi]"""
    x = float(x)
    while x > math.pi:
        x -= 2 * math.pi
    while x < -math.pi:
        x += 2 * math.pi
    return x


def ang_diff_deg(a: float, b: float) -> float:
    """a-b 的最小角差（deg），返回 [-180, 180]"""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Haversine 距离，单位：米"""
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(R * c)


def ll_to_xy_m(lat, lon, lat0, lon0):
    """
    小范围投影：equirectangular（局部切平面近似）
    x: 北向(m), y: 东向(m)
    """
    R = 6371000.0
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0r = math.radians(lat0)
    x = R * np.radians(lat - lat0)
    y = R * np.radians(lon - lon0) * math.cos(lat0r)
    return x, y


def xy_to_ll(x, y, lat0, lon0):
    R = 6371000.0
    lat0r = math.radians(lat0)
    lat = np.degrees(x / R) + lat0
    lon = np.degrees(y / (R * math.cos(lat0r))) + lon0
    return lat, lon


def bearing_to_unit_xy(psi_rad):
    """航向角 psi: 0=北，90=东（数学上用北为x正向），输出单位方向 (dx,dy) 对应 (北向,东向)"""
    dx = math.cos(psi_rad)
    dy = math.sin(psi_rad)
    return dx, dy


def lateral_unit_xy(psi_rad):
    """左法向（相对航向左侧）的单位向量 (nx, ny)"""
    # 航向单位向量 (cos, sin)，左法向 (-sin, cos)
    nx = -math.sin(psi_rad)
    ny = math.cos(psi_rad)
    return nx, ny


# =========================
# AIS 数据与参数计算
# =========================
# 你 CSV 的 17 列名（严格校验用）
REQUIRED_COLUMNS = [
    "mmsi", "base_date_time", "longitude", "latitude", "sog", "cog", "heading",
    "vessel_name", "imo", "call_sign", "vessel_type", "status", "length",
    "width", "draft", "cargo", "transceiver"
]

# 读取后：把新列名映射为原代码内部使用的列名（不影响后续逻辑）
COLUMN_RENAME_TO_INTERNAL = {
    "mmsi": "MMSI",
    "base_date_time": "BaseDateTime",
    "longitude": "LON",
    "latitude": "LAT",
    "sog": "SOG",
    "cog": "COG",
    "heading": "Heading",
    "vessel_name": "VesselName",
    "imo": "IMO",
    "call_sign": "CallSign",
    "vessel_type": "VesselType",
    "status": "Status",
    "length": "Length",
    "width": "Width",
    "draft": "Draft",
    "cargo": "Cargo",
    "transceiver": "TransceiverClass",
}


class AISDataset:
    def __init__(self):
        self.df = None
        self.file_path = None
        self.lat0 = None
        self.lon0 = None
        self.params = None  # 操纵特性参数

    def load_csv(self, path: str):
        df = pd.read_csv(path)

        # 统一列名：去空格 + 小写（以便严格对齐你给的 17 列）
        df.columns = [c.strip() for c in df.columns]
        cols_lower = [c.lower() for c in df.columns]
        df.columns = cols_lower

        cols = list(df.columns)

        # 严格校验列名（不允许缺/多）
        if cols != REQUIRED_COLUMNS:
            # 允许顺序不同但集合一致
            if set(cols) != set(REQUIRED_COLUMNS):
                raise ValueError(
                    f"CSV列名不匹配。\n"
                    f"你的列名: {cols}\n"
                    f"要求列名: {REQUIRED_COLUMNS}"
                )
            else:
                # 顺序不同则重排
                df = df[REQUIRED_COLUMNS]

        # 重命名为内部列名（后续代码不改）
        df = df.rename(columns=COLUMN_RENAME_TO_INTERNAL)

        # 时间解析并排序
        df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], errors="coerce", utc=True)
        df = df.dropna(subset=["BaseDateTime", "LAT", "LON", "SOG", "COG"])
        df = df.sort_values("BaseDateTime").reset_index(drop=True)

        # 基本清洗
        df["LAT"] = df["LAT"].astype(float)
        df["LON"] = df["LON"].astype(float)
        df["SOG"] = df["SOG"].astype(float)  # knots
        df["COG"] = df["COG"].astype(float)  # deg

        # 增加内部列（不写回 CSV，只在内存使用）
        t0 = df["BaseDateTime"].iloc[0]
        df["_tsec"] = (df["BaseDateTime"] - t0).dt.total_seconds().astype(float)

        # 单位换算：knots->m/s
        df["_v_mps"] = df["SOG"] * 0.514444

        # 航向角（rad）
        df["_psi_rad"] = np.radians(df["COG"].apply(wrap_deg_360).values)

        # 计算 dt
        dt = df["_tsec"].diff().values
        dt[0] = np.nan
        df["_dt"] = dt

        # 加速度 a = dv/dt
        dv = df["_v_mps"].diff().values
        a = dv / dt
        a[0] = np.nan
        # 对异常 dt 做处理
        a = np.where((dt > 0.0) & (dt < 3600.0), a, np.nan)
        df["_a_mps2"] = a

        # 回转角速度 omega = dpsi/dt（psi 用最小角差）
        dpsi = np.zeros(len(df), dtype=float)
        dpsi[:] = np.nan
        for i in range(1, len(df)):
            if np.isfinite(dt[i]) and dt[i] > 0:
                d = wrap_rad_pi(df["_psi_rad"].iloc[i] - df["_psi_rad"].iloc[i - 1])
                dpsi[i] = d
        omega = dpsi / dt
        omega = np.where((dt > 0.0) & (dt < 3600.0), omega, np.nan)
        df["_omega_rads"] = omega

        # 设定局部投影参考点为数据中位数附近，减少数值偏移
        self.lat0 = float(df["LAT"].iloc[0])
        self.lon0 = float(df["LON"].iloc[0])
        x, y = ll_to_xy_m(df["LAT"].values, df["LON"].values, self.lat0, self.lon0)
        df["_x_m"] = x  # 北向
        df["_y_m"] = y  # 东向

        self.df = df
        self.file_path = path

        # 计算操纵特性参数
        self.params = self.compute_maneuver_params()
        return df

    def compute_maneuver_params(self):
        """
        操纵特性（基于历史数据估计）：
        - 最大速度 vmax (m/s)
        - 最大正加速度 amax_pos (m/s^2)（P95 取稳健）
        - 最大负加速度 amax_neg (m/s^2)（P05 更负，停车用）
        - 最大回转角速度 omega_max (rad/s)（P95 取稳健）
        - 最小转弯半径 r_min (m)（基于 v/|omega| 的 P05）
        - 回转直径 d_turn = 2*r_min
        - 停车距离 stop_dist_max (m)：以 vmax 和 |amax_neg| 估计 v^2/(2|a|)
        """
        df = self.df
        v = df["_v_mps"].values
        a = df["_a_mps2"].values
        w = df["_omega_rads"].values

        v_valid = v[np.isfinite(v) & (v >= 0)]
        vmax = float(np.nanmax(v_valid)) if len(v_valid) > 0 else 0.1

        a_valid = a[np.isfinite(a)]
        # 正加速度稳健上界
        a_pos = a_valid[a_valid > 0]
        amax_pos = float(np.nanpercentile(a_pos, 95)) if len(a_pos) > 10 else float(np.nanmax(a_pos)) if len(a_pos) > 0 else 0.2

        # 负加速度（更负），停车用：取 5分位（更负）
        a_neg = a_valid[a_valid < 0]
        amax_neg = float(np.nanpercentile(a_neg, 5)) if len(a_neg) > 10 else float(np.nanmin(a_neg)) if len(a_neg) > 0 else -0.2
        if amax_neg >= -1e-6:
            amax_neg = -0.2

        w_valid = np.abs(w[np.isfinite(w)])
        omega_max = float(np.nanpercentile(w_valid, 95)) if len(w_valid) > 10 else float(np.nanmax(w_valid)) if len(w_valid) > 0 else math.radians(2.0) / 1.0

        # r = v/|omega|
        r_list = []
        for vi, wi in zip(v, w):
            if np.isfinite(vi) and np.isfinite(wi) and abs(wi) > 1e-4 and vi > 0.2:
                r_list.append(vi / abs(wi))
        if len(r_list) > 20:
            r_min = float(np.nanpercentile(r_list, 5))
        elif len(r_list) > 0:
            r_min = float(np.nanmin(r_list))
        else:
            # fallback：给一个保守值
            r_min = max(10.0, vmax / max(omega_max, 1e-3))

        d_turn = 2.0 * r_min

        stop_dist_max = vmax * vmax / (2.0 * abs(amax_neg))  # v^2/(2|a|)
        return {
            "vmax_mps": vmax,
            "amax_pos_mps2": amax_pos,
            "amax_neg_mps2": amax_neg,
            "omega_max_rads": omega_max,
            "r_min_m": r_min,
            "turn_diameter_m": d_turn,
            "stop_dist_m": stop_dist_max
        }


# =========================
# 预测模型：CTRA + 操纵特性约束
# =========================
class PredictorCTRA:
    def __init__(self, params: dict):
        self.params = params

    def estimate_state_from_history(self, df: pd.DataFrame, idx: int, window_sec: float = 60.0):
        """
        用历史窗口估计当前状态：
        - 位置 (x,y)
        - 航向 psi
        - 速度 v
        - 加速度 a（用窗口拟合/差分的稳健估计）
        - 回转角速度 omega（同上）
        """
        idx = int(idx)
        if idx <= 0:
            idx = 1
        t_now = df["_tsec"].iloc[idx]
        t_start = t_now - window_sec
        sub = df[(df["_tsec"] >= t_start) & (df["_tsec"] <= t_now)].copy()
        if len(sub) < 5:
            sub = df.iloc[max(0, idx - 10): idx + 1].copy()

        x0 = float(df["_x_m"].iloc[idx])
        y0 = float(df["_y_m"].iloc[idx])
        psi0 = float(df["_psi_rad"].iloc[idx])
        v0 = float(df["_v_mps"].iloc[idx])

        # 速度平滑（稳健：取窗口中位数）
        v_med = float(np.nanmedian(sub["_v_mps"].values))
        if np.isfinite(v_med):
            v0 = v_med

        # 加速度估计：取窗口 a 的中位数（更稳）
        a_vals = sub["_a_mps2"].values
        a_vals = a_vals[np.isfinite(a_vals)]
        if len(a_vals) >= 5:
            a0 = float(np.nanmedian(a_vals))
        else:
            a0 = 0.0

        # omega 估计：取窗口 omega 的中位数
        w_vals = sub["_omega_rads"].values
        w_vals = w_vals[np.isfinite(w_vals)]
        if len(w_vals) >= 5:
            omega0 = float(np.nanmedian(w_vals))
        else:
            omega0 = 0.0

        # 操纵特性限幅
        vmax = max(self.params.get("vmax_mps", 0.1), 0.1)
        omega_max = max(self.params.get("omega_max_rads", 1e-3), 1e-3)
        amax_pos = max(self.params.get("amax_pos_mps2", 0.1), 0.05)
        amax_neg = min(self.params.get("amax_neg_mps2", -0.1), -0.05)

        v0 = float(np.clip(v0, 0.0, vmax))
        omega0 = float(np.clip(omega0, -omega_max, omega_max))
        a0 = float(np.clip(a0, amax_neg, amax_pos))

        return {
            "x": x0, "y": y0, "psi": psi0,
            "v": v0, "a": a0, "omega": omega0,
            "t_now": float(t_now)
        }

    def predict_xy_ctra(self, state: dict, horizon_sec: float, dt: float = 5.0):
        """
        CTRA: Constant Turn Rate and Acceleration
        输出：t_list, x_list, y_list, psi_list, v_list
        """
        horizon_sec = float(horizon_sec)
        dt = float(dt)
        n = int(math.ceil(horizon_sec / dt))
        if n < 1:
            n = 1

        vmax = max(self.params.get("vmax_mps", 0.1), 0.1)
        omega_max = max(self.params.get("omega_max_rads", 1e-3), 1e-3)
        amax_pos = max(self.params.get("amax_pos_mps2", 0.1), 0.05)
        amax_neg = min(self.params.get("amax_neg_mps2", -0.1), -0.05)

        x = float(state["x"])
        y = float(state["y"])
        psi = float(state["psi"])
        v = float(state["v"])
        a = float(state["a"])
        omega = float(state["omega"])

        # 二次限幅，确保稳健
        v = float(np.clip(v, 0.0, vmax))
        omega = float(np.clip(omega, -omega_max, omega_max))
        a = float(np.clip(a, amax_neg, amax_pos))

        t_list = [0.0]
        x_list = [x]
        y_list = [y]
        psi_list = [psi]
        v_list = [v]

        for k in range(1, n + 1):
            # 更新速度
            v_next = v + a * dt
            v_next = float(np.clip(v_next, 0.0, vmax))

            # 更新航向
            psi_next = psi + omega * dt
            psi_next = wrap_rad_pi(psi_next)

            # 位移：用“当前步中间速度”近似
            v_mid = 0.5 * (v + v_next)
            dx, dy = bearing_to_unit_xy(psi_mid := (psi + psi_next) * 0.5)
            x_next = x + v_mid * dt * dx
            y_next = y + v_mid * dt * dy

            # 保存
            t_list.append(k * dt)
            x_list.append(x_next)
            y_list.append(y_next)
            psi_list.append(psi_next)
            v_list.append(v_next)

            x, y, psi, v = x_next, y_next, psi_next, v_next

        # 截断到 horizon_sec（如果最后一个点超过）
        t_arr = np.array(t_list, dtype=float)
        x_arr = np.array(x_list, dtype=float)
        y_arr = np.array(y_list, dtype=float)
        psi_arr = np.array(psi_list, dtype=float)
        v_arr = np.array(v_list, dtype=float)

        if t_arr[-1] > horizon_sec + 1e-6:
            # 线性插值截断
            t_new = np.linspace(0.0, horizon_sec, int(math.floor(horizon_sec / dt)) + 1)
            x_new = np.interp(t_new, t_arr, x_arr)
            y_new = np.interp(t_new, t_arr, y_arr)
            psi_new = np.interp(t_new, t_arr, psi_arr)
            v_new = np.interp(t_new, t_arr, v_arr)
            return t_new, x_new, y_new, psi_new, v_new

        return t_arr, x_arr, y_arr, psi_arr, v_arr


# =========================
# 真值未来轨迹插值（解决 AIS 不规则）
# =========================
def interp_future_truth(df: pd.DataFrame, t0: float, horizon_sec: float, dt: float = 5.0):
    """
    给定起点时刻 t0 (sec) 与预测 horizon，输出真值轨迹（插值到固定时间网格）
    返回：t_grid, x_true, y_true, ok(是否足够真值)
    """
    t_end = t0 + horizon_sec
    sub = df[(df["_tsec"] >= t0) & (df["_tsec"] <= t_end)].copy()
    if len(sub) < 2:
        return None, None, None, False

    t_sub = sub["_tsec"].values
    x_sub = sub["_x_m"].values
    y_sub = sub["_y_m"].values

    # 固定网格
    n = int(math.floor(horizon_sec / dt)) + 1
    t_grid = t0 + np.arange(n) * dt
    if t_grid[-1] > t_end + 1e-6:
        t_grid[-1] = t_end

    # 插值
    x_true = np.interp(t_grid, t_sub, x_sub)
    y_true = np.interp(t_grid, t_sub, y_sub)
    return t_grid - t0, x_true, y_true, True


# =========================
# 误差计算
# =========================
def compute_errors_xy(x_pred, y_pred, x_true, y_true):
    """
    输入同长度的预测/真值轨迹（xy 米），输出：
    - mean_err, rmse, end_err
    - err_series（每时刻欧氏误差）
    """
    x_pred = np.asarray(x_pred, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    x_true = np.asarray(x_true, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    n = min(len(x_pred), len(x_true))
    if n <= 1:
        return np.nan, np.nan, np.nan, None

    dx = x_pred[:n] - x_true[:n]
    dy = y_pred[:n] - y_true[:n]
    err = np.sqrt(dx * dx + dy * dy)

    mean_err = float(np.nanmean(err))
    rmse = float(np.sqrt(np.nanmean(err * err)))
    end_err = float(err[n - 1])
    return mean_err, rmse, end_err, err


# =========================
# 喇叭口：随时间+速度+加速度+回转角速度变化
# =========================
def sigma_profile(t_arr, v_arr, a0, omega0, params, horizon_sec):
    """
    生成每个时刻的横向sigma(米)：
    - 随时间增长（越远越宽）
    - 随速度增加更宽
    - 随加速度幅值更宽
    - 随回转角速度幅值更宽（转弯不确定性增大）
    """
    t_arr = np.asarray(t_arr, dtype=float)
    v_arr = np.asarray(v_arr, dtype=float)

    vmax = max(params.get("vmax_mps", 0.1), 0.1)
    omega_max = max(params.get("omega_max_rads", 1e-3), 1e-3)
    amax_pos = max(params.get("amax_pos_mps2", 0.1), 0.05)
    amax_neg = min(params.get("amax_neg_mps2", -0.1), -0.05)
    amax_abs = max(abs(amax_pos), abs(amax_neg), 0.1)

    # 归一化
    tn = np.clip(t_arr / max(horizon_sec, 1e-6), 0.0, 1.0)
    vn = np.clip(v_arr / vmax, 0.0, 1.5)
    an = np.clip(abs(a0) / amax_abs, 0.0, 2.0)
    wn = np.clip(abs(omega0) / omega_max, 0.0, 2.0)

    # 基础宽度（米）
    base = 8.0

    # 时间项：二次增长（远期更明显）
    time_term = 40.0 * (tn ** 2)

    # 速度项：高速更宽
    v_term = 20.0 * (vn ** 1.2)

    # 转弯项：转弯更宽
    w_term = 35.0 * (wn ** 1.0) * (0.4 + 0.6 * tn)

    # 加速度项：加速/减速阶段更不确定
    a_term = 15.0 * (an ** 1.0) * (0.3 + 0.7 * tn)

    sigma = base + time_term + v_term + w_term + a_term
    sigma = np.clip(sigma, 5.0, 250.0)
    return sigma


def build_cone_boundaries(x_center, y_center, psi_arr, sigma_arr):
    """
    给定中心线(x,y)、航向psi、横向sigma，输出左右边界(xL,yL,xR,yR)
    """
    x_center = np.asarray(x_center, dtype=float)
    y_center = np.asarray(y_center, dtype=float)
    psi_arr = np.asarray(psi_arr, dtype=float)
    sigma_arr = np.asarray(sigma_arr, dtype=float)

    xL = np.zeros_like(x_center)
    yL = np.zeros_like(y_center)
    xR = np.zeros_like(x_center)
    yR = np.zeros_like(y_center)

    for i in range(len(x_center)):
        nx, ny = lateral_unit_xy(psi_arr[i])
        xL[i] = x_center[i] + sigma_arr[i] * nx
        yL[i] = y_center[i] + sigma_arr[i] * ny
        xR[i] = x_center[i] - sigma_arr[i] * nx
        yR[i] = y_center[i] - sigma_arr[i] * ny
    return xL, yL, xR, yR


# =========================
# 综合评分 C：用于 P50/P90 选择
# =========================
def composite_score_C(metrics_2, metrics_10, metrics_20, w=(0.25, 0.45, 0.30)):
    """
    C：综合(2/10/20加权)
    这里默认以 RMSE 为主（更稳健），权重默认偏重 10min。
    """
    w2, w10, w20 = w
    rmse2 = metrics_2.get("rmse", np.nan)
    rmse10 = metrics_10.get("rmse", np.nan)
    rmse20 = metrics_20.get("rmse", np.nan)

    if not np.isfinite(rmse2) or not np.isfinite(rmse10) or not np.isfinite(rmse20):
        return np.nan
    return float(w2 * rmse2 + w10 * rmse10 + w20 * rmse20)


# =========================
# UI：滚动左侧 + 右侧绘图 + 自动导出
# =========================
class ScrollableFrame(ttk.Frame):
    """一个可滚动Frame，用于左侧信息太多时可以滚动"""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.vscroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vscroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vscroll.pack(side="right", fill="y")

        # 鼠标滚轮支持（Windows）
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass


class HornPaperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AIS 喇叭口动态预测（论文版：2/10/20 + P50/P90 自动导出）")
        self.root.geometry("1250x860")

        # 数据/模型
        self.data = AISDataset()
        self.predictor = None  # PredictorCTRA
        self.loaded = False

        # 播放控制
        self.playing = False
        self.timer_ms = 200  # UI刷新间隔（ms）
        self.play_rate = 1   # 每次推进几个点
        self.idx = 0

        # 预测设置
        self.dt_pred = 5.0
        self.horizons_min = [2, 10, 20]  # minutes
        self.horizon_enabled = {
            2: tk.BooleanVar(value=True),
            10: tk.BooleanVar(value=True),
            20: tk.BooleanVar(value=True)
        }

        # 导出目录
        self.export_dir = None

        # UI 构建
        self._build_ui()
        self._init_plot()

    # ---------------- UI ----------------
    def _build_ui(self):
        # 左侧滚动面板
        left = ScrollableFrame(self.root)
        left.pack(side="left", fill="y", padx=8, pady=8)

        # 右侧绘图区
        right = ttk.Frame(self.root)
        right.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        self.left = left
        self.right = right

        # ========= 左侧控件 =========
        sec1 = ttk.LabelFrame(left.inner, text="1) 数据与播放")
        sec1.pack(fill="x", pady=6)

        ttk.Button(sec1, text="加载CSV（严格17列）", command=self.on_load_csv).pack(fill="x", padx=6, pady=6)

        self.lbl_file = ttk.Label(sec1, text="未加载数据")
        self.lbl_file.pack(fill="x", padx=6, pady=2)

        row = ttk.Frame(sec1)
        row.pack(fill="x", padx=6, pady=6)
        self.btn_start = ttk.Button(row, text="开始", command=self.on_start, state="disabled")
        self.btn_pause = ttk.Button(row, text="暂停", command=self.on_pause, state="disabled")
        self.btn_reset = ttk.Button(row, text="重置到起点", command=self.on_reset, state="disabled")
        self.btn_start.pack(side="left", expand=True, fill="x", padx=3)
        self.btn_pause.pack(side="left", expand=True, fill="x", padx=3)
        self.btn_reset.pack(side="left", expand=True, fill="x", padx=3)

        row2 = ttk.Frame(sec1)
        row2.pack(fill="x", padx=6, pady=4)
        ttk.Label(row2, text="播放速度(每步跳点数)").pack(side="left")
        self.var_rate = tk.IntVar(value=1)
        ttk.Spinbox(row2, from_=1, to=30, textvariable=self.var_rate, width=6, command=self.on_rate_change).pack(side="left", padx=6)

        self.lbl_idx = ttk.Label(sec1, text="idx: - / -   time: - s")
        self.lbl_idx.pack(fill="x", padx=6, pady=2)

        sec2 = ttk.LabelFrame(left.inner, text="2) 多时间尺度预测设置")
        sec2.pack(fill="x", pady=6)

        for h in self.horizons_min:
            ttk.Checkbutton(sec2, text=f"启用 {h} min", variable=self.horizon_enabled[h], command=self.redraw).pack(anchor="w", padx=8, pady=2)

        row3 = ttk.Frame(sec2)
        row3.pack(fill="x", padx=6, pady=6)
        ttk.Label(row3, text="预测步长 dt_pred(s)").pack(side="left")
        self.var_dt_pred = tk.DoubleVar(value=self.dt_pred)
        ttk.Spinbox(row3, from_=1.0, to=30.0, increment=1.0, textvariable=self.var_dt_pred, width=8, command=self.on_dt_change).pack(side="left", padx=6)

        sec3 = ttk.LabelFrame(left.inner, text="3) 实时误差（2/10/20）")
        sec3.pack(fill="x", pady=6)
        self.txt_err = tk.Text(sec3, height=10, wrap="word")
        self.txt_err.pack(fill="both", expand=True, padx=6, pady=6)

        sec4 = ttk.LabelFrame(left.inner, text="4) 船舶操纵特性（历史估计）")
        sec4.pack(fill="x", pady=6)
        self.txt_params = tk.Text(sec4, height=10, wrap="word")
        self.txt_params.pack(fill="both", expand=True, padx=6, pady=6)

        sec5 = ttk.LabelFrame(left.inner, text="5) 论文静态导出（A4风格）")
        sec5.pack(fill="x", pady=6)

        ttk.Button(sec5, text="导出当前帧（2/10/20 全套）", command=self.export_current_frame, state="disabled").pack(fill="x", padx=6, pady=4)
        ttk.Button(sec5, text="自动导出论文全套（P50典型 + P90困难）", command=self.export_auto_p50_p90, state="disabled").pack(fill="x", padx=6, pady=4)

        # ========= 右侧绘图 =========
        self.fig = plt.Figure(figsize=(9.2, 7.2), dpi=100)
        self.ax_map = self.fig.add_subplot(211)
        self.ax_err = self.fig.add_subplot(212)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _init_plot(self):
        self.ax_map.clear()
        self.ax_map.set_title("轨迹对比（蓝=历史，红点=当前，虚线=预测，实线=真实未来，阴影=喇叭口）")
        self.ax_map.set_xlabel("东向 y (m)")
        self.ax_map.set_ylabel("北向 x (m)")
        self.ax_map.grid(True, alpha=0.25)

        self.ax_err.clear()
        self.ax_err.set_title("误差随时间（2/10/20 min）")
        self.ax_err.set_xlabel("预测时间 t (s)")
        self.ax_err.set_ylabel("位置误差 (m)")
        self.ax_err.grid(True, alpha=0.25)

        self.canvas.draw()

    # ---------------- 事件处理 ----------------
    def on_rate_change(self):
        self.play_rate = int(self.var_rate.get())

    def on_dt_change(self):
        self.dt_pred = float(self.var_dt_pred.get())
        self.redraw()

    def on_load_csv(self):
        path = filedialog.askopenfilename(
            title="选择 AIS CSV 文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            df = self.data.load_csv(path)
            self.predictor = PredictorCTRA(self.data.params)
            self.loaded = True
            self.idx = 0
            self.lbl_file.config(text=f"已加载：{os.path.basename(path)}  (N={len(df)})")
            self.btn_start.config(state="normal")
            self.btn_reset.config(state="normal")
            self.btn_pause.config(state="disabled")

            # 启用导出按钮
            for w in self.left.inner.winfo_children():
                if isinstance(w, ttk.LabelFrame) and w.cget("text").startswith("5)"):
                    for b in w.winfo_children():
                        if isinstance(b, ttk.Button):
                            b.config(state="normal")

            self._update_params_text()
            self.redraw()
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def on_start(self):
        if not self.loaded:
            return
        self.playing = True
        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal")
        self._loop()

    def on_pause(self):
        self.playing = False
        self.btn_start.config(state="normal")
        self.btn_pause.config(state="disabled")

    def on_reset(self):
        if not self.loaded:
            return
        self.idx = 0
        self.redraw()

    # ---------------- 主循环 ----------------
    def _loop(self):
        if not self.playing:
            return
        if not self.loaded:
            return

        df = self.data.df
        self.idx += max(1, self.play_rate)
        if self.idx >= len(df) - 2:
            self.idx = len(df) - 2
            self.playing = False
            self.btn_start.config(state="normal")
            self.btn_pause.config(state="disabled")

        self.redraw()
        self.root.after(self.timer_ms, self._loop)

    # ---------------- 绘图与计算 ----------------
    def redraw(self):
        if not self.loaded:
            self._init_plot()
            return

        df = self.data.df
        idx = int(np.clip(self.idx, 1, len(df) - 2))
        self.idx = idx

        t_now = float(df["_tsec"].iloc[idx])
        self.lbl_idx.config(text=f"idx: {idx} / {len(df)-1}   time: {t_now:.1f} s")

        # 历史轨迹（到当前）
        x_hist = df["_x_m"].iloc[:idx + 1].values
        y_hist = df["_y_m"].iloc[:idx + 1].values

        # 当前点
        x0 = float(df["_x_m"].iloc[idx])
        y0 = float(df["_y_m"].iloc[idx])

        # 估计当前状态
        state = self.predictor.estimate_state_from_history(df, idx, window_sec=60.0)

        # 清空绘图
        self.ax_map.clear()
        self.ax_err.clear()

        # Map
        self.ax_map.set_title("轨迹对比（蓝=历史，红点=当前，虚线=预测，实线=真实未来，阴影=喇叭口）")
        self.ax_map.set_xlabel("东向 y (m)")
        self.ax_map.set_ylabel("北向 x (m)")
        self.ax_map.grid(True, alpha=0.25)

        # 误差
        self.ax_err.set_title("误差随时间（2/10/20 min）")
        self.ax_err.set_xlabel("预测时间 t (s)")
        self.ax_err.set_ylabel("位置误差 (m)")
        self.ax_err.grid(True, alpha=0.25)

        # 画历史
        self.ax_map.plot(y_hist, x_hist, linewidth=2.0, label="历史轨迹(至当前)")

        # 当前点
        self.ax_map.plot([y0], [x0], "o", markersize=7, label="当前")

        # 分 horizon 预测/真值/喇叭口/误差
        txt_lines = []
        for h_min in self.horizons_min:
            if not bool(self.horizon_enabled[h_min].get()):
                continue

            horizon_sec = float(h_min) * 60.0
            dt_pred = float(self.dt_pred)

            # 预测
            t_pred, x_pred, y_pred, psi_pred, v_pred = self.predictor.predict_xy_ctra(
                state=state, horizon_sec=horizon_sec, dt=dt_pred
            )

            # 真值
            t_true, x_true, y_true, ok = interp_future_truth(df, t_now, horizon_sec, dt=dt_pred)
            if not ok:
                txt_lines.append(f"[{h_min}min] 真值不足，无法对比/统计。")
                self.ax_map.plot(y_pred, x_pred, "--", linewidth=2.0, label=f"{h_min}min 预测")
                continue

            # 误差
            mean_err, rmse, end_err, err_series = compute_errors_xy(x_pred, y_pred, x_true, y_true)
            txt_lines.append(
                f"[{h_min}min] Mean={mean_err:.1f} m, RMSE={rmse:.1f} m, End={end_err:.1f} m"
            )

            # 喇叭口（sigma）
            sigma = sigma_profile(t_pred, v_pred, state["a"], state["omega"], self.data.params, horizon_sec)
            xL, yL, xR, yR = build_cone_boundaries(x_pred, y_pred, psi_pred, sigma)

            # 画喇叭口阴影
            poly_x = np.concatenate([xL, xR[::-1]])
            poly_y = np.concatenate([yL, yR[::-1]])
            self.ax_map.fill(poly_y, poly_x, alpha=0.18, label=f"{h_min}min 置信范围")

            # 画预测虚线 vs 真实实线
            self.ax_map.plot(y_pred, x_pred, "--", linewidth=2.0, label=f"{h_min}min 预测")
            self.ax_map.plot(y_true, x_true, "-", linewidth=2.0, label=f"{h_min}min 真实未来")

            # 画误差曲线
            if err_series is not None:
                self.ax_err.plot(t_pred[:len(err_series)], err_series, linewidth=2.0, label=f"{h_min}min 误差")

        # 视窗自适应
        all_y = list(y_hist)
        all_x = list(x_hist)
        pad = 60.0
        if len(all_x) > 10:
            self.ax_map.set_xlim(min(all_y) - pad, max(all_y) + pad)
            self.ax_map.set_ylim(min(all_x) - pad, max(all_x) + pad)

        self.ax_map.legend(loc="best", fontsize=9)
        self.ax_err.legend(loc="best", fontsize=9)

        self.canvas.draw()

        # 左侧误差文本
        self.txt_err.delete("1.0", tk.END)
        self.txt_err.insert(tk.END, "\n".join(txt_lines))

    def _update_params_text(self):
        p = self.data.params
        s = []
        s.append(f"vmax 最大速度: {p['vmax_mps']:.3f} m/s ({p['vmax_mps']/0.514444:.2f} knots)")
        s.append(f"amax_pos 最大正加速度(P95): {p['amax_pos_mps2']:.3f} m/s²")
        s.append(f"amax_neg 最大负加速度(P05): {p['amax_neg_mps2']:.3f} m/s²")
        s.append(f"omega_max 最大回转角速度(P95): {p['omega_max_rads']:.5f} rad/s ({math.degrees(p['omega_max_rads']):.3f} deg/s)")
        s.append(f"r_min 最小转弯半径(P05): {p['r_min_m']:.1f} m")
        s.append(f"turn_diameter 回转直径: {p['turn_diameter_m']:.1f} m")
        s.append(f"stop_dist 停车距离估计(vmax & amax_neg): {p['stop_dist_m']:.1f} m")
        s.append("")
        s.append("说明：这些参数来自历史AIS估计，用于预测限幅与喇叭口动态扩张。")

        self.txt_params.delete("1.0", tk.END)
        self.txt_params.insert(tk.END, "\n".join(s))

    # ---------------- 导出：当前帧 ----------------
    def export_current_frame(self):
        if not self.loaded:
            return
        out_dir = filedialog.askdirectory(title="选择导出目录")
        if not out_dir:
            return
        self.export_dir = out_dir
        idx = self.idx
        self._export_frame_set(idx, out_dir, tag="CURRENT")

    # ---------------- 自动导出：P50/P90 ----------------
    def export_auto_p50_p90(self):
        if not self.loaded:
            return
        out_dir = filedialog.askdirectory(title="选择导出目录（将创建 paper_figs_auto_P50_P90 子目录）")
        if not out_dir:
            return

        auto_dir = os.path.join(out_dir, "paper_figs_auto_P50_P90")
        os.makedirs(auto_dir, exist_ok=True)

        # 扫描候选帧：避免太密集导致时间过长（可调 stride）
        df = self.data.df
        N = len(df)
        stride = max(5, int(N / 300))  # 最多扫 ~300 个点
        candidates = list(range(30, N - 60, stride))

        records = []
        for idx in candidates:
            m = self._compute_metrics_for_idx(idx)
            if m is None:
                continue
            records.append(m)

        if len(records) < 10:
            messagebox.showwarning("导出失败", "可用候选帧太少（真值不足/数据太短）。")
            return

        scores = np.array([r["scoreC"] for r in records], dtype=float)
        scores = scores[np.isfinite(scores)]
        if len(scores) < 10:
            messagebox.showwarning("导出失败", "综合评分C不可用（可能真值不足）。")
            return

        p50_val = float(np.nanpercentile(scores, 50))
        p90_val = float(np.nanpercentile(scores, 90))

        # 找最接近 p50/p90 的帧
        def pick_closest(target):
            best = None
            best_d = 1e18
            for r in records:
                if not np.isfinite(r["scoreC"]):
                    continue
                d = abs(r["scoreC"] - target)
                if d < best_d:
                    best_d = d
                    best = r
            return best

        r50 = pick_closest(p50_val)
        r90 = pick_closest(p90_val)

        if r50 is None or r90 is None:
            messagebox.showwarning("导出失败", "无法找到 P50/P90 对应帧。")
            return

        # 导出两套
        self._export_frame_set(r50["idx"], auto_dir, tag="P50_Typical", extra=r50)
        self._export_frame_set(r90["idx"], auto_dir, tag="P90_Hard", extra=r90)

        # 汇总CSV
        summary_path = os.path.join(auto_dir, "summary.csv")
        pd.DataFrame([r50, r90]).to_csv(summary_path, index=False, encoding="utf-8-sig")

        messagebox.showinfo(
            "导出完成",
            f"已导出：\nP50 idx={r50['idx']}  scoreC={r50['scoreC']:.2f}\n"
            f"P90 idx={r90['idx']}  scoreC={r90['scoreC']:.2f}\n"
            f"目录：{auto_dir}"
        )

    # ---------------- 指标计算（供自动选帧） ----------------
    def _compute_metrics_for_idx(self, idx: int):
        df = self.data.df
        idx = int(idx)
        if idx < 5 or idx > len(df) - 5:
            return None

        t_now = float(df["_tsec"].iloc[idx])
        state = self.predictor.estimate_state_from_history(df, idx, window_sec=60.0)

        result = {"idx": idx, "t_now": t_now}
        metrics = {}
        for h_min in self.horizons_min:
            horizon_sec = h_min * 60.0
            t_pred, x_pred, y_pred, psi_pred, v_pred = self.predictor.predict_xy_ctra(state, horizon_sec, dt=self.dt_pred)
            t_true, x_true, y_true, ok = interp_future_truth(df, t_now, horizon_sec, dt=self.dt_pred)
            if not ok:
                return None
            mean_err, rmse, end_err, _ = compute_errors_xy(x_pred, y_pred, x_true, y_true)
            metrics[h_min] = {"mean": mean_err, "rmse": rmse, "end": end_err}

            result[f"mean_{h_min}m"] = mean_err
            result[f"rmse_{h_min}m"] = rmse
            result[f"end_{h_min}m"] = end_err

        # 2/10/20 的综合评分
        scoreC = composite_score_C(metrics[2], metrics[10], metrics[20], w=(0.25, 0.45, 0.30))
        result["scoreC"] = scoreC

        # 附加：当前操纵状态（方便论文说明）
        result["v0_mps"] = float(state["v"])
        result["a0_mps2"] = float(state["a"])
        result["omega0_rads"] = float(state["omega"])
        return result

    # ---------------- 论文图导出（单帧全套） ----------------
    def _export_frame_set(self, idx: int, out_dir: str, tag: str, extra: dict = None):
        """
        导出该 idx 的论文全套图：
        - FigA: 2/10/20 三联轨迹对比 + 喇叭口（A4风格）
        - FigB: 2/10/20 三联误差曲线（A4风格）
        同时导出一个 json 记录关键指标。
        """
        df = self.data.df
        idx = int(idx)
        idx = int(np.clip(idx, 1, len(df) - 2))

        t_now = float(df["_tsec"].iloc[idx])
        state = self.predictor.estimate_state_from_history(df, idx, window_sec=60.0)

        # 历史段（为了论文展示，取前后一定长度）
        hist_back_sec = 15 * 60.0
        t_start = t_now - hist_back_sec
        hist = df[(df["_tsec"] >= t_start) & (df["_tsec"] <= t_now)].copy()
        x_hist = hist["_x_m"].values
        y_hist = hist["_y_m"].values

        # 收集每个 horizon 的数据
        pack = {}
        for h_min in self.horizons_min:
            horizon_sec = h_min * 60.0
            t_pred, x_pred, y_pred, psi_pred, v_pred = self.predictor.predict_xy_ctra(state, horizon_sec, dt=self.dt_pred)
            t_true, x_true, y_true, ok = interp_future_truth(df, t_now, horizon_sec, dt=self.dt_pred)
            if not ok:
                continue
            mean_err, rmse, end_err, err_series = compute_errors_xy(x_pred, y_pred, x_true, y_true)

            sigma = sigma_profile(t_pred, v_pred, state["a"], state["omega"], self.data.params, horizon_sec)
            xL, yL, xR, yR = build_cone_boundaries(x_pred, y_pred, psi_pred, sigma)

            pack[h_min] = {
                "t_pred": t_pred, "x_pred": x_pred, "y_pred": y_pred, "psi_pred": psi_pred, "v_pred": v_pred,
                "t_true": t_true, "x_true": x_true, "y_true": y_true,
                "xL": xL, "yL": yL, "xR": xR, "yR": yR,
                "mean": mean_err, "rmse": rmse, "end": end_err,
                "err_series": err_series
            }

        # ---------- FigA：轨迹三联图 ----------
        figA = plt.figure(figsize=(8.27, 11.69), dpi=180)  # A4 竖版
        figA.suptitle(f"{tag} | 轨迹对比+喇叭口（idx={idx}, t={t_now:.1f}s）", fontsize=14)

        for i, h_min in enumerate(self.horizons_min, start=1):
            ax = figA.add_subplot(3, 1, i)
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("东向 y (m)")
            ax.set_ylabel("北向 x (m)")
            ax.set_title(f"{h_min} min：预测(虚线) vs 真实未来(实线) + 置信范围")

            if h_min not in pack:
                ax.text(0.1, 0.5, "真值不足，无法绘制", transform=ax.transAxes, fontsize=12)
                continue

            d = pack[h_min]
            ax.plot(y_hist, x_hist, linewidth=2.0, label="历史(至当前)")
            ax.plot([y_hist[-1]], [x_hist[-1]], "o", label="当前")

            poly_x = np.concatenate([d["xL"], d["xR"][::-1]])
            poly_y = np.concatenate([d["yL"], d["yR"][::-1]])
            ax.fill(poly_y, poly_x, alpha=0.18, label="置信范围")

            ax.plot(d["y_pred"], d["x_pred"], "--", linewidth=2.0, label="预测")
            ax.plot(d["y_true"], d["x_true"], "-", linewidth=2.0, label="真实未来")

            txt = f"Mean={d['mean']:.1f}m  RMSE={d['rmse']:.1f}m  End={d['end']:.1f}m"
            ax.text(0.02, 0.95, txt, transform=ax.transAxes, va="top", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

            y_all = np.concatenate([y_hist, d["y_pred"], d["y_true"]])
            x_all = np.concatenate([x_hist, d["x_pred"], d["x_true"]])
            pad = 80.0
            ax.set_xlim(np.nanmin(y_all) - pad, np.nanmax(y_all) + pad)
            ax.set_ylim(np.nanmin(x_all) - pad, np.nanmax(x_all) + pad)

            ax.legend(loc="best", fontsize=9)

        figA.tight_layout(rect=[0, 0.02, 1, 0.98])

        # ---------- FigB：误差三联图 ----------
        figB = plt.figure(figsize=(8.27, 11.69), dpi=180)
        figB.suptitle(f"{tag} | 误差随时间（idx={idx}, t={t_now:.1f}s）", fontsize=14)

        for i, h_min in enumerate(self.horizons_min, start=1):
            ax = figB.add_subplot(3, 1, i)
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("预测时间 t (s)")
            ax.set_ylabel("位置误差 (m)")
            ax.set_title(f"{h_min} min：误差曲线")

            if h_min not in pack:
                ax.text(0.1, 0.5, "真值不足，无法绘制", transform=ax.transAxes, fontsize=12)
                continue

            d = pack[h_min]
            err = d["err_series"]
            t = d["t_pred"][:len(err)]
            ax.plot(t, err, linewidth=2.0, label="误差")

            txt = f"Mean={d['mean']:.1f}m  RMSE={d['rmse']:.1f}m  End={d['end']:.1f}m"
            ax.text(0.02, 0.95, txt, transform=ax.transAxes, va="top", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))
            ax.legend(loc="best", fontsize=9)

        figB.tight_layout(rect=[0, 0.02, 1, 0.98])

        # ---------- 写文件 ----------
        base = f"{tag}_idx{idx}"
        pngA = os.path.join(out_dir, base + "_A_track.png")
        pdfA = os.path.join(out_dir, base + "_A_track.pdf")
        pngB = os.path.join(out_dir, base + "_B_error.png")
        pdfB = os.path.join(out_dir, base + "_B_error.pdf")

        figA.savefig(pngA, bbox_inches="tight")
        figA.savefig(pdfA, bbox_inches="tight")
        figB.savefig(pngB, bbox_inches="tight")
        figB.savefig(pdfB, bbox_inches="tight")
        plt.close(figA)
        plt.close(figB)

        # 记录 JSON（便于论文写作引用）
        rec = {
            "tag": tag,
            "idx": idx,
            "t_now": t_now,
            "state": {
                "v0_mps": float(state["v"]),
                "a0_mps2": float(state["a"]),
                "omega0_rads": float(state["omega"]),
                "psi0_deg": float(math.degrees(state["psi"]))
            },
            "maneuver_params": self.data.params,
            "metrics": {}
        }
        for h_min in self.horizons_min:
            if h_min in pack:
                rec["metrics"][f"{h_min}min"] = {
                    "mean_m": float(pack[h_min]["mean"]),
                    "rmse_m": float(pack[h_min]["rmse"]),
                    "end_m": float(pack[h_min]["end"])
                }

        if extra is not None:
            rec["extra"] = extra

        json_path = os.path.join(out_dir, base + "_record.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)


def main():
    root = tk.Tk()
    app = HornPaperApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
