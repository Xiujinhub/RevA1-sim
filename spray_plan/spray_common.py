# -*- coding: utf-8 -*-
"""
喷淋规划移植公共层 (MATLAB -> Python -> C++)

【本目录用途】
    这是 `matlab/plot_spray_path_o2e.m` 与 `matlab/clean_rim_contour.m` 的 Python 移植,
    目的**不是**替代 MATLAB, 而是作为 **移植到 C++ (`jetson/task_worker`) 的中间参照**:
        MATLAB (已验证行为)  ->  Python (本目录, 可跑可对账)  ->  C++ (最终实现)
    Python 版与 MATLAB 版**逐字段同口径**(函数名/变量名/输出字段一一对应),
    并刻意写成"**易直译 C++**"的风格:
      - 顺序逻辑用**显式 for 循环**, 不用 numpy 花式广播(别把流程藏起来);
      - 数据结构用 dataclass  <=>  C++ struct;
      - 每个"魔法"调用单独封装, 并注明 C++ 里的等价做法(Eigen 等);
      - 数组统一 np.ndarray(shape=(n,3), float64)  <=>  std::vector<Eigen::Vector3d>。

【本文件 = 公共层】
    - 常量(与 matlab/hand_eye_config.m 同源): TCP / 到达姿态 / 拍照位姿 / standoff / 工艺参数
    - 基础几何: 欧拉角旋转、平面拟合、弧长重采样、多边形与覆盖率工具、环形插值/平滑、
                Kabsch + trimmed ICP、曲线转角
    (MATLAB 里这些函数在 plot_spray_path_o2e.m / make_spray_recon.m 中各存了一份拷贝;
     移植版统一放这里 —— 即 MATLAB README 待办 D7 想做的事。)

【C++ 移植提示速查】
    np.linalg.svd(A, full_matrices=False)  ->  Eigen::JacobiSVD<MatrixXd>(A, ComputeThinU|ComputeThinV)
        !! numpy 返回 Vh(行 = 右奇异向量); MATLAB / Eigen 返回 V(列 = 右奇异向量), 不要混
    np.linalg.norm(v) / np.cross(a,b)      ->  v.norm() / a.cross(b)
    MATLAB mod(x,n) 结果恒非负              ->  自己写 fmod_positive(); C++ 的 % 对负数语义不同
    MATLAB round(x) 四舍五入远离 0          ->  std::round(); Python 的 round() 是银行家舍入, 不可用
    MATLAB 下标从 1 开始                    ->  Python/C++ 从 0 开始(本移植已换算)
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ==================== 常量: 与 matlab/hand_eye_config.m 保持一致 ====================
TOOL_KEYS = ["tool1", "tool2", "nozzle1", "nozzle2"]          # 行序 = 下面各表顺序

# 工具 TCP (尖端相对法兰, mm) —— 即 hand_eye_config.cfg.TCP_mm
TCP_MM = {
    "tool1":   np.array([-2.343855654993024, 0.48717676677189833, 294.7992870405087]),
    "tool2":   np.array([90.85388920106247, 31.02611498131784, 421.45140673414233]),
    "nozzle1": np.array([41.02235917199713, 45.46610134883331, 233.78347492299675]),
    "nozzle2": np.array([58.85311314786616, -9.11954858245911, 236.12713092268872]),
}

# 各工具默认到达姿态 (deg: Rx Ry Rz)
TARGET_ANGLES_DEG = {
    "tool1":   np.array([196.64, 34.90, 111.76]),
    "tool2":   np.array([179.97, -0.82, 282.67]),
    "nozzle1": np.array([196.64, 34.90, 111.76]),
    "nozzle2": np.array([182.2, -0.86, 109.57]),
}

# 获取 o2e 时的机械臂位姿(拍照位姿, 固定)
CAPTURE_T_EE_MM = np.array([110.858, -5.458, 396.436])
CAPTURE_ANGLES_DEG = np.array([196.64, 34.89, 111.77])

# 喷淋工艺间距 (mm): 2026-09-19 由 [.., 86.2, 83.9] 改为 [.., 96.2, 93.9]
#   等效"喷嘴 z 统一 0.33 m"(夹爪余量 35.2mm), 与生产链路一致
SPRAY_STANDOFF_MM = {"tool1": 0.0, "tool2": 0.0, "nozzle1": 96.2, "nozzle2": 93.9}

# ==================== 工艺参数 (与 plot_spray_path_o2e.m 顶部一致) ====================
SPRAY_CONE_DEG = 30.0
SPRAY_OVERLAP = 0.20
EDGE_INSET_MM = 10.0
RING_STANDOFF_MM = 108.2        # 内缩圈单独 standoff (= 喷嘴默认 96.2 + 12); None = 用喷嘴默认
RING_OFFSET_MM = None           # 内缩距离; None = 0.5*D_fp
RIM_DWELL_S = 0.8
RASTER_DWELL_S = 0.8
RASTER_TOP_TO_BOTTOM = True
RIM_SPLIT_HALVES = True
COVER_CELL_MM = 4.0
CLEAR_WARN_MM = 20.0
ARRIVAL_ANGLES = np.array([173.45, -26.81, 287.50])
NOZZLE_TOOL = "nozzle1"
GRIPPER_TOOL = "tool1"
PASS_ORDER = ["rim_left", "rim_right", "ring", "raster"]

# 沿口清洗参数 (与 plot_spray_path_o2e.m / make_spray_recon.m 同口径)
CLEAN_RIM = True
CLEAN_RIM_BIN_MM = 6.0
CLEAN_RIM_TPL_FILE = "urinal_rim_template_o2e.json"   # 相对 matlab/datas/
CLEAN_RIM_TPL_TOL = 12.0
CLEAN_RIM_TURN_SOFT = 22.0
CLEAN_RIM_SMOOTH_WIN = 5


# ==================== 基础数学 (C++: 自写 / Eigen) ====================
def round_matlab(x: float) -> int:
    """MATLAB round(x): 四舍五入且远离 0。

    !! Python 内建 round() 是银行家舍入(round(2.5)=2), 不能替代;
    C++ 用 std::round(x) 即与 MATLAB 一致。
    """
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def mod_positive(x: float, n: int) -> int:
    """MATLAB mod(x, n): 结果恒非负; C++ 的 % 对负数会返回负值, 移植时必须自己写"""
    return int(math.fmod(math.fmod(x, n) + n, n))


def euler_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """XYZ 外旋: R = Rz(rz)·Ry(ry)·Rx(rx) (与 hand_eye_transform.euler_to_R 一致)"""
    cx, sx = math.cos(math.radians(rx_deg)), math.sin(math.radians(rx_deg))
    cy, sy = math.cos(math.radians(ry_deg)), math.sin(math.radians(ry_deg))
    cz, sz = math.cos(math.radians(rz_deg)), math.sin(math.radians(rz_deg))
    Rx = np.array([[1.0, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1.0, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
    return Rz @ Ry @ Rx


@dataclass
class PlaneFrame:
    """盆口平面与面内坐标 (MATLAB: c0 / n_pl / u_long / v_short / uc / vc / uv / ang / rad / d_pl)"""
    c0: np.ndarray      # (3,)   面内质心
    n: np.ndarray       # (3,)   法向
    u: np.ndarray       # (3,)   面内长轴
    v: np.ndarray       # (3,)   面内短轴
    uc: float
    vc: float
    uv: np.ndarray      # (n,2)  面内坐标
    ang: np.ndarray     # (n,)   面内极角 (rad)
    rad: np.ndarray     # (n,)   面内半径
    d_pl: np.ndarray    # (n,)   相对拟合平面的法向偏差


def plane_fit(points: np.ndarray, fix_signs: bool = False,
              robot_ref: Optional[np.ndarray] = None) -> PlaneFrame:
    """最小二乘平面 + 面内极坐标 (MATLAB 里"平面与面内坐标"那一段)

    fix_signs=False 与**旧版** clean_rim_contour.m 一致(不修正符号);
    fix_signs=True  与 plot_spray_path_o2e.m(:313-323) 及 clean_rim_contour.m(2026-09-19 定型后)
                    一致 —— n 朝机器人(robot_ref), u 朝基座 +Z。
    ⚠ 特征向量符号是任意的(MATLAB svd / C++ jacobiEigen3 规则不同): 不定型时重建沿口/内缩圈的
      "起点/绕向"会随数据翻转(现象: ring 起点跑到另一端, 看着像上下颠倒)。**定型的用途就是消掉它**。
    robot_ref 缺省 = 标定拍照位姿 CAPTURE_T_EE_MM(等价 MATLAB 里传 T_EE_MM)。
    C++: Eigen::JacobiSVD<MatrixXd>(A, ComputeThinU | ComputeThinV), 取 V 的第 1/3 **列**。
    """
    p = np.asarray(points, dtype=float)
    c0 = p.mean(axis=0)
    _, _, vh = np.linalg.svd(p - c0, full_matrices=False)   # vh 的“行”= 右奇异向量
    n_pl = vh[2].copy()
    u_long = vh[0].copy()
    if fix_signs:
        t_ee = CAPTURE_T_EE_MM if robot_ref is None else np.asarray(robot_ref, float).ravel()
        if float(n_pl @ (t_ee - c0)) < 0:
            n_pl = -n_pl
        if u_long[2] < 0:
            u_long = -u_long
    v_short = np.cross(n_pl, u_long)
    v_short = v_short / np.linalg.norm(v_short)
    uv = np.c_[p @ u_long - c0 @ u_long, p @ v_short - c0 @ v_short]
    uc, vc = float(uv[:, 0].mean()), float(uv[:, 1].mean())
    ang = np.arctan2(uv[:, 1] - vc, uv[:, 0] - uc)
    rad = np.hypot(uv[:, 0] - uc, uv[:, 1] - vc)
    d_pl = p @ n_pl - c0 @ n_pl
    return PlaneFrame(c0=c0, n=n_pl, u=u_long, v=v_short, uc=uc, vc=vc,
                      uv=uv, ang=ang, rad=rad, d_pl=d_pl)


# ==================== 环形序列工具 (对应 MATLAB fill_circular / movmean_circ) ====================
def fill_circular(v: np.ndarray) -> np.ndarray:
    """环形向量里的 NaN 用左右最近有效值线性插值补齐 (C++: 找前后有效索引线性插值)"""
    x = np.asarray(v, dtype=float).ravel().copy()
    k = x.size
    ok = np.flatnonzero(~np.isnan(x))
    if ok.size == 0:
        return np.zeros(k)
    if ok.size == 1:
        return np.full(k, x[ok[0]])
    t = np.arange(k)
    xk = np.concatenate([ok - k, ok, ok + k])
    vk = np.tile(x[ok], 3)
    return np.interp(t, xk, vk)


def movmean_circ(v: np.ndarray, w: int) -> np.ndarray:
    """环形移动平均, 窗口 = 2*floor(w/2)+1 (与 MATLAB movmean_circ 一致)"""
    x = np.asarray(v, dtype=float).ravel()
    n = x.size
    h = int(math.floor(w / 2))
    idx = (np.arange(n)[:, None] + np.arange(-h, h + 1)[None, :]) % n
    return x[idx].mean(axis=1)


# ==================== 多边形 / 覆盖率工具 (对应 MATLAB 局部函数, 逐行移植) ====================
def poly_area(p2: np.ndarray) -> float:
    """多边形面积 (有序点集, 允许非凸) —— C++: 鞋带公式手写循环即可"""
    x = p2[:, 0]
    y = p2[:, 1]
    return float(abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))) / 2.0)


def in_polygon_mask(p2: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """点集是否在多边形内部 (射线法, 支持非凸) —— C++: 按下面同一个循环写"""
    x = p2[:, 0]
    y = p2[:, 1]
    n = poly.shape[0]
    m = np.zeros(x.shape[0], dtype=bool)
    j = n - 1
    for i in range(n):                      # MATLAB: j = n, i = 1..n  -> 0-based 平移
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + np.finfo(float).eps) + xi)
        m = np.logical_xor(m, cond)
        j = i
    return m


def dist_to_poly(p2: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """各点到多边形边界的最短距离 (mm)"""
    n = poly.shape[0]
    d = np.full(p2.shape[0], np.inf)
    for i in range(n):
        a = poly[i, :]
        b = poly[(i + 1) % n, :]            # MATLAB: mod(i, n) + 1
        ab = b - a
        l2 = float(ab @ ab)
        if l2 < np.finfo(float).eps:
            continue
        tt = ((p2 - a) @ ab) / l2
        tt = np.clip(tt, 0.0, 1.0)
        proj = a + tt[:, None] * ab
        d = np.minimum(d, np.linalg.norm(p2 - proj, axis=1))
    return d


def poly_scanline(poly: np.ndarray, x0: float) -> np.ndarray:
    """直线(第 1 坐标 == x0)与多边形的交点, 返回交点第 2 坐标 (升序)"""
    n = poly.shape[0]
    cr = []
    for i in range(n):
        j = (i + 1) % n
        xa, ya = poly[i, 0], poly[i, 1]
        xb, yb = poly[j, 0], poly[j, 1]
        if xa != xb and (xa - x0) * (xb - x0) <= 0:
            t = (x0 - xa) / (xb - xa)
            cr.append(ya + t * (yb - ya))
    return np.sort(np.array(cr, dtype=float))


def resample_arclen(p: np.ndarray, s_arc: np.ndarray, s_q: np.ndarray) -> np.ndarray:
    """按弧长重采样: P 有序点列, s_arc 累计弧长, s_q 目标弧长 (线性插值)"""
    q = np.zeros((len(s_q), 3))
    for i, sq in enumerate(s_q):
        s = min(max(float(sq), 0.0), float(s_arc[-1]))
        k = int(np.searchsorted(s_arc, s, side="left"))      # MATLAB: find(s_arc >= s, 1)
        if k <= 0:
            q[i, :] = p[0, :]
        else:
            s0, s1 = float(s_arc[k - 1]), float(s_arc[k])
            w = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
            q[i, :] = p[k - 1, :] * (1 - w) + p[k, :] * w
    return q


def seg_point_dist(a: np.ndarray, b: np.ndarray, pts: np.ndarray) -> Tuple[float, int]:
    """线段 A->B 到点集 P (N×3) 的最近距离与最近点号 (0-based) —— 干涉判据核心"""
    if pts.size == 0:
        return float("nan"), -1
    ab = b - a
    l2 = float(ab @ ab)
    if l2 < np.finfo(float).eps:
        dd = np.linalg.norm(pts - a, axis=1)
    else:
        tt = ((pts - a) @ ab) / l2
        tt = np.clip(tt, 0.0, 1.0)
        proj = a + tt[:, None] * ab
        dd = np.linalg.norm(pts - proj, axis=1)
    imin = int(np.argmin(dd))
    return float(dd[imin]), imin


def footprint_on_plane(apex: np.ndarray, dvec: np.ndarray, half_ang_deg: float,
                       p0: np.ndarray, nvec: np.ndarray, nseg: int = 72) -> np.ndarray:
    """锥体(顶点 apex, 轴 dvec, 半角 half_ang_deg)与平面(P0, 法向 nvec)的交线 = 喷淋足迹"""
    apex = np.asarray(apex, dtype=float).ravel()
    dvec = np.asarray(dvec, dtype=float).ravel()
    dvec = dvec / np.linalg.norm(dvec)
    nvec = np.asarray(nvec, dtype=float).ravel()
    e1 = np.cross(dvec, nvec)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(dvec, np.array([1.0, 0, 0]))
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(dvec, np.array([0, 1.0, 0]))
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(dvec, e1)
    ca, sa = math.cos(math.radians(half_ang_deg)), math.sin(math.radians(half_ang_deg))
    fp = np.zeros((nseg, 3))
    ok = True
    for i in range(nseg):
        ph = 2.0 * math.pi * i / nseg                       # MATLAB: (i-1)/nseg
        dirv = ca * dvec + sa * (math.cos(ph) * e1 + math.sin(ph) * e2)
        den = float(dirv @ nvec)
        if abs(den) < 1e-9:
            ok = False
            break
        tq = float((p0 - apex) @ nvec) / den
        if tq <= 0:
            ok = False
            break
        fp[i, :] = apex + tq * dirv
    return fp if ok else np.zeros((0, 3))


def coverage_mask(fp_list: Sequence[np.ndarray], cell_uv: np.ndarray, in_open: np.ndarray,
                  c0: np.ndarray, u_long: np.ndarray, v_short: np.ndarray) -> np.ndarray:
    """足迹列表覆盖了哪些盆口单元 (只在盆口多边形内累计; 包围盒预筛加速)"""
    cover = np.zeros(in_open.shape[0], dtype=bool)
    for fp in fp_list:
        if fp is None or fp.size == 0:
            continue
        d_fp = fp - c0
        fp_uv = np.c_[d_fp @ u_long, d_fp @ v_short]
        bm = ((cell_uv[:, 0] >= fp_uv[:, 0].min()) & (cell_uv[:, 0] <= fp_uv[:, 0].max())
              & (cell_uv[:, 1] >= fp_uv[:, 1].min()) & (cell_uv[:, 1] <= fp_uv[:, 1].max()))
        if not bm.any():
            continue
        sub = np.zeros(in_open.shape[0], dtype=bool)
        sub[bm] = in_polygon_mask(cell_uv[bm, :], fp_uv)
        cover = cover | (sub & in_open)
    return cover


# ==================== 刚体对齐 (模板) 与沿口专用工具 (对应 clean_rim_contour.m) ====================
def kabsch(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """求刚体变换 (R, t) 使 R·A + t ≈ B (A/B 为对应点, p×3)

    C++: Eigen::Umeyama / 自写 SVD —— 注意 det(V·Uᵀ) 的符号修正(防镜像)。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    d = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(vt.T @ u.T)))])
    r = vt.T @ d @ u.T
    return r, cb - r @ ca


def icp_kabsch(tpl: np.ndarray, target: np.ndarray, n_it: int = 30) -> Tuple[np.ndarray, float]:
    """模板点集 T (m×3) 刚体对齐到目标点集 X (n×3): trimmed ICP + Kabsch

    保留比例从 60% 逐步放宽到 80% (与 MATLAB icp_kabsch 完全一致)。
    返回 (对齐后的模板点, 末轮保留点的 RMS); RMS 为 NaN 表示未成功迭代。
    C++: 最近邻暴力 O(m·n) 即可(模板 ~700 点 x 目标 ~500 点), 或用 KD-Tree。
    """
    tpl = np.asarray(tpl, dtype=float)
    x = np.asarray(target, dtype=float)
    tout = tpl + (x.mean(axis=0) - tpl.mean(axis=0))     # 初始化: 只做质心对齐
    rms = float("nan")
    for it in range(1, max(1, int(n_it)) + 1):
        d2 = ((tout[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1)
        jmin = d2.argmin(axis=1)
        dmin = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
        keep_frac = 0.6 + 0.2 * min(1.0, it / 10.0)
        ds = np.sort(dmin)
        thr = ds[max(1, round_matlab(keep_frac * ds.size)) - 1]
        keep = dmin <= thr
        if int(keep.sum()) < 6:
            break
        r, t = kabsch(tout[keep], x[jmin[keep]])
        tout = (r @ tout.T).T + t
        rms = float(np.sqrt((dmin[keep] ** 2).mean()))
    return tout, rms


def build_rim_curve(c0: np.ndarray, n_pl: np.ndarray, u_long: np.ndarray, v_short: np.ndarray,
                    uc: float, vc: float, th: np.ndarray, r: np.ndarray, d: np.ndarray) -> np.ndarray:
    """由 (θ 网格, 面内半径 r, 法向偏差 d) 生成 3D 闭合沿口曲线

    P = c0 + (uc + r·cosθ)·u + (vc + r·sinθ)·v + d·n   (对应 MATLAB build_rim_curve)
    """
    out = np.zeros((th.size, 3))
    for i in range(th.size):
        out[i, :] = (c0 + (uc + r[i] * math.cos(th[i])) * u_long
                     + (vc + r[i] * math.sin(th[i])) * v_short + d[i] * n_pl)
    return out


def poly_metrics(uv: np.ndarray, dseg: np.ndarray) -> Tuple[float, float]:
    """面内多边形面积与闭合周长 (对应 MATLAB poly_metrics)"""
    x = uv[:, 0]
    y = uv[:, 1]
    a = abs(float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))) / 2.0
    perim = float(np.sum(dseg)) + float(np.linalg.norm(uv[-1] - uv[0]))
    return a, perim


def half_ratio(p: np.ndarray) -> float:
    """左右半圈弧长比 (以面内 u 极值为界, v>0 侧为左半) —— 与 MATLAB half_ratio 一致"""
    p = np.asarray(p, dtype=float)
    c = p.mean(axis=0)
    _, _, vh = np.linalg.svd(p - c, full_matrices=False)
    n = vh[2]
    u = vh[0]
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    uv = np.c_[p @ u - c @ u, p @ v - c @ v]
    it, ib = int(np.argmax(uv[:, 0])), int(np.argmin(uv[:, 0]))
    nc = p.shape[0]
    if it <= ib:
        ia = np.arange(it, ib + 1)
        ib_idx = np.concatenate([np.arange(ib, nc), np.arange(0, it + 1)])
    else:
        ia = np.concatenate([np.arange(it, nc), np.arange(0, ib + 1)])
        ib_idx = np.arange(ib, it + 1)
    pa, pb = p[ia], p[ib_idx]
    if uv[ia, 1].mean() < uv[ib_idx, 1].mean():
        pa, pb = pb, pa
    la = float(np.linalg.norm(np.diff(pa, axis=0), axis=1).sum()) + float(np.linalg.norm(p[ia[-1]] - p[ia[0]]))
    lb = float(np.linalg.norm(np.diff(pb, axis=0), axis=1).sum()) + float(np.linalg.norm(p[ib_idx[-1]] - p[ib_idx[0]]))
    return la / max(lb, 1e-12)


def curve_turns(p: np.ndarray, closed: bool = True) -> np.ndarray:
    """曲线相邻段转角 (deg); closed=True 时首尾相接 (对应 MATLAB curve_turns)"""
    q = np.vstack([p, p[:1]]) if closed else np.asarray(p, dtype=float)
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    dirv = np.diff(q, axis=0) / np.maximum(seg[:, None], 1e-12)
    cs = (dirv[:-1] * dirv[1:]).sum(axis=1)
    return np.degrees(np.arccos(np.clip(cs, -1.0, 1.0)))

