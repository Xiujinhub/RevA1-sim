#!/usr/bin/env python3
"""点云 → 机械臂路径：**现场取边规划** + MuJoCo 仿真回放。

两条链路
--------
**① 现场取边（默认；与 `python/toilet/toilet_plan_from_cloud.py` 同一套实现）**

``toilet_plan.py`` 与 ``spray_plan/`` 是从现场脚本**原样拷过来**的（只改了共享层路径 +
NumPy 2 的 ``ptp``），所以取边、走法、喷距、法兰位姿、覆盖率都和现场一致：

* **取边**：红 = 盆口边缘、青 = 收缩后边缘 → 在点云里逐角向找"视线深度的局部极小 = 脊线"，
  得到**实测边缘点**（不是检测折线、也不是投影外轮廓）；平面/面内坐标定型（n 朝机器人、u 朝 +Z）；
* **走法**：内缩圈（沿内法向偏移 40 mm，越界自动半距/径向回退）+ 栅格
  （短轴蛇形、上→下、贴边 10 mm、剔除不足 2 点的尾巴线）；可选唇遍（沿收缩边两半圈）；
* **喷距**：``auto`` = 拍到盆口的轴向距离，再按"足迹 ≤ 35%×口短边"收紧；
  足迹 ``D = 2·standoff·tan(15°)``，间距 ``s = D·(1-20%)``；
* **法兰位姿**：固定到达姿态 ``ARRIVAL_ANGLES``：
  ``P_flange = P_target - so·ax - R·TCP``，喷嘴尖 = ``P_flange + R·TCP``；
* **指标**：覆盖率、工具轴线到点云的最小间隙、逐遍航点/覆盖率摘要。

本模块接着做法兰**全位姿 IK**：位置 + 工具轴（EE z）+ 绕工具轴的滚转（J6），
再按关节路点回放（工具尖留轨迹）。与现场 MoveS 的差别只有：仿真里不逐点停 0.8 s。

**② 通用轮廓（fallback）**

点云里没有红/青检测边时（sink / 小便池扁平点云等），用点云自身投影外轮廓生成
环形/螺旋/栅格，并把路径点吸附回点云。算法与现场无关，只作为兜底。

命令行
------
::

    python arm_path.py --cloud point_cloud/toilet-1.json              # 现场取边：规划 + 报告
    python arm_path.py --cloud point_cloud/toilet-1.json --sim        # 再加 IK + 回放
    python arm_path.py --cloud point_cloud/toilet-1.json --lip        # 加唇遍（沿收缩边）
    python arm_path.py --cloud point_cloud/toilet-1.json --mode ring  # 只走内缩圈
    python arm_path.py --cloud point_cloud/sink_1_o2e.json --engine outline   # 强制通用轮廓
    python arm_path.py --cloud point_cloud/toilet-1.json --render runs/plan.png --json runs/plan.json

界面入口：``revA1_gui.py`` 的「点云 → 路径仿真」卡片（算法下拉框：自动 / 现场取边 / 通用轮廓）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

import arm_core as core
import point_cloud as pc
import revA1_spec as spec

try:
    import mujoco
except Exception:                                     # noqa: BLE001
    mujoco = None


# =============================================================== 参数 / 结果
@dataclass
class PlanSettings:
    """一次路径规划的工艺参数（界面上的控件就对应这些字段）。"""

    engine: str = "auto"               # auto / field（现场取边）/ outline（通用轮廓）
    mode: str = "ring+raster"          # ring / spiral / raster / ring+raster
    standoff_mm: float = 0.0           # 0 = 自动（现场取边按 footprint 收紧；通用轮廓用 400）
    step_mm: float = 0.0               # 0 = 自动（现场取边按喷锥足迹；通用轮廓用 40）
    inset_mm: float = 10.0             # 轮廓内缩（通用轮廓用；现场取边固定 EDGE_INSET_MM=10）
    snap_mm: float = 40.0              # 目标点吸附到真实点云的搜索半径（通用轮廓用）
    region_radius_mm: float = 360.0    # 只规划"离点云中位点这么近"的部分（通用轮廓用）
    normal_radius_mm: float = 40.0     # 估局部法向的邻域半径（通用轮廓用）
    max_reach_mm: float = 1040.0       # 工具尖离基座的最大半径（通用轮廓用）
    min_reach_mm: float = 320.0        # 太近也不规划（通用轮廓用）
    max_points: int = 420              # 路径点上限（界面卡顿时调小）
    approach_mm: float = 150.0         # 起点前先退这么远（接近点）
    speed_mm_s: float = 160.0          # 仿真速度：路径长度 / 速度 = 运动时间
    min_seg_s: float = 0.05            # 单段最短时间
    max_seg_s: float = 0.80            # 单段最长时间
    ik_pos_tol_mm: float = 2.0         # IK 位置容差
    ik_axis_tol_deg: float = 4.0       # IK 姿态容差
    ik_iters: int = 180                # 单次 IK 迭代上限
    ik_max_jump_deg: float = 180.0     # 相邻路径点允许的最大关节跳变（180 = 不限制；解支切换交给转场动画）
    seed_min_points: int = 12          # 检测种子少于这么多点就不用
    seed_tol_mm: float = 40.0          # 种子点与点云对齐的容差（验证种子是不是同一帧）
    # ---- 现场取边（toilet_plan.py 原样移植）-------------------------------
    lip: bool = False                  # 唇遍（沿收缩边 rim_left/rim_right；现场默认不生成）
    ring_offset_mm: float = 40.0       # 内缩圈的内缩量（现场 --ring-offset 默认 40）
    ring_standoff_mm: float = -1.0     # <0 = 与其它遍相同（现场 --ring-standoff 默认 None）
    footprint_frac: float = 0.35       # auto 喷距时"足迹 <= F×口短边"（现场默认 0.35）
    drop_hl: str = "all"               # 取边工作点云丢掉哪些染色点（现场 --drop-hl 默认 all）
    arrival_angles: tuple = (173.45, -26.81, 287.50)   # 到达姿态（现场喷淋链路）
    full_pose_ik: bool = True          # 法兰全位姿 IK（位置+工具轴+关节6滚转）；False = 只做位置+轴


@dataclass
class PathPlan:
    """规划结果：目标点 / 法向 / 工具尖 / 关节角 / 可执行分段 + 报告。"""

    targets: np.ndarray                    # (N,3) 目标点（基座系 m，已吸附到点云）
    normals: np.ndarray                    # (N,3) 目标点法向（朝机器人）
    tcp: np.ndarray                        # (N,3) 工具尖目标 = 目标点 + n·standoff
    q: np.ndarray                          # (N,6) IK 关节角（失败行保留上一支 / 全 NaN）
    ok: np.ndarray                         # (N,)  是否解出来了
    seconds: np.ndarray                    # (N,)  上一段 → 这一段的时长
    on_cloud: np.ndarray                   # (N,)  是否吸附到了真实点云
    runs: list[np.ndarray] = field(default_factory=list)      # 连续可执行的关节路点
    run_seconds: list[np.ndarray] = field(default_factory=list)
    rim: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    u: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    v: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    radius_mm: float = 0.0
    seed_source: str = ""
    report: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    ms: float = 0.0

    @property
    def n(self) -> int:
        return int(self.targets.shape[0])

    @property
    def n_ok(self) -> int:
        return int(np.count_nonzero(self.ok))

    @property
    def path_length_m(self) -> float:
        if self.tcp.shape[0] < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.tcp, axis=0), axis=1).sum())

    @property
    def total_seconds(self) -> float:
        return float(sum(float(s.sum()) for s in self.run_seconds))

    def summary(self) -> str:
        pct = 100.0 * self.n_ok / max(self.n, 1)
        return (f"路径 {self.n} 点（IK 可解 {self.n_ok} / {pct:.0f}%），"
                f"工具尖路程 {self.path_length_m:.2f} m，仿真时长 {self.total_seconds:.1f} s，"
                f"轮廓直径 ~{2.0 * self.radius_mm / 1000.0:.2f} m")


# =============================================================== 小工具
def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=float).ravel()
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else a


def voxel_downsample(points, cell_mm: float) -> np.ndarray:
    """体素抽稀（每个体素留一个点）：几十万点降到几万点，规划快一个量级。"""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0 or float(cell_mm) <= 0:
        return pts
    keys = np.floor(pts / float(cell_mm)).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def pca_plane(points):
    """点集 → ``(质心, 法向, 面内长轴)``（最小二乘平面）。"""
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    c = p.mean(axis=0)
    if p.shape[0] < 3:
        return c, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    _, _, vh = np.linalg.svd(p - c, full_matrices=False)
    return c, _unit(vh[2]), _unit(vh[0])


class PointGrid:
    """体素哈希的最近点 / 邻域查询（不依赖 scipy 的 cKDTree）。"""

    def __init__(self, points, cell_mm: float):
        self.pts = np.asarray(points, dtype=float).reshape(-1, 3)
        self.cell = float(max(cell_mm, 1e-6))
        self.grid: dict = {}
        keys = np.floor(self.pts / self.cell).astype(np.int64)
        for i, k in enumerate(keys):
            self.grid.setdefault((int(k[0]), int(k[1]), int(k[2])), []).append(i)
        self._offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                      for dz in (-1, 0, 1)]

    def candidates(self, q) -> list[int]:
        b = np.floor(np.asarray(q, dtype=float).ravel() / self.cell).astype(np.int64)
        out: list[int] = []
        for o in self._offs:
            out += self.grid.get((int(b[0] + o[0]), int(b[1] + o[1]),
                                  int(b[2] + o[2])), [])
        return out

    def nearest(self, q, radius_mm: float):
        """返回 ``(最近点, 距离)``；``radius_mm`` 内没有就 ``(None, inf)``。"""
        cand = self.candidates(q)
        if not cand:
            return None, float("inf")
        p = self.pts[cand]
        d = np.linalg.norm(p - np.asarray(q, dtype=float).ravel(), axis=1)
        k = int(np.argmin(d))
        if float(d[k]) > float(radius_mm):
            return None, float("inf")
        return p[k], float(d[k])

    def around(self, q, radius_mm: float) -> np.ndarray:
        cand = self.candidates(q)
        if not cand:
            return np.zeros((0, 3))
        p = self.pts[cand]
        d = np.linalg.norm(p - np.asarray(q, dtype=float).ravel(), axis=1)
        return p[d <= float(radius_mm)]


def _circ_fill(values, bins: int) -> np.ndarray:
    """环形数组里的 NaN 用左右最近有效值线性插值补齐。"""
    x = np.asarray(values, dtype=float).ravel().copy()
    ok = np.flatnonzero(np.isfinite(x))
    if ok.size == 0:
        return np.zeros(bins)
    if ok.size == 1:
        return np.full(bins, x[ok[0]])
    t = np.arange(bins)
    xk = np.concatenate([ok - bins, ok, ok + bins])
    vk = np.tile(x[ok], 3)
    return np.interp(t, xk, vk)


def _circ_smooth(values, win: int = 3) -> np.ndarray:
    """环形移动平均（``win`` 为半窗）。"""
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    if n == 0 or win <= 0:
        return x
    idx = (np.arange(n)[:, None] + np.arange(-win, win + 1)[None, :]) % n
    return x[idx].mean(axis=1)


# =============================================================== 读点云 + 种子
def load_inputs(cloud_path, *, pose=pc.DEFAULT_POSE, extrinsic=pc.DEFAULT_EXTRINSIC,
                intrinsic=pc.DEFAULT_INTRINSIC, frame: str = "auto", ref: str = "tcp",
                tool_offset_mm: float | None = None, offset_mm=(0.0, 0.0, 0.0),
                yaw_deg: float = 0.0, stride: int = 1,
                depth_min_mm: float = pc.MIN_DEPTH_MM) -> dict:
    """读点云 → 基座系点 + 检测种子（都换算到基座系），并给出相机位置。

    参数和 :func:`point_cloud.build_cloud_scene` 同一套（拍摄姿态 / 手眼标定 /
    参考点 / 手动微调），所以**规划用的点云和画面里看到的点云是同一份**。
    返回 ``dict(cloud, points, seeds, camera, frame, pose, R, t)``；
    ``seeds`` 里每组是 ``dict(label/kind/target/closed/frame/points)``，``points`` 已换成米。
    """
    if tool_offset_mm is None:
        try:
            tool_offset_mm = float(spec.TOOL_TIP_LOCAL_Z) * 1000.0
        except Exception:                              # noqa: BLE001
            tool_offset_mm = 42.7
    cloud = pc.load_cloud(cloud_path, intrinsic_path=intrinsic, stride=stride,
                          depth_min_mm=depth_min_mm)
    R, t = pc.load_extrinsic(extrinsic)
    pose6 = pc.parse_pose6(pose)
    use = frame
    if not frame or str(frame).lower() in ("auto", "detect"):
        use = pc.detect_cloud_frame(cloud, R=R, t=t, intrinsic_path=intrinsic)["frame"]
    else:
        use = pc.normalize_frame(frame) or "cam"
    kw = dict(ref=ref, tool_offset_mm=float(tool_offset_mm),
              offset_mm=offset_mm, yaw_deg=yaw_deg)
    points = pc.transform_to_base(cloud.points_mm, R, t, pose6, frame=use, **kw)
    seeds = []
    for g in (cloud.meta.get("seed_groups") or []):
        pts_mm = np.asarray(g["points"], dtype=float).reshape(-1, 3)
        if pts_mm.shape[0] < 3:
            continue
        base = pc.transform_to_base(pts_mm, R, t, pose6,
                                    frame=pc.normalize_frame(g.get("frame")) or "cam", **kw)
        seeds.append(dict(label=g.get("label", ""), kind=g.get("kind", ""),
                          target=g.get("target", ""), closed=bool(g.get("closed")),
                          rgb=tuple(g.get("rgb") or ()), frame=g.get("frame", ""),
                          points=base))
    camera = pc.camera_in_base(R, t, pose6, ref=ref, tool_offset_mm=float(tool_offset_mm),
                               offset_mm=offset_mm, yaw_deg=yaw_deg)
    cloud.frame = use
    return dict(cloud=cloud, points=points, seeds=seeds, camera=camera,
                frame=use, pose=pose6, R=R, t=t)


# =============================================================== 取轮廓
def _frame_from_normal(n, up=(0.0, 0.0, 1.0)):
    """给定法向 → 面内 ``(u, v)``：u 尽量朝"上"，v = n × u。"""
    n = _unit(n)
    upv = np.asarray(up, dtype=float).ravel()
    u = upv - float(upv @ n) * n
    if float(np.linalg.norm(u)) < 1e-6:               # 法向竖直时换一个参考轴
        alt = np.array([0.0, 1.0, 0.0])
        u = alt - float(alt @ n) * n
    u = _unit(u)
    return u, _unit(np.cross(n, u))


def _pick_seed(seeds, grid: PointGrid, settings: PlanSettings):
    """从检测种子挑一条"闭合边缘线"当轮廓；和点云对不上（不同帧）就丢。"""
    best, best_score = None, -1.0
    for g in (seeds or []):
        pts = np.asarray(g.get("points", []), dtype=float).reshape(-1, 3)
        if pts.shape[0] < int(settings.seed_min_points):
            continue
        step = max(pts.shape[0] // 48, 1)
        tol = float(settings.seed_tol_mm) / 1000.0
        d = [grid.nearest(p, 2.0 * tol)[1] for p in pts[::step][:48]]
        hit = float(np.mean(np.asarray(d) <= tol))
        if hit < 0.6:                                 # 种子点不在点云上 → 坐标系/帧对不上
            continue
        score = (2.0e6 if g.get("closed") else 0.0) + (5.0e5 if g.get("kind") == "edge" else 0.0)
        score += 1.0e5 if g.get("target") else 0.0
        score += float(pts.shape[0])
        if score > best_score:
            best, best_score = dict(g), score
    return best


def _outline_from_points(uv, nbins: int = 72, pct: float = 90.0, win: int = 2,
                         min_pts: int = 3):
    """面内点 → 极坐标轮廓 ``(theta_deg, r_mm)``（每角向取分位数，NaN 补齐 + 平滑）。

    ``min_pts``：每个角向至少几个点才算数。稠密点云用 3；**检测边缘线常常只有 20 来个点**，
    这时要放宽到 1（否则一个角向都凑不齐 → 轮廓全 0）。
    """
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    rad = np.hypot(uv[:, 0], uv[:, 1])
    ang = np.degrees(np.arctan2(uv[:, 1], uv[:, 0]))
    mids = -180.0 + (np.arange(nbins) + 0.5) * (360.0 / nbins)
    r = np.full(nbins, np.nan)
    for i, a0 in enumerate(mids):
        m = (ang >= a0 - 180.0 / nbins) & (ang < a0 + 180.0 / nbins)
        if int(m.sum()) >= int(min_pts):
            r[i] = float(np.percentile(rad[m], pct))
    r = _circ_fill(r, nbins)
    return mids, _circ_smooth(r, win)


def _profile_at(theta_deg, mids, values):
    """环形剖面在任意角度上的线性插值（自动跨 ±180°）。"""
    th = np.asarray(theta_deg, dtype=float)
    return np.interp(th, np.r_[mids - 360.0, mids, mids + 360.0],
                     np.r_[values, values, values])


def _build_frame(points, *, seeds, settings: PlanSettings, camera=None, robot=None):
    """定"作业面"：中心 / 法向 / 面内轴 / 极坐标轮廓 / 沿法向的高度剖面。

    有检测种子 → 用种子（闭合边线）拟合平面和半径，点云只用来核对对齐；
    没有种子 → 用点云本身：以中位点为锚，朝机器人一侧取"视线方向"当法向，
    投影后按角向分位数取外轮廓。
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    robot = np.zeros(3) if robot is None else np.asarray(robot, dtype=float).ravel()
    anchors = np.median(pts, axis=0)
    # 只留锚点附近的一块（滤掉地面/墙/远处的背景）；点云单位是米
    near = pts[np.linalg.norm(pts - anchors, axis=1)
               <= float(settings.region_radius_mm) / 1000.0]

    seed = _pick_seed(seeds, PointGrid(near, max(settings.seed_tol_mm / 1000.0, 0.01)),
                      settings) if seeds else None
    if seed is not None:
        rim = np.asarray(seed["points"], dtype=float).reshape(-1, 3)
        c, n, _ = pca_plane(rim)
        if float(n @ (robot - c)) < 0:
            n = -n
        u, v = _frame_from_normal(n)
        rel = rim - c
        uv = np.c_[rel @ u, rel @ v]
        nb = int(np.clip(rim.shape[0] // 2, 12, 72))     # 20 点的轮廓别硬分 72 个角向
        mids, r = _outline_from_points(uv, nbins=nb, pct=50.0, win=2, min_pts=1)
        dn = rel @ n
        ang = np.degrees(np.arctan2(uv[:, 1], uv[:, 0]))
        d = np.full(nb, np.nan)
        half = 180.0 / nb
        for i, a0 in enumerate(mids):
            m = (ang >= a0 - half) & (ang < a0 + half)
            if int(m.sum()) >= 1:
                d[i] = float(np.median(dn[m]))
        d = _circ_smooth(_circ_fill(d, nb), 2)
        src = (f"检测种子「{seed.get('label') or seed.get('kind')}」"
               f"（{rim.shape[0]} 点，{'闭合' if seed.get('closed') else '非闭合'}）")
    else:
        if camera is not None:
            n = -_unit(anchors - np.asarray(camera, dtype=float).ravel())
        else:
            n = -_unit(anchors - robot)
        if float(n @ (robot - anchors)) < 0:
            n = -n
        c = anchors.copy()
        u, v = _frame_from_normal(n)
        rel = near - c
        uv = np.c_[rel @ u, rel @ v]
        mids, r = _outline_from_points(uv, nbins=72, pct=92.0, win=3)
        # 外圈一带的真实高度：给轮廓每个角向一个沿法向的偏移（贴合弧面）
        rad = np.hypot(uv[:, 0], uv[:, 1])
        ang = np.degrees(np.arctan2(uv[:, 1], uv[:, 0]))
        dn = rel @ n
        d = np.full(72, np.nan)
        for i, a0 in enumerate(mids):
            m = ((ang >= a0 - 2.5) & (ang < a0 + 2.5)
                 & (np.abs(rad - r[i]) <= 0.15 * max(r[i], 0.001)))
            if int(m.sum()) >= 3:
                d[i] = float(np.median(dn[m]))
        d = _circ_smooth(_circ_fill(d, 72), 2)
        src = "点云自身外轮廓（无检测种子）"
    rim3d = (c[None, :] + r[:, None] * np.cos(np.radians(mids))[:, None] * u[None, :]
             + r[:, None] * np.sin(np.radians(mids))[:, None] * v[None, :]
             + d[:, None] * n[None, :])
    return dict(center=c, normal=_unit(n), u=u, v=v, mids=mids, r=r, d=d,
                rim=rim3d, radius_mm=float(np.median(r)) * 1000.0, source=src, near=near)


# =============================================================== 生成路径点
def _ring_waypoints(fr, settings: PlanSettings):
    mids, r_out, d = fr["mids"], fr["r"], fr["d"]
    inset = float(settings.inset_mm) / 1000.0
    step = max(float(settings.step_mm) / 1000.0, 0.005)
    r_med = float(np.median(r_out))
    r_keep = max(r_med - inset, 0.2 * r_med, 0.008)
    perim = 2.0 * math.pi * r_keep
    n = int(np.clip(round(perim / step), 24, settings.max_points))
    th = np.linspace(-180.0, 180.0, n, endpoint=False)
    rr = np.clip(_profile_at(th, mids, r_out) - inset, 0.008, None)
    dd = _profile_at(th, mids, d)
    return th, rr, dd


def _spiral_waypoints(fr, settings: PlanSettings):
    mids, r_out, d = fr["mids"], fr["r"], fr["d"]
    inset = float(settings.inset_mm) / 1000.0
    step = max(float(settings.step_mm) / 1000.0, 0.005)
    r_med = float(np.median(r_out))
    r0 = max(r_med - inset, 0.3 * r_med, 0.010)
    r1 = max(0.18 * r_med, 0.008)
    turns = (r0 - r1) / step
    if turns < 0.4:
        return _ring_waypoints(fr, settings)
    total_ang = 360.0 * turns
    n = int(np.clip(round(total_ang / 8.0), 40, settings.max_points))
    th = np.linspace(-180.0, -180.0 + total_ang, n)
    rr = np.linspace(r0, r1, n)
    rr = np.minimum(rr, _profile_at(th, mids, r_out) - inset)
    rr = np.clip(rr, 0.006, None)
    return th, rr, _profile_at(th, mids, d)


def _raster_waypoints(fr, settings: PlanSettings, shrink_mm: float = 0.0):
    """面内栅格（S 形）：按轮廓多边形裁掉外面的部分，逐行交替方向。"""
    mids, r_out, d = fr["mids"], fr["r"], fr["d"]
    inset = float(settings.inset_mm) / 1000.0
    r_med = float(np.median(r_out))
    scale = max(0.08, 1.0 - (inset + float(shrink_mm) / 1000.0) / max(r_med, 0.001))
    poly = np.c_[r_out * scale * np.cos(np.radians(mids)),
                 r_out * scale * np.sin(np.radians(mids))]
    step = max(float(settings.step_mm) / 1000.0, 0.005)
    us = np.arange(poly[:, 0].min(), poly[:, 0].max() + 1e-9, step)
    out_uv: list[tuple[float, float]] = []
    flip = False
    for U in us:
        vs: list[float] = []
        for i in range(poly.shape[0]):
            p0, p1 = poly[i], poly[(i + 1) % poly.shape[0]]
            if (p0[0] - U) * (p1[0] - U) <= 0.0 and abs(p1[0] - p0[0]) > 1e-9:
                f = (U - p0[0]) / (p1[0] - p0[0])
                if 0.0 <= f <= 1.0:
                    vs.append(float(p0[1] + f * (p1[1] - p0[1])))
        if len(vs) < 2:
            continue
        v0, v1 = min(vs), max(vs)
        seg_n = max(int(round(abs(v1 - v0) / max(step * 0.75, 1e-9))), 1)
        vv = np.linspace(v0, v1, seg_n + 1)
        if flip:
            vv = vv[::-1]
        flip = not flip
        out_uv.extend([(float(U), float(x)) for x in vv])
    if not out_uv:
        return _ring_waypoints(fr, settings)
    uv = np.asarray(out_uv, dtype=float)
    th = np.degrees(np.arctan2(uv[:, 1], uv[:, 0]))
    rr = np.hypot(uv[:, 0], uv[:, 1])
    return th, rr, _profile_at(th, mids, d)


def waypoints_for_mode(fr, settings: PlanSettings):
    """按 ``mode`` 生成 ``(theta_deg, r_mm, d_mm)`` 三个等长数组。"""
    mode = str(settings.mode or "ring+raster").lower()
    if mode == "ring":
        return [_ring_waypoints(fr, settings)]
    if mode == "spiral":
        return [_spiral_waypoints(fr, settings)]
    if mode == "raster":
        return [_raster_waypoints(fr, settings)]
    # 默认 ring+raster：先走一圈，再在里面栅格扫（栅格缩进 2×inset，别压着环）
    return [_ring_waypoints(fr, settings),
            _raster_waypoints(fr, settings, shrink_mm=float(settings.inset_mm))]


def _waypoints_xyz(fr, parts, settings: PlanSettings):
    """极坐标路径 → 基座系 3D 理想点（还没吸附到点云）。"""
    c, n, u, v = fr["center"], fr["normal"], fr["u"], fr["v"]
    th = np.concatenate([p[0] for p in parts])
    rr = np.concatenate([p[1] for p in parts])
    dd = np.concatenate([p[2] for p in parts])
    if th.size > int(settings.max_points):
        idx = np.linspace(0, th.size - 1, int(settings.max_points)).round().astype(int)
        idx = np.unique(idx)
        th, rr, dd = th[idx], rr[idx], dd[idx]
    a = np.radians(th)
    return (c[None, :] + rr[:, None] * np.cos(a)[:, None] * u[None, :]
            + rr[:, None] * np.sin(a)[:, None] * v[None, :]
            + dd[:, None] * n[None, :])


# =============================================================== 吸附 / 法向 / IK
def _local_normal(grid: PointGrid, p, fallback, settings: PlanSettings):
    """目标点局部法向：邻域点 PCA；邻域太小或太"歪"就退回作业面法向。"""
    nb = grid.around(p, float(settings.normal_radius_mm) / 1000.0)
    if nb.shape[0] < 8:
        return fallback
    _, n, _ = pca_plane(nb)
    if float(n @ fallback) < 0:
        n = -n
    if float(np.degrees(np.arccos(np.clip(float(n @ fallback), -1.0, 1.0)))) > 35.0:
        return fallback
    return n


def _standoff_tcp(target, normal, settings: PlanSettings):
    """目标点 + 法向 × 喷距 → 工具尖；太远就沿法向往机器人一侧收。"""
    off = float(settings.standoff_mm) / 1000.0
    tcp = target + normal * off
    for _ in range(4):
        r = float(np.linalg.norm(tcp))
        if r <= float(settings.max_reach_mm) / 1000.0:
            break
        off += (r - float(settings.max_reach_mm) / 1000.0) + 0.01
        tcp = target + normal * off
    return tcp, off


def _ik_one(sim, pos, axis, seed, settings: PlanSettings):
    """单点 IK：先用上一支当种子，不中再换 home / 中性姿态（返回 ``q, ok, ep, ea``）。"""
    pos = np.asarray(pos, dtype=float).ravel()
    axis = None if axis is None else _unit(axis)
    pos_tol = float(settings.ik_pos_tol_mm) / 1000.0
    axis_tol = math.radians(float(settings.ik_axis_tol_deg))
    seeds = [np.asarray(seed, dtype=float).ravel(),
             np.asarray(spec.HOME_QPOS, dtype=float),
             np.asarray(spec.NEUTRAL_QPOS, dtype=float)]
    max_jump = math.radians(float(settings.ik_max_jump_deg))
    qpos0, qvel0 = sim.data.qpos.copy(), sim.data.qvel.copy()
    best = None
    try:
        for k, s in enumerate(seeds):
            q, ep, ea = core.solve_ik_pose(sim.model, sim.data, "tool_site", pos, axis,
                                           seed=s, iters=int(settings.ik_iters))
            q = np.asarray(q, dtype=float)
            score = float(ep) + 0.05 * float(ea)
            if best is None or score < best[0]:
                best = (score, q, float(ep), float(ea))
            near = k == 0 or float(np.abs(q - seeds[0]).max()) <= max_jump
            if near and float(ep) <= pos_tol and (axis is None or float(ea) <= axis_tol):
                return q, True, float(ep), float(ea)
    finally:
        sim.data.qpos[:] = qpos0
        sim.data.qvel[:] = qvel0
        if mujoco is not None:
            mujoco.mj_forward(sim.model, sim.data)
    if (best is not None and float(np.abs(best[1] - seeds[0]).max()) <= max_jump
            and best[2] <= pos_tol and (axis is None or best[3] <= axis_tol)):
        return best[1], True, best[2], best[3]
    return best[1], False, best[2], best[3]


def _snap_to_surface(target, uv_all, d_all, pts_all, u, v, c, snap_m: float):
    """沿作业面法向"垂直投影"到可见表面：在 ``snap_m`` 面内半径里挑**最靠机器人**的那层点。

    为什么不是三维最近点：马桶内圈/碗底的路径点在平面下方 100+ mm，三维最近点会吸到旁边
    的口沿上；而"同一 (u,v) 上、朝机器人方向最近的那层点"才是从机器人看过去的**可见表面**。
    """
    rel = np.asarray(target, dtype=float).ravel() - c
    puv = np.array([float(rel @ u), float(rel @ v)])
    du = uv_all - puv
    inplane = np.hypot(du[:, 0], du[:, 1])
    sel = np.flatnonzero(inplane <= float(snap_m))
    if sel.size == 0:
        return None
    d = d_all[sel]
    top = sel[d >= float(np.percentile(d, 90)) - 0.002]      # 最靠机器人的 10% 那一层
    if top.size == 0:
        top = sel
    k = int(top[np.argmin(inplane[top])])
    return pts_all[k]


def _seg_seconds(tcp, settings: PlanSettings) -> np.ndarray:
    """相邻工具尖距离 / 速度 → 单段时长（夹在 min/max 之间）。"""
    n = tcp.shape[0]
    out = np.full(n, 0.25)
    if n >= 2:
        d = np.linalg.norm(np.diff(tcp, axis=0), axis=1)
        ts = d / max(float(settings.speed_mm_s) / 1000.0, 1e-6)
        out[1:] = np.clip(ts, float(settings.min_seg_s), float(settings.max_seg_s))
    out[0] = float(settings.min_seg_s)
    return out


def _make_runs(q, ok, seconds, max_jump_deg: float = 90.0):
    """把 IK 结果切成"连续可执行"的关节路点段（失败点 / 大跳变处断开）。"""
    runs: list[np.ndarray] = []
    run_s: list[np.ndarray] = []
    cur: list[np.ndarray] = []
    cur_s: list[float] = []
    prev = None
    for i in range(int(q.shape[0])):
        if not bool(ok[i]):
            if len(cur) >= 2:
                runs.append(np.asarray(cur, dtype=float))
                run_s.append(np.asarray(cur_s, dtype=float))
            cur, cur_s, prev = [], [], None
            continue
        qi = np.asarray(q[i], dtype=float)
        if prev is not None and float(np.abs(np.degrees(qi - prev)).max()) > float(max_jump_deg):
            if len(cur) >= 2:
                runs.append(np.asarray(cur, dtype=float))
                run_s.append(np.asarray(cur_s, dtype=float))
            cur, cur_s = [], [max(float(seconds[i]), 0.6)]
        else:
            cur_s.append(float(seconds[i]))
        cur.append(qi)
        prev = qi
    if len(cur) >= 2:
        runs.append(np.asarray(cur, dtype=float))
        run_s.append(np.asarray(cur_s, dtype=float))
    return runs, run_s


# =============================================================== 现场取边（原样移植）
def _seed_colors(seeds) -> tuple[bool, bool]:
    """点云里是否同时有**红**（盆口边缘）和**青**（收缩后边缘）检测点。

    现场 `toilet_plan_from_cloud.py` 就是靠这两条边做自举/取边；没有它们（sink / 小便池的
    扁平点云等）就只能走通用的"点云外轮廓"规划。
    """
    red = cyan = False
    for g in (seeds or []):
        rgb = tuple(g.get("rgb") or ())
        if len(rgb) >= 3:
            r, gg, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
            red = red or (r > 150.0 and gg < 100.0 and b < 100.0)
            cyan = cyan or (r < 100.0 and gg > 150.0 and b > 150.0)
    return red, cyan


def _effective_photo_pose(pose, *, ref="tcp", tool_offset_mm=42.7,
                          offset_mm=(0.0, 0.0, 0.0), yaw_deg: float = 0.0):
    """GUI 的「参考点 / 微调 offset / 绕 z 旋转」折成一个等效拍照位姿（现场算法直接吃这个）。

    现场是 ``p_base = R_photo·p_o2e + t_photo``；本工程在它后面还乘了：
    法兰参考点 ``t -= R[:,2]·L``、绕基座 z 转 ``R = Rz(yaw)R``、平移 ``t += offset``。
    因为 R_photo 本身就是 ``Rz·Ry·Rx``，左乘 Rz(yaw) 只等价于 **rz += yaw**。
    """
    pose6 = pc.parse_pose6(pose)
    R, p = pc.pose_matrix(pose6)
    if str(ref).lower() in ("flange", "ee", "ee_site", "link6"):
        p = p - R[:, 2] * (float(tool_offset_mm) / 1000.0)
    yaw = math.radians(float(yaw_deg))
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    p = rz @ p + np.asarray(offset_mm, dtype=float).ravel() / 1000.0
    return [float(p[0] * 1000.0), float(p[1] * 1000.0), float(p[2] * 1000.0),
            float(pose6[3]), float(pose6[4]), float(pose6[5]) + float(yaw_deg)]


def _site_pose(sim, name: str = "ee_site"):
    """读一个 site 的位姿 ``(pos[m], R 3×3)``（先 FK 一次）。"""
    mujoco.mj_forward(sim.model, sim.data)
    sid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise pc.PointCloudError(f"模型里没有 site「{name}」")
    return (np.asarray(sim.data.site_xpos[sid], dtype=float).copy(),
            np.asarray(sim.data.site_xmat[sid], dtype=float).reshape(3, 3).copy())


def _rot_err_deg(Ra, Rb) -> float:
    """两个旋转矩阵之间的夹角[deg]。"""
    c = (float(np.trace(np.asarray(Ra).T @ np.asarray(Rb))) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))


def _ik_flange(sim, pos, R_target, seeds, settings: PlanSettings):
    """法兰**全位姿** IK：位置 + 工具轴（EE z）+ 绕工具轴的滚转（关节 6）。

    现场 `compute_poses` 给的是法兰位姿（固定到达姿态 ``R_arr``），喷嘴相对法兰还有
    41/45/234 mm 的侧偏；只解"位置 + 工具轴"会留下滚转自由度，喷嘴尖会甩到别处。
    所以先按位置 + z 轴解 IK，再算 ``R_cur^T·R_target`` 的绕 z 角、加到 J6 上；
    最后用 FK 校验位置/姿态误差（默认 2 mm / 3°）。

    返回 ``(q, ok, pos_err_mm, rot_err_deg, roll_matched)``。
    """
    pos = np.asarray(pos, dtype=float).ravel()
    R_target = np.asarray(R_target, dtype=float).reshape(3, 3)
    pos_tol = float(settings.ik_pos_tol_mm)
    axis_tol = math.radians(float(settings.ik_axis_tol_deg))
    qpos0, qvel0 = sim.data.qpos.copy(), sim.data.qvel.copy()
    best = None
    try:
        for seed in seeds:
            q, ep, ea = core.solve_ik_pose(sim.model, sim.data, "ee_site", pos,
                                           R_target[:, 2], seed=np.asarray(seed, dtype=float),
                                           iters=int(settings.ik_iters))
            q = np.asarray(q, dtype=float)
            if ep * 1000.0 > 4.0 or ea > math.radians(8.0):
                if best is None or ep + 0.05 * ea < best[0]:
                    best = (ep + 0.05 * ea, q, ep * 1000.0, math.degrees(ea), False)
                continue
            sim.data.qpos[:] = q
            _, R_cur = _site_pose(sim, "ee_site")
            m = R_cur.T @ R_target
            roll0 = math.atan2(float(m[1, 0]), float(m[0, 0]))
            for d in (roll0, roll0 - 2.0 * math.pi, roll0 + 2.0 * math.pi):
                q2 = q.copy()
                q2[5] += d
                if q2[5] < sim.lo[5] or q2[5] > sim.hi[5]:
                    continue
                sim.data.qpos[:] = q2
                p2, R2 = _site_pose(sim, "ee_site")
                pe = float(np.linalg.norm(p2 - pos)) * 1000.0
                re = _rot_err_deg(R2, R_target)
                score = pe + 0.2 * re
                if best is None or score < best[0]:
                    best = (score, q2, pe, re, True)
                if pe <= pos_tol and re <= 3.0:
                    return q2, True, pe, re, True
    finally:
        sim.data.qpos[:] = qpos0
        sim.data.qvel[:] = qvel0
        mujoco.mj_forward(sim.model, sim.data)
    if best is None:
        return np.asarray(seeds[0], dtype=float), False, float("inf"), float("inf"), False
    _, q, pe, re, rolled = best
    ok = pe <= max(pos_tol, 2.5) and re <= 5.0
    return q, ok, pe, re, rolled

def _field_argv(cloud_path, settings: PlanSettings, eff_pose) -> list[str]:
    """把 PlanSettings 翻成现场脚本的命令行参数（默认值全部沿用现场脚本）。"""
    st = settings
    argv = [str(cloud_path), "--no-save", "--no-fail",
            "--photo-pose", *[f"{v:.6g}" for v in eff_pose],
            "--arrival-angles", *[f"{v:.6g}" for v in st.arrival_angles],
            "--drop-hl", str(st.drop_hl),
            "--footprint-frac", f"{float(st.footprint_frac):.6g}",
            "--ring-offset", ("auto" if float(st.ring_offset_mm) <= 0
                              else f"{float(st.ring_offset_mm):.6g}")]
    if float(st.standoff_mm) > 0:
        argv += ["--standoff", f"{float(st.standoff_mm):.6g}"]
    else:
        argv += ["--standoff", "auto"]
    if float(st.ring_standoff_mm) >= 0:
        argv += ["--ring-standoff", f"{float(st.ring_standoff_mm):.6g}"]
    if st.lip:
        argv += ["--lip"]
    return argv


def build_field_plan(cloud_path, settings: PlanSettings, *, sim=None,
                     pose=pc.DEFAULT_POSE, ref: str = "tcp",
                     tool_offset_mm: float = 42.7, offset_mm=(0.0, 0.0, 0.0),
                     yaw_deg: float = 0.0) -> PathPlan:
    """**现场取边**：原样调用 ``toilet_plan.py``（toilet 实例的移植版）+ MuJoCo IK/回放。

    与通用轮廓的区别（也是这次改的重点）：
      * 盆口不是"投影外轮廓"，而是**从点云脊线取出来的实测边**（红=盆口、青=收缩边）；
      * 路径是现场的 **内缩圈（沿内法向偏移 40mm）+ 栅格（短轴蛇形、上→下、贴边 10mm）**，
        间距由喷锥足迹 ``D=2·standoff·tan(15°)、s=D·(1-20%)`` 决定；
      * 法兰位姿用现场的 **固定到达姿态** ``ARRIVAL_ANGLES``：``P_flange = P_target - so·ax - R·TCP``，
        喷嘴尖 = ``P_flange + R·TCP``（不再是"沿局部法向估一个姿态"）；
      * 输出覆盖率 / 工具轴线到点云的间隙 / 逐遍摘要，和现场脚本同口径。
    """
    try:
        import toilet_plan as fld
        import spray_planner as sp
    except Exception as exc:                              # noqa: BLE001
        raise pc.PointCloudError(f"现场取边模块不可用（toilet_plan.py / spray_plan/）：{exc}") from exc
    if str(settings.mode) == "spiral":
        raise pc.PointCloudError("现场取边算法没有「螺旋」走法：它只生成 内缩圈 + 栅格（可选唇遍）")
    eff = _effective_photo_pose(pose, ref=ref, tool_offset_mm=tool_offset_mm,
                                offset_mm=offset_mm, yaw_deg=yaw_deg)
    argv = _field_argv(cloud_path, settings, eff)
    args = fld.build_arg_parser().parse_args(argv)
    logs: list[str] = []
    t0 = time.perf_counter()
    old_log = fld.log
    fld.log = lambda msg="": logs.append(str(msg))        # 把现场脚本的中文日志收进报告
    try:
        res = fld.run_plan(str(cloud_path), args)
    except SystemExit as exc:
        raise pc.PointCloudError(f"现场取边规划中止：{exc}") from exc
    finally:
        fld.log = old_log
    field_ms = (time.perf_counter() - t0) * 1000.0
    return _field_result_to_plan(res, settings, sim=sim, sp=sp, fld=fld,
                                 logs=logs, pose_eff=eff, field_ms=field_ms)


def _field_result_to_plan(res: dict, st: PlanSettings, *, sim, sp, fld,
                          logs, pose_eff, field_ms: float = 0.0) -> PathPlan:
    """现场 ``run_plan`` 的结果 → 本工程的 ``PathPlan``（外加 MuJoCo 全位姿 IK）。"""
    t0 = time.perf_counter()
    pf = res["pf"]
    asm = res["asm"]
    poses = res["poses"]
    cov = res["cov"]
    clr = res["clr"]
    keys = list(asm["pass_key"])
    n_all = int(len(asm["wp_pos"]))
    # ---- 按界面"走法"筛遍（现场默认 = 内缩圈 + 栅格；唇遍要显式勾）-----------
    mode = str(st.mode or "ring+raster")
    keep = {"ring", "raster"} if mode == "ring+raster" else {mode}
    if st.lip:
        keep |= {"rim_left", "rim_right"}
    idx = [i + 1 for i, k in enumerate(keys) if k in keep]
    if idx and len(idx) < len(keys):
        mask = np.isin(np.asarray(asm["wp_pass"], dtype=float), idx)
        asm2 = dict(asm)
        for key in ("wp_pos", "wp_type", "wp_line", "wp_dwell", "wp_pass", "so_wp"):
            asm2[key] = np.asarray(asm[key])[mask]
        remap = {old: k + 1 for k, old in enumerate(idx)}
        asm2["wp_pass"] = np.asarray([remap[int(v)] for v in asm2["wp_pass"]], dtype=float)
        asm2["pass_key"] = [keys[i - 1] for i in idx]
        asm2["pass_short"] = [asm["pass_short"][i - 1] for i in idx]
        asm2["pass_txt"] = [asm["pass_txt"][i - 1] for i in idx]
        asm2["n_wp"] = int(len(asm2["wp_pos"]))
        asm = asm2
        poses = sp.compute_poses(asm["wp_pos"], asm["so_wp"], pf.n, st.arrival_angles,
                                 "nozzle1", "tool1")
        cov = sp.compute_coverage(poses, asm, pf.uv, pf.c0, pf.u, pf.v, pf.n,
                                  fld.SPRAY_CONE_DEG, fld.COVER_CELL_MM)
        clr = sp.compute_clearance(poses, fld.voxel_downsample(res["cloud_base"], 2.0),
                                   fld.CLEAR_WARN_MM)
    n = len(asm["wp_pos"])
    if n == 0:
        raise pc.PointCloudError("现场取边规划没有产生航点（走法筛空了）")

    # ---- 法兰全位姿 IK ------------------------------------------------------
    R_arr = np.asarray(poses["r_arr"], dtype=float)
    q = np.full((n, len(spec.JOINTS)), np.nan)
    ok = np.zeros(n, dtype=bool)
    ik_ms = 0.0
    roll_bad = 0
    if sim is not None:
        t1 = time.perf_counter()
        prev = np.asarray(sim.q_pos(), dtype=float)
        seeds0 = [np.asarray(spec.HOME_QPOS, dtype=float),
                  np.asarray(spec.NEUTRAL_QPOS, dtype=float)]
        for i in range(n):
            pos = np.asarray(poses["p_flange"][i], dtype=float) / 1000.0
            if st.full_pose_ik:
                qi, good, _pe, _re, rolled = _ik_flange(sim, pos, R_arr,
                                                        [prev] + seeds0, st)
                if good and not rolled:
                    roll_bad += 1
            else:
                qpos0, qvel0 = sim.data.qpos.copy(), sim.data.qvel.copy()
                try:
                    qi, ep, ea = core.solve_ik_pose(sim.model, sim.data, "ee_site", pos,
                                                    R_arr[:, 2], seed=prev,
                                                    iters=int(st.ik_iters))
                    good = (ep * 1000.0 <= float(st.ik_pos_tol_mm) * 1.5
                            and ea <= math.radians(float(st.ik_axis_tol_deg) * 1.5))
                finally:
                    sim.data.qpos[:] = qpos0
                    sim.data.qvel[:] = qvel0
                    mujoco.mj_forward(sim.model, sim.data)
            q[i] = np.asarray(qi, dtype=float)
            ok[i] = bool(good)
            if good:
                prev = np.asarray(qi, dtype=float)
        ik_ms = (time.perf_counter() - t1) * 1000.0

    # ---- 组装 PathPlan ------------------------------------------------------
    targets = np.asarray(asm["wp_pos"], dtype=float) / 1000.0
    tcp = np.asarray(poses["noz_b"], dtype=float) / 1000.0
    ax = np.asarray(poses["ax_spray"], dtype=float).ravel()
    normals = np.tile(-ax, (n, 1))
    seconds = _seg_seconds(tcp, st)
    runs, run_s = _make_runs(q, ok, seconds) if sim is not None else ([], [])
    uv = np.asarray(pf.uv, dtype=float)
    radius_mm = float(np.median(np.hypot(uv[:, 0], uv[:, 1])))
    plan = PathPlan(
        targets=targets, normals=normals, tcp=tcp, q=q, ok=ok, seconds=seconds,
        on_cloud=np.ones(n, dtype=bool), runs=runs, run_seconds=run_s,
        rim=np.asarray(res["rim_pts"], dtype=float) / 1000.0,
        center=np.asarray(pf.c0, dtype=float) / 1000.0,
        normal=np.asarray(pf.n, dtype=float), u=np.asarray(pf.u, dtype=float),
        v=np.asarray(pf.v, dtype=float), radius_mm=radius_mm,
        seed_source="现场取边：红=盆口边缘 / 青=收缩后边缘（点云脊线）",
        stats=dict(engine="field", planned=int(n), skipped_reach=0,
                   on_cloud=int(n), ik_ms=ik_ms, mode=mode, step_mm=float(res["s_fp"]),
                   standoff_mm=float(res["so_mm"]), seed="field",
                   so_mm=float(res["so_mm"]), D_fp=float(res["D_fp"]),
                   s_fp=float(res["s_fp"]), coverage=float(cov["pc_all"]),
                   uncover_cm2=float(cov["uncover_cm2"]),
                   min_nozzle_mm=float(clr["min_nozzle"]),
                   min_gripper_mm=float(clr["min_gripper"]),
                   roll_bad=int(roll_bad),
                   passes=list(asm["pass_short"]),
                   n_ring=int((np.asarray(asm["wp_type"]) == 1).sum()),
                   n_raster=int((np.asarray(asm["wp_type"]) == 2).sum())),
        ms=(time.perf_counter() - t0) * 1000.0 + float(field_ms))
    plan.report = _field_report(plan, dict(res, asm=asm, poses=poses, cov=cov, clr=clr,
                                           n_all=n_all),
                                st, pose_eff, logs, roll_bad)
    return plan

def _field_report(plan: PathPlan, res: dict, st: PlanSettings, pose_eff,
                  logs, roll_bad: int) -> list[str]:
    pf, cov, clr, ps = res["pf"], res["cov"], res["clr"], res["poses"]
    lines = [
        "算法    : 现场取边（toilet_plan.py 原样移植：点云脊线取边 → 内缩圈/栅格 → 法兰位姿）",
        f"拍照位姿: [{pose_eff[0]:.2f}, {pose_eff[1]:.2f}, {pose_eff[2]:.2f}] mm / "
        f"[{pose_eff[3]:.2f}, {pose_eff[4]:.2f}, {pose_eff[5]:.2f}]°（含参考点/微调）",
        f"盆口    : 质心 ({pf.c0[0]:.1f}, {pf.c0[1]:.1f}, {pf.c0[2]:.1f}) mm ｜ "
        f"法向 ({pf.n[0]:.3f}, {pf.n[1]:.3f}, {pf.n[2]:.3f}) ｜ "
        f"面内 {np.ptp(pf.uv[:, 0]):.0f}×{np.ptp(pf.uv[:, 1]):.0f} mm ｜ "
        f"面积 {float(res['a_open']) / 100:.0f} cm²",
        f"喷距    : standoff {res['so_mm']:.1f} mm（auto：按足迹 ≤ "
        f"{st.footprint_frac * 100:.0f}%×口短边收紧）｜ 足迹 D {res['D_fp']:.1f} mm ｜ "
        f"间距 s {res['s_fp']:.1f} mm",
        "遍      : " + " ｜ ".join(res["asm"]["pass_txt"]),
        f"位姿    : 喷嘴口到目标 {ps['d_pt'].min():.1f}~{ps['d_pt'].max():.1f} mm ｜ "
        f"喷淋轴与盆口法向夹角 {float(np.median(ps['inc_deg'])):.1f}°",
        f"间隙    : 工具轴线到点云 喷嘴 {clr['min_nozzle']:.1f} / 夹爪 "
        f"{clr['min_gripper']:.1f} mm",
        f"覆盖率  : {cov['pc_all']:.1f}%（未覆盖 {cov['uncover_cm2']:.1f} cm²；"
        f"仅沿口 {cov['pc_rim']:.1f}% / 仅栅格 {cov['pc_raster']:.1f}%）",
    ]
    if int(res.get("n_all", plan.n)) != plan.n:
        lines.append(f"走法筛选: 现场原始 {int(res['n_all'])} 点 → 当前 {plan.n} 点"
                     f"（{st.mode}）")
    if plan.n_ok:
        full = 100.0 * (plan.n_ok - roll_bad) / max(plan.n_ok, 1)
        lines.append(f"IK      : {plan.n_ok}/{plan.n} 点可解（全位姿 {full:.0f}%），"
                     f"{len(plan.runs)} 段连续轨迹，规划 {plan.ms:.0f} ms"
                     f"（IK {plan.stats['ik_ms']:.0f} ms）")
        if roll_bad:
            lines.append(f"          {roll_bad} 个点关节 6 行程不够、滚转没完全对上"
                         f"（位置/工具轴仍然正确）")
    elif plan.n:
        lines.append("IK      : 未执行（无仿真模型）")
    for ln in logs:
        s = ln.strip()
        if any(k in s for k in ("点云取边-", "贴合校验-", "栅格尾巴", "喷洗距离", "合计航点")):
            lines.append("现场    : " + s)
    lines.append("说明    : 现场航点带 0.8 s 停留；本仿真按连续关节路点回放（不逐点停顿）")
    return lines



def build_plan(points_base, settings: PlanSettings | None = None, *, seeds=None,
               camera=None, sim=None, robot=(0.0, 0.0, 0.0),
               cloud_path=None, pose=pc.DEFAULT_POSE, ref: str = "tcp",
               tool_offset_mm: float | None = None, offset_mm=(0.0, 0.0, 0.0),
               yaw_deg: float = 0.0) -> PathPlan:
    """点云（基座系 m）→ 作业路径 + 关节路点（给了 ``sim`` 才做 IK）。

    默认 ``engine="auto"``：点云里同时有**红/青检测边缘**时走**现场取边**
    （``toilet_plan.py`` 原样移植，见 :func:`build_field_plan`）；否则走**通用轮廓**。
    ``cloud_path`` 是现场取边要重新读的原始点云文件；``pose/ref/offset/yaw`` 用来把它和
    场景里的点云摆成同一位置。
    """
    t0 = time.perf_counter()
    st = settings or PlanSettings()
    if tool_offset_mm is None:
        try:
            tool_offset_mm = float(spec.TOOL_TIP_LOCAL_Z) * 1000.0
        except Exception:                                  # noqa: BLE001
            tool_offset_mm = 42.7

    # ---- 引擎选择：现场取边（点云脊线）优先 --------------------------------
    red, cyan = _seed_colors(seeds)
    field_err = ""
    eng = str(st.engine or "auto").lower()
    if eng in ("auto", "field") and cloud_path and red and cyan:
        try:
            return build_field_plan(cloud_path, st, sim=sim, pose=pose, ref=ref,
                                    tool_offset_mm=float(tool_offset_mm),
                                    offset_mm=offset_mm, yaw_deg=yaw_deg)
        except Exception as exc:                          # noqa: BLE001
            if eng == "field":
                raise
            field_err = f"{type(exc).__name__}: {exc}"
    elif eng == "field":
        raise pc.PointCloudError(
            "现场取边需要点云里同时有红色（盆口）和青色（收缩）检测边缘；"
            "这份点云没有 → 请选「通用轮廓」，或换 toilet-1.json")

    # ---- 通用轮廓：standoff / step 的 0 = 自动 -----------------------------
    if float(st.standoff_mm) <= 0 or float(st.step_mm) <= 0:
        st = replace(st,
                     standoff_mm=(float(st.standoff_mm) if float(st.standoff_mm) > 0 else 400.0),
                     step_mm=(float(st.step_mm) if float(st.step_mm) > 0 else 40.0))
    pts = np.asarray(points_base, dtype=float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 20:
        raise pc.PointCloudError("点云点太少（< 20），没法规划路径")
    ds = voxel_downsample(pts, 0.004)                   # 规划用 4 mm 体素
    cell = max(float(st.snap_mm), float(st.normal_radius_mm), 10.0) / 1000.0
    grid = PointGrid(voxel_downsample(pts, 0.002), cell)
    fr = _build_frame(ds, seeds=seeds, settings=st, camera=camera, robot=robot)
    ideal = _waypoints_xyz(fr, waypoints_for_mode(fr, st), st)

    # ---- 吸附到真实点云 + 估法向 + 算工具尖 --------------------------------
    plane_n = fr["normal"]
    rel_ds = ds - fr["center"]
    uv_all = np.c_[rel_ds @ fr["u"], rel_ds @ fr["v"]]
    d_all = rel_ds @ plane_n
    snap_m = float(st.snap_mm) / 1000.0
    targets: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    tcps: list[np.ndarray] = []
    on_cloud: list[bool] = []
    skipped_reach = 0
    prev_target = None
    step_m = max(float(st.step_mm) / 1000.0, 0.005)
    for p in ideal:
        snap = _snap_to_surface(p, uv_all, d_all, ds, fr["u"], fr["v"],
                                fr["center"], snap_m)
        if snap is not None and prev_target is not None:
            # 连续性保护：同一条栅格/环上相邻点的面内距离只有 step 量级，
            # 若吸附后 3D 位置跳出去（跳到另一层表面/碗壁背面），就退回作业面上的理想点，
            # 否则轨迹会出现"来回穿面"的长直线。
            prev_uv = np.array([(prev_target - fr["center"]) @ fr["u"],
                                (prev_target - fr["center"]) @ fr["v"]])
            cur_uv = np.array([(p - fr["center"]) @ fr["u"], (p - fr["center"]) @ fr["v"]])
            if (float(np.linalg.norm(cur_uv - prev_uv)) <= 2.0 * step_m + snap_m
                    and float(np.linalg.norm(snap - prev_target)) > 4.0 * step_m + 0.02):
                snap = None
        tgt = np.asarray(snap if snap is not None else p, dtype=float)
        n = _local_normal(grid, tgt, plane_n, st)
        tcp, _off = _standoff_tcp(tgt, n, st)
        r = float(np.linalg.norm(tcp))
        if r < float(st.min_reach_mm) / 1000.0 or r > float(st.max_reach_mm) / 1000.0:
            skipped_reach += 1
            continue
        targets.append(tgt)
        normals.append(n)
        tcps.append(tcp)
        on_cloud.append(snap is not None)
        prev_target = tgt
    if not targets:
        raise pc.PointCloudError("规划出的点都在机械臂工作空间外（调大喷距 / 缩小轮廓）")

    # ---- 接近点：从第一个目标沿法向退 ``approach`` 再进刀 -------------------
    t0t, n0 = targets[0], normals[0]
    off0 = float(np.linalg.norm(tcps[0] - t0t))
    approach_tcp = t0t + n0 * (off0 + float(st.approach_mm) / 1000.0)
    if float(np.linalg.norm(approach_tcp)) <= float(st.max_reach_mm) / 1000.0:
        targets.insert(0, t0t + n0 * float(st.approach_mm) / 1000.0)
        normals.insert(0, n0)
        tcps.insert(0, approach_tcp)
        on_cloud.insert(0, False)

    targets_a = np.asarray(targets, dtype=float)
    normals_a = np.asarray(normals, dtype=float)
    tcp_a = np.asarray(tcps, dtype=float)
    on_cloud_a = np.asarray(on_cloud, dtype=bool)
    seconds = _seg_seconds(tcp_a, st)
    q = np.full((tcp_a.shape[0], len(spec.JOINTS)), np.nan)
    ok = np.zeros(tcp_a.shape[0], dtype=bool)
    ik_ms = 0.0
    if sim is not None:
        t1 = time.perf_counter()
        prev = np.asarray(sim.q_pos(), dtype=float)
        for i in range(tcp_a.shape[0]):
            axis = -normals_a[i]
            qi, good, _ep, _ea = _ik_one(sim, tcp_a[i], axis, prev, st)
            if not good:                                # 兜底：工具沿"基座 → 目标"的径向下压
                radial = _unit(targets_a[i] - np.asarray(robot, dtype=float))
                if float(radial @ axis) < 0.94:
                    qi2, good2, _p2, _a2 = _ik_one(sim, tcp_a[i], radial, prev, st)
                    if good2:
                        qi, good = qi2, True
            q[i] = qi
            ok[i] = bool(good)
            if good:
                prev = qi
        ik_ms = (time.perf_counter() - t1) * 1000.0
    runs, run_s = _make_runs(q, ok, seconds) if sim is not None else ([], [])

    plan = PathPlan(targets=targets_a, normals=normals_a, tcp=tcp_a, q=q, ok=ok,
                    seconds=seconds, on_cloud=on_cloud_a, runs=runs, run_seconds=run_s,
                    rim=fr["rim"], center=fr["center"], normal=fr["normal"],
                    u=fr["u"], v=fr["v"], radius_mm=fr["radius_mm"],
                    seed_source=fr["source"],
                    stats=dict(planned=int(ideal.shape[0]), skipped_reach=skipped_reach,
                               on_cloud=int(on_cloud_a.sum()), ik_ms=ik_ms,
                               engine="outline", mode=st.mode, step_mm=st.step_mm,
                               standoff_mm=st.standoff_mm, seed=fr["source"]),
                    ms=(time.perf_counter() - t0) * 1000.0)
    plan.report = _report(plan, st)
    if field_err:
        plan.stats["field_error"] = field_err
        plan.report.insert(0, f"算法    : 通用轮廓（现场取边不可用：{field_err}）")
    else:
        plan.report.insert(0, "算法    : 通用轮廓（点云外轮廓 + 吸附；本份点云没有红/青检测边缘）")
    return plan


def _report(plan: PathPlan, st: PlanSettings) -> list[str]:
    stt = plan.stats
    lines = [f"轮廓    : {plan.seed_source}",
             f"作业面  : 中心 ({plan.center[0]:.3f}, {plan.center[1]:.3f}, "
             f"{plan.center[2]:.3f}) m ｜ 法向 ({plan.normal[0]:.2f}, "
             f"{plan.normal[1]:.2f}, {plan.normal[2]:.2f}) ｜ 半径 ~{plan.radius_mm:.0f} mm",
             f"路径    : {st.mode} × {plan.n} 点（吸附点云 {stt['on_cloud']}，"
             f"工作空间外跳过 {stt['skipped_reach']}）",
             f"喷距    : {st.standoff_mm:.0f} mm ｜ 步距 {st.step_mm:.0f} mm ｜ "
             f"接近 {st.approach_mm:.0f} mm"]
    if plan.n_ok:
        lines.append(f"IK      : {plan.n_ok}/{plan.n} 点可解，{len(plan.runs)} 段连续轨迹，"
                     f"规划 {plan.ms:.0f} ms（IK {stt['ik_ms']:.0f} ms）")
    elif plan.n:
        lines.append("IK      : 未执行（无仿真模型）")
    return lines

def make_run_motion(sim, plan: PathPlan, index: int):
    """第 ``index`` 段关节路点 → ``arm_core.Motion``（整体 smoothstep，首尾不激振）。"""
    if not plan.runs or index < 0 or index >= len(plan.runs):
        return None
    qs = np.asarray(plan.runs[index], dtype=float)
    secs = plan.run_seconds[index]
    total = max(float(np.sum(secs)), 0.2)
    return core.Motion(np.asarray(sim.cmd, dtype=float), qs[-1], total, sim.dt,
                       kind="plan", label=f"路径 {index + 1}/{len(plan.runs)}",
                       waypoints=qs)


def tool_trail_name(sim) -> str:
    """仿真时记哪根工具尖的轨迹：优先喷嘴1（真机喷洗用的那根），没有就用第一个工具。"""
    names = spec.tool_names()
    for want in ("喷嘴1", "刷子"):
        if want in names:
            return want
    return names[0] if names else ""


def _draw_path(scene, plan: PathPlan, *, every: int | None = None) -> None:
    """在 MuJoCo 的 ``MjvScene`` 上画路径 / 轮廓 / 目标点（只显示，不参与物理）。"""
    if mujoco is None or scene is None:
        return

    def segs(pts, color, radius):
        if pts is None or len(pts) < 2:
            return
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            ln = float(np.linalg.norm(d))
            if ln < 1e-6:
                continue
            core.add_marker(scene, mujoco.mjtGeom.mjGEOM_CAPSULE,
                            (radius, ln / 2 + radius, 0.0), (a + b) / 2.0,
                            core.frame_from_z(d).reshape(-1), color)

    pts = np.asarray(plan.tcp, dtype=float)
    k = int(every or max(len(pts) // 90, 1))
    segs(pts[::k], (0.20, 0.95, 1.00, 0.55), 0.0018)
    segs(np.asarray(plan.rim, dtype=float), (1.00, 0.55, 0.12, 0.85), 0.0016)
    for p in np.asarray(plan.targets, dtype=float)[::max(len(plan.targets) // 24, 1)]:
        core.add_marker(scene, mujoco.mjtGeom.mjGEOM_SPHERE, (0.006, 0.0, 0.0), p,
                        np.eye(3).reshape(-1), (0.25, 0.95, 0.45, 0.9))


def render_preview(scene_path, out_png, plan: PathPlan, *, width: int = 1280,
                   height: int = 800, sim=None) -> Path:
    """离屏渲染一张"点云 + 路径 + 机械臂"图（没有 GL 时会抛异常，调用方自己接）。"""
    import imageio.v2 as imageio

    sim = sim or core.ArmSim(scene_path, log_scene=False)
    sim.path_points = np.asarray(plan.tcp, dtype=float)
    r = mujoco.Renderer(sim.model, int(height), int(width))
    mid = 0.55 * np.asarray(plan.center, dtype=float)          # 机械臂 + 作业面之间
    cam = core.Camera(distance=max(1.6, 2.6 * float(plan.radius_mm) / 1000.0 + 0.8),
                      azimuth=145.0, elevation=-14.0, lookat=mid)
    try:
        r.update_scene(sim.data, camera=cam.mjv)
        _draw_path(r.scene, plan)
        sim.add_tool_arrows(r.scene)
        img = r.render()
    finally:
        r.close()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(out), img)
    return out


def simulate(plan: PathPlan, sim, *, renderer=None, video=None, fps: float = 30.0,
             max_seconds: float = 120.0) -> dict:
    """按关节路点分段回放，返回 ``dict(steps, seconds, tip_path, done)``。"""
    trail = tool_trail_name(sim)
    tr = sim.trail(trail) if trail else None
    if tr is not None:
        tr.clear()
    tip: list[np.ndarray] = []
    t0 = time.perf_counter()
    steps = 0
    writer = None
    if video is not None:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(video), fps=float(fps))
    try:
        def play(motion) -> None:
            nonlocal steps
            sim.motion = motion
            sim.mode = "plan"
            while sim.moving() and (time.perf_counter() - t0) < max_seconds:
                sim.step()
                steps += 1
                tip.append(sim.tip().copy())
                if tr is not None:
                    tr.add(sim.tip())
                if writer is not None and renderer is not None and steps % max(int(30 / fps), 1) == 0:
                    renderer.update_scene(sim.data)
                    writer.append_data(renderer.render())

        for i in range(len(plan.runs)):
            qs = np.asarray(plan.runs[i], dtype=float)
            dq = float(np.abs(np.degrees(qs[0] - np.asarray(sim.cmd, dtype=float))).max())
            if dq > 8.0:                                   # 段间换解支 → 先平滑转场
                play(core.Motion(np.asarray(sim.cmd, dtype=float), qs[0],
                                 min(max(dq / 120.0, 0.5), 2.0), sim.dt,
                                 kind="plan", label="转场"))
            play(make_run_motion(sim, plan, i))
            if (time.perf_counter() - t0) >= max_seconds:
                break
        if writer is not None:
            writer.close()
            writer = None
    finally:
        if writer is not None:
            writer.close()
    return dict(steps=steps, seconds=time.perf_counter() - t0,
                tip_path=np.asarray(tip, dtype=float).reshape(-1, 3),
                done=bool(len(plan.runs)))


def default_cloud() -> Path:
    """自检 / 命令行不给 ``--cloud`` 时用的点云：优先 toilet-1，再退回其它。"""
    for name in ("toilet-1.json", "urinal_o2e_stride5.json", "sink_1_o2e.json"):
        p = pc.CLOUD_DIR / name
        if p.is_file():
            return p
    files = pc.list_clouds()
    if not files:
        raise pc.PointCloudError("point_cloud/ 里没有点云文件")
    return files[0]


# =============================================================== 命令行
def _print(lines, log=print) -> None:
    for line in lines:
        log(line)


def run_selftest(args) -> int:
    """无窗口自检：读 toilet-1（多格式）→ 规划 → IK → 回放一小段。"""
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"：{detail}" if detail else ""))
        ok = ok and cond

    path = Path(args.cloud) if args.cloud else default_cloud()
    t0 = time.perf_counter()
    inputs = load_inputs(path, pose=args.pose, frame=args.frame)
    check("读点云 + 换算到基座系", inputs["points"].shape[0] > 1e3,
          f"{path.name}：{inputs['points'].shape[0]} 点，{len(inputs['seeds'])} 组种子，"
          f"{(time.perf_counter() - t0):.2f} s")
    sim = core.ArmSim(spec.SCENE_XML, log_scene=False)
    st = PlanSettings(mode=args.mode, standoff_mm=args.standoff, step_mm=args.step,
                      inset_mm=args.inset, snap_mm=args.snap, speed_mm_s=args.speed,
                      max_points=args.max_points, approach_mm=args.approach,
                      engine=getattr(args, "engine", "auto"), lip=bool(getattr(args, "lip", False)),
                      ring_offset_mm=float(getattr(args, "ring_offset", 40.0)),
                      ring_standoff_mm=float(getattr(args, "ring_standoff", -1.0)),
                      footprint_frac=float(getattr(args, "footprint_frac", 0.35)),
                      ik_iters=min(int(args.ik_iters), 160))
    plan = build_plan(inputs["points"], st, seeds=inputs["seeds"],
                      camera=inputs["camera"], sim=sim,
                      cloud_path=str(path), pose=args.pose)
    check("生成路径", plan.n > 20, plan.summary())
    check("IK 可解率 ≥ 80%", plan.n_ok / max(plan.n, 1) >= 0.8,
          f"{plan.n_ok}/{plan.n}，{len(plan.runs)} 段")
    if plan.runs:
        m = make_run_motion(sim, plan, 0)
        check("第一段运动可创建", m is not None and m.n > 2, f"{m.n} 步")
        sim.motion = m
        sim.mode = "plan"
        for _ in range(min(m.n + 5, 4000)):
            sim.step()
        check("仿真推进不报错", float(sim.sim_time) > 0.0, f"sim_time {sim.sim_time:.2f}s")
    _print(plan.report)
    print(f"\n自检{'通过' if ok else '失败'}：{plan.summary()}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="点云 → 机械臂路径规划 + MuJoCo 仿真回放",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--cloud", default="", help="点云文件（默认 point_cloud/ 下优先 toilet-1）")
    ap.add_argument("--pose", default=pc.pose_text(pc.DEFAULT_POSE),
                    help="拍摄姿态 x y z(mm) rx ry rz(deg)（真机 pose 那 6 个数）")
    ap.add_argument("--frame", default="auto", choices=("auto", "cam", "end"),
                    help="点云自带坐标系（auto = 用 ROI 反投影判定）")
    ap.add_argument("--extrinsic", default=str(pc.DEFAULT_EXTRINSIC), help="手眼标定 XML")
    ap.add_argument("--intrinsic", default=str(pc.DEFAULT_INTRINSIC), help="相机内参 XML")
    ap.add_argument("--engine", default="auto", choices=("auto", "field", "outline"),
                    help="规划算法：auto = 有红/青检测边用现场取边、否则通用轮廓；"
                         "field = 只用现场取边（toilet 实例）；outline = 只用通用轮廓")
    ap.add_argument("--mode", default="ring+raster",
                    choices=("ring", "spiral", "raster", "ring+raster"), help="路径走法")
    ap.add_argument("--standoff", type=float, default=0.0,
                    help="工具尖到目标点的距离[mm]；0 = 自动（现场取边按 footprint 收紧，通用轮廓 400）")
    ap.add_argument("--step", type=float, default=0.0,
                    help="相邻路径间距[mm]；0 = 自动（现场取边按喷锥足迹，通用轮廓 40）")
    ap.add_argument("--inset", type=float, default=10.0, help="轮廓内缩[mm]（通用轮廓用）")
    ap.add_argument("--snap", type=float, default=40.0, help="目标点吸附到点云的半径[mm]（通用轮廓用）")
    ap.add_argument("--lip", action="store_true", help="现场取边时加唇遍（沿收缩边，默认不加）")
    ap.add_argument("--ring-offset", type=float, default=40.0,
                    help="现场取边内缩圈的内缩量[mm]（<=0 = 按足迹半径）")
    ap.add_argument("--ring-standoff", type=float, default=-1.0,
                    help="现场取边内缩圈单独喷距[mm]（<0 = 与其它遍相同）")
    ap.add_argument("--footprint-frac", type=float, default=0.35,
                    help="现场取边 auto 喷距时“足迹 <= F×口短边”的 F（0 = 不限）")
    ap.add_argument("--speed", type=float, default=160.0, help="仿真速度[mm/s]")
    ap.add_argument("--approach", type=float, default=150.0, help="接近距离[mm]")
    ap.add_argument("--max-points", type=int, default=420, help="路径点上限")
    ap.add_argument("--ik-iters", type=int, default=180, help="单点 IK 迭代上限")
    ap.add_argument("--no-ik", action="store_true", help="只出几何路径（不解 IK、不建模型）")
    ap.add_argument("--sim", action="store_true", help="建模型并跑一遍仿真（不显示窗口）")
    ap.add_argument("--render", default="", help="离屏渲染一张图（路径 + 点云 + 机械臂）")
    ap.add_argument("--video", default="", help="仿真录像（mp4，需 imageio-ffmpeg）")
    ap.add_argument("--json", default="", help="把路径写成 JSON（tcp/targets/q/报告）")
    ap.add_argument("--selftest", action="store_true", help="无窗口自检")
    args = ap.parse_args(argv)

    if args.selftest:
        return run_selftest(args)

    path = Path(args.cloud) if args.cloud else default_cloud()
    inputs = load_inputs(path, pose=args.pose, frame=args.frame,
                         extrinsic=args.extrinsic, intrinsic=args.intrinsic)
    st = PlanSettings(mode=args.mode, standoff_mm=args.standoff, step_mm=args.step,
                      inset_mm=args.inset, snap_mm=args.snap, speed_mm_s=args.speed,
                      max_points=args.max_points, approach_mm=args.approach,
                      ik_iters=args.ik_iters, engine=args.engine, lip=args.lip,
                      ring_offset_mm=args.ring_offset, ring_standoff_mm=args.ring_standoff,
                      footprint_frac=args.footprint_frac)
    sim = None
    scene = None
    if not args.no_ik or args.sim or args.render or args.video:
        scene = pc.build_scene(inputs["points"], inputs["cloud"].depth_mm,
                               scene_out=pc.ASSETS / "revA1_plan_scene.xml",
                               prefix="pc_plan", base_scene=pc.BASE_SCENE,
                               cart_height=float(pc.DEFAULT_CART_HEIGHT_M))
        sim = core.ArmSim(scene.scene, log_scene=False)
    plan = build_plan(inputs["points"], st, seeds=inputs["seeds"],
                      camera=inputs["camera"], sim=sim,
                      cloud_path=str(path), pose=args.pose)
    print(f"\n点云    : {path.name}（{inputs['cloud'].n} 点，"
          f"{inputs['cloud'].source}，{inputs['frame']}）")
    _print(plan.report)
    print(f"工具尖  : 起点 {np.round(plan.tcp[0], 3).tolist()} → "
          f"终点 {np.round(plan.tcp[-1], 3).tolist()}")

    if args.json:
        payload = dict(
            cloud=str(path), frame=inputs["frame"], mode=args.mode,
            settings=st.__dict__, stats=plan.stats, report=plan.report,
            plane=dict(center=plan.center.tolist(), normal=plan.normal.tolist(),
                       u=plan.u.tolist(), v=plan.v.tolist(),
                       radius_mm=plan.radius_mm, source=plan.seed_source),
            rim=np.round(plan.rim, 6).tolist(),
            targets=np.round(plan.targets, 6).tolist(),
            normals=np.round(plan.normals, 6).tolist(),
            tcp=np.round(plan.tcp, 6).tolist(),
            q_deg=[[None if not np.isfinite(v) else round(float(v), 3) for v in row]
                   for row in np.degrees(plan.q)],
            ok=plan.ok.astype(int).tolist(),
        )
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"路径 JSON：{out}")

    if args.sim and sim is not None and plan.runs:
        res = simulate(plan, sim)
        print(f"仿真    : {res['steps']} 步 / {res['seconds']:.1f} s，"
              f"工具尖轨迹 {res['tip_path'].shape[0]} 点")
    if args.render:
        out = render_preview(scene.scene, args.render, plan, sim=sim)
        print(f"预览图  : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

