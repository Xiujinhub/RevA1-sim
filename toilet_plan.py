#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
toilet_plan_from_cloud.py —— **只用点云里"真实点"**规划马桶盆口喷淋轨迹（不拿检测出的边缘折线当内圆）。

【为什么换口径：现场核对 view_box_cloud.py】
    P 键导出的 `box_cloud_*.json` 里，`boxes[].points[]` 的每个点带一个 `"hl"` 字段：
        `"edge"`     = 程序里被染成边缘线颜色的像素（红 = 分割边缘线，青 = toilet 收缩边缘 ×0.9）；
        `"keypoint"` = 被染成蓝色的点（检测框中心 / 内部离散点 / 各类采样点）；
        没有 `hl`     = 未染色的真实点云点。
    在 `view_box_cloud.py` 里按 `L` 隐藏高亮层后留下的（未染色）点才是真实点 ——
    检测出的那两条折线是**逐点采样 + 逐点取深度**出来的：深度噪声大、还夹着"跑进碗里/跳到外壁"的
    误检段；把它们直接当"盆口内圆"，轨迹就会在那些点跳。**而点云本身是密而连续的**：
    盆口边缘在点云里是一条光滑的**脊线**（沿视线方向的深度极小处）。
    所以本脚本：盆口形状**全部从点云取**；检测结果只用来"自举"（中心/法向/半径初值），
    并且默认把染色点从**取边用的工作点云**里剔掉（`--drop-hl all`）。

【链路】(几何/规划原语全部复用 python/spray_plan 公共层, 不重复实现)
    ① 流式解析 box_cloud：o2e 点 + hl 标记（+ 高亮分组，仅作自举）
    ② 工作点云 = 未染色点（`--drop-hl`），自举盆口面（染色边线 PCA）→ 迭代：
          面内极坐标 (θ, r) + 视线深度 z → 逐角向在半径带内找 **z 的局部极小 = 脊线**
          → 用脊线点重拟合盆口面（中心/法向/半径）
    ③ 脊线 → 圆滑 → 闭合 → `plane_frame`（定型：n 朝机器人、u 朝 +Z）
    ④ 航点：`make_ring_pass`（内缩圈）+ `make_raster_pass`（栅格），可选 `make_rim_passes`（唇遍，`--lip`）
       栅格尾巴：`prune_short_raster_lines()` 把"被内缩裁到只剩 1 个点的扫描线"整条剔掉
       （`--raster-min-pts`，默认 2）—— 否则 S 形轨迹末尾会多出一个孤零零的"尾巴点"
    ⑤ `compute_poses`（法兰/TCP/入射角）+ `compute_clearance`（工具轴线到点云最小间隙）
       + `compute_coverage`（足迹覆盖率）
    ⑥ 输出 `_cloud_plan.json` / `.csv` / `_moves.txt`（TB6 MoveS，格式与既有喷洗轨迹一致）/ 可选 PNG

【与检测边线的区别（实测 212257 帧）】
    云点 431369：未染色 356002、`hl=edge` 3680、`hl=keypoint` 71687。
    盆口半径带里沿面法向的深度剖面：**半径 ≈ 180~205mm 处是脊线**，口内一侧深度一路 +100mm、
    外壁一侧到 −140mm；脊线处深度连续，逐角向取中位数后没有"突变点"。

【用法】(yolov9 环境)
    python python/toilet/toilet_plan_from_cloud.py python/toilet/point-cloud/box_cloud_20260919_212257.json
    python python/toilet/toilet_plan_from_cloud.py <json> --no-save          # 只看指标
    python python/toilet/toilet_plan_from_cloud.py <json> --drop-hl none     # 染色点也留（对比）
    python python/toilet/toilet_plan_from_cloud.py <json> --rim-bin-deg 3 --rim-band 70
    python python/toilet/toilet_plan_from_cloud.py <json> --standoff 96.2    # 用生产固定喷距
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                            # Windows 控制台默认 GBK，中文会抛 UnicodeEncodeError
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
# 公共层随本工程一起拷贝在 <RevA1-sim>/spray_plan/（现场 python/spray_plan 的原样副本）
_SHARED = os.path.abspath(os.path.join(_HERE, 'spray_plan'))
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from spray_common import (CAPTURE_ANGLES_DEG, CAPTURE_T_EE_MM, CLEAR_WARN_MM, COVER_CELL_MM,
                          EDGE_INSET_MM, RASTER_TOP_TO_BOTTOM, RIM_DWELL_S, RIM_SPLIT_HALVES,
                          RING_OFFSET_MM, RING_STANDOFF_MM, SPRAY_CONE_DEG, SPRAY_OVERLAP,
                          SPRAY_STANDOFF_MM, TCP_MM, euler_to_R, poly_area)
from spray_planner import (Pass, assemble_passes, close_rim_curve, compute_clearance,
                           compute_coverage, compute_poses, footprint_spacing, make_ring_pass,
                           make_rim_passes, make_raster_pass, o2e_to_base, plane_frame)

SEP = '=' * 78
TOOL_CN = {'tool1': '夹爪', 'tool2': '工具2', 'nozzle1': '喷嘴1', 'nozzle2': '喷嘴2'}
ARRIVAL_ANGLES = np.array([173.45, -26.81, 287.50])
NOZZLE_TOOL, GRIPPER_TOOL = 'nozzle1', 'tool1'
CORNER_ROUND, CORNER_MAX_TURN_DEG, CORNER_MIN_STEP_MM = True, 15.0, 5.0
RING_MERGE_MM, RING_TURN_SOFT_FINE, RING_ROUND_MAX_IT = 2.0, 12.0, 200


def log(msg: str = '') -> None:
    print(msg, flush=True)



# ===========================================================================
# ① 读 box_cloud：点云 + hl 标记 + 高亮分组（分组只用于自举）
# ===========================================================================

_PT_RE = re.compile(r'\{\s*"px":\s*(-?\d+),\s*"py":\s*(-?\d+),\s*"o2c":\s*\[([^\]]*)\],\s*'
                    r'"o2e":\s*\[([^\]]*)\](?:,\s*"rgb":\s*\[([^\]]*)\])?'
                    r'(?:,\s*"hl":\s*"([^"]*)")?\s*\}')
_GRP_RE = re.compile(r'\{\s*"index":\s*(-?\d+),\s*"kind":\s*"([^"]*)",\s*"target":\s*"([^"]*)"'
                     r',\s*"label":\s*"([^"]*)",\s*"closed":\s*(true|false),'
                     r'\s*"rgb":\s*\[([^\]]*)\]')
_XYZ_RE = re.compile(r'"o2e":\s*\[([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+)\]')


def load_box_cloud(path: str) -> dict:
    """读 box_cloud JSON：全部点（o2e）+ 逐点 hl 标记 + 高亮分组 + 元信息。"""
    txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
    pts, hl, rgb = [], [], []
    for m in _PT_RE.finditer(txt):
        x, y, z = [float(v) for v in m.group(4).split(',')]
        pts.append((x, y, z))
        hl.append(m.group(6) or '')
        rgb.append([float(v) for v in m.group(5).split(',')] if m.group(5)
                   else [0.0, 0.0, 0.0])
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    hl = np.asarray(hl)
    rgb = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    groups = []
    hi = txt.find('"highlight_points"')
    if hi >= 0:
        sub = txt[hi:]
        hdrs = list(_GRP_RE.finditer(sub))
        for k, m in enumerate(hdrs):
            seg = sub[m.start(): hdrs[k + 1].start() if k + 1 < len(hdrs) else len(sub)]
            tri = _XYZ_RE.findall(seg)
            gp = np.asarray([[float(a), float(b), float(c)] for a, b, c in tri], dtype=np.float64)
            groups.append({'index': int(m.group(1)), 'kind': m.group(2), 'target': m.group(3),
                           'label': m.group(4), 'closed': m.group(5) == 'true',
                           'rgb': tuple(int(float(v)) for v in m.group(6).split(',')), 'pts': gp})
    meta = {'file': os.path.abspath(path), 'bytes': len(txt)}
    m = re.search(r'"timestamp":\s*"([^"]*)"', txt)
    meta['timestamp'] = m.group(1) if m else ''
    i = txt.find('"T_cam_end"')
    meta['T_cam_end'] = None
    if i >= 0:
        seg = txt[i:i + 4000]
        closed = seg.find(']]')
        body = seg[:closed + 2] if closed > 0 else seg[:600]
        nums = [float(v) for v in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', body)]
        if len(nums) >= 16:
            meta['T_cam_end'] = np.asarray(nums[:16], dtype=np.float64).reshape(4, 4)
    return {'pts': pts, 'hl': hl, 'rgb': rgb, 'groups': groups, 'meta': meta}


def view_axis_o2e(T_cam_end: Optional[np.ndarray] = None) -> np.ndarray:
    """视线方向（o2e 系，单位向量，指向"越来越远"）。

    相机装在末端、光轴 ≈ 末端 +z：T_cam_end 的第三列就是相机 z 在 o2e 里的表示（文件里带就用它）。
    """
    if T_cam_end is not None:
        z = np.asarray(T_cam_end, dtype=np.float64)[:3, 2]
        nz = float(np.linalg.norm(z))
        if nz > 1e-9:
            return z / nz
    return np.array([0.0, 0.0, 1.0])

# ===========================================================================
# ② 点云取边：逐角向找"视线深度的局部极小 = 盆口脊线"
# ===========================================================================

def neighbors_within(cloud: np.ndarray, query: np.ndarray, rad_mm: float) -> np.ndarray:
    """取 query 各点在 cloud 里 rad_mm 球形邻域内的点（体素网格加速，避免 N×M 距离矩阵）。"""
    cloud = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
    query = np.asarray(query, dtype=np.float64).reshape(-1, 3)
    if not len(cloud) or not len(query):
        return np.zeros((0, 3))
    cell = max(float(rad_mm), 1e-6)
    grid: dict = {}
    for i, k in enumerate(np.floor(cloud / cell).astype(np.int64)):
        grid.setdefault((int(k[0]), int(k[1]), int(k[2])), []).append(i)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    hit: set = set()
    for q in query:
        b = np.floor(q / cell).astype(np.int64)
        cand: list = []
        for o in offs:
            cand += grid.get((int(b[0] + o[0]), int(b[1] + o[1]), int(b[2] + o[2])), [])
        if not cand:
            continue
        a = cloud[cand]
        d = np.linalg.norm(a - q, axis=1)
        hit.update(int(c) for c in np.asarray(cand)[d <= rad_mm])
    if not hit:
        return np.zeros((0, 3))
    return cloud[np.fromiter(sorted(hit), dtype=np.int64, count=len(hit))]


def nearest_dist(cloud: np.ndarray, query: np.ndarray, rad_mm: float = 15.0) -> np.ndarray:
    """query 每点到 cloud 的最近距离（超过 rad_mm 的直接记为 rad_mm，用于"是否贴面"判据）。"""
    cloud = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
    query = np.asarray(query, dtype=np.float64).reshape(-1, 3)
    out = np.full(len(query), float(rad_mm))
    if not len(cloud) or not len(query):
        return out
    cell = max(float(rad_mm), 1e-6)
    grid: dict = {}
    for i, k in enumerate(np.floor(cloud / cell).astype(np.int64)):
        grid.setdefault((int(k[0]), int(k[1]), int(k[2])), []).append(i)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for j, q in enumerate(query):
        b = np.floor(q / cell).astype(np.int64)
        cand: list = []
        for o in offs:
            cand += grid.get((int(b[0] + o[0]), int(b[1] + o[1]), int(b[2] + o[2])), [])
        if cand:
            out[j] = min(float(rad_mm),
                         float(np.min(np.linalg.norm(cloud[cand] - q, axis=1))))
    return out


def nearest_point(cloud: np.ndarray, query: np.ndarray, rad_mm: float = 12.0):
    """query 每点最近的 cloud 点：返回 (最近点 (m,3) 无则 NaN, 距离 (m,)）。"""
    cloud = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
    query = np.asarray(query, dtype=np.float64).reshape(-1, 3)
    out = np.full((len(query), 3), np.nan)
    dist = np.full(len(query), np.nan)
    if not len(cloud) or not len(query):
        return out, dist
    cell = max(float(rad_mm), 1e-6)
    grid: dict = {}
    for i, k in enumerate(np.floor(cloud / cell).astype(np.int64)):
        grid.setdefault((int(k[0]), int(k[1]), int(k[2])), []).append(i)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for j, q in enumerate(query):
        b = np.floor(q / cell).astype(np.int64)
        cand: list = []
        for o in offs:
            cand += grid.get((int(b[0] + o[0]), int(b[1] + o[1]), int(b[2] + o[2])), [])
        if not cand:
            continue
        a = cloud[cand]
        dd = np.linalg.norm(a - q, axis=1)
        k = int(np.argmin(dd))
        if float(dd[k]) <= rad_mm:
            out[j], dist[j] = a[k], float(dd[k])
    return out, dist


def rim_plane_from_seed(cloud: np.ndarray, seed: np.ndarray, nb_mm: float = 25.0,
                        n_round: int = 3, trim_mm: float = 8.0):
    """盆口面拟合（解 212205/212253 那种"检测只描出一段弧 → PCA 平面歪"）。

    检测点本身**深度是跳的**（同一点云上可差几百 mm），但**它邻域的点云是连续的**：所以先在点云里
    取每个检测点 nb_mm 邻域的点，再用"最小二乘 + 剔尾"迭代拟合平面。nb_mm 要够大（默认 25mm）：
    邻域太小就变成一个厚球团，"最小方差方向"不是面法向；够大时点云里**平坦口沿面**占主导 → 法向正确。

    但那个邻域点集的质心**不能当中心**：口沿平台向外延伸几十 mm，质心外移会让半径全错。
    所以：**法向取自点云口沿面，中心用检测环质心**，面高取检测点在该法向下的中位高度。
    """
    c_r, n_r, u_r, _ = plane_axes(seed)                # 检测环：中心/横向基可靠
    pts = neighbors_within(cloud, seed, float(nb_mm))
    if len(pts) < 50:
        return plane_axes(seed)
    n = n_r
    for _ in range(max(1, int(n_round))):
        c_f, n_f, _, _ = plane_axes(pts, n_ref=n)
        d = (pts - c_f) @ n_f
        keep = np.abs(d - np.median(d)) <= max(float(trim_mm), 1e-6)
        n = n_f
        if int(keep.sum()) < 50:
            break
        pts = pts[keep]
    if float(n @ n_r) < 0.0:
        n = -n
    off = float(np.median((seed - c_r) @ n))           # 面高 = 检测点在该法向下的中位高度
    c = c_r + off * n
    u = u_r - float(u_r @ n) * n
    nu = float(np.linalg.norm(u))
    if nu < 1e-9:
        u = np.cross(n, np.array([0.0, 0.0, 1.0]))
        nu = float(np.linalg.norm(u))
    u = u / max(nu, 1e-12)
    v = np.cross(n, u)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    return c, n, u, v


def extract_rim_from_cloud(cloud: np.ndarray, seed: np.ndarray, view_axis: np.ndarray,
                           bin_deg: float = 4.0, band_mm: float = 20.0, r_step_mm: float = 2.0,
                           min_pts_bin: int = 20, n_iter: int = 2, smooth_k: int = 1,
                           slope_tol: float = 0.30, run_flat: int = 2, plane_nb_mm: float = 25.0,
                           r_tol_mm: float = 8.0, h_tol_mm: float = 6.0, plat_tol_mm: float = 3.0,
                           plane: Optional[tuple] = None, h_ceil_mm=None, h_ceil_tol_mm: float = 2.0,
                           n_round: int = 3, snap_mm: float = 12.0
                           ) -> Tuple[np.ndarray, dict]:
    """从点云取盆口边缘线：**半径沿检测边线，高度取点云里该角向的真实边缘点**。

    - 横向（X/Y）：检测边线的半径逐角向可靠，直接用它，不让点云去改半径；
    - 高度（Z）  ：带内按高度直方图找**口沿平台**（面积最大的水平面 = 点数最多的那一层），
                  平台上取最接近检测半径的点、±3mm 邻域中位 → 真实点云点；
                  **输出直接给实测点云点**（不重建，避免被压成拟合平面）；
    - 平面       ：法向取自点云口沿面、中心取检测环质心；可用 `plane=` 指定共用平面（两条边同口径）；
    - 高度上限   ：`h_ceil_mm`（逐角向）。收缩边（青）在口内、不可能高过口沿（红）→ 用红边高度当上限，
                  否则个别箱会吸到口沿平台上（表现为"个别点突然变高、深度变小"）；
    - 鲁棒       ：逐箱 (r, h) 与环形邻域中值差超阈 → 用邻域中值替换，**多轮**（防成簇尖峰）。

    参数：cloud 工作点云；seed 检测边缘点；view_axis 视线方向；bin_deg 角向箱宽；band_mm 搜索带半宽；
          min_pts_bin 每箱最少点；n_iter 面重拟合轮数；smooth_k 补齐值环形中值半窗；
          plane_nb_mm 求法向的邻域半径；r_tol_mm/h_tol_mm 剔点阈值；plat_tol_mm 平台厚度容差；
          plane 指定共用平面 (c, n, u, v)；h_ceil_mm 逐角向高度上限；n_round 剔点轮数；
          snap_mm 补齐点吸附到点云的搜索半径。
    返回 (rim_pts (m,3) 按 θ 排序, info)
    """
    cloud = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
    if plane is not None:
        c, n, u, v = [np.asarray(a, dtype=np.float64).ravel() for a in plane]
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        n_iter = 1                       # 指定共用平面时不再重拟合（否则两条边口径会漂）
    else:
        c, n, u, v = rim_plane_from_seed(cloud, seed, nb_mm=plane_nb_mm)
    r0 = float(np.median(np.hypot((seed - c) @ u, (seed - c) @ v)))     # 自举半径（**固定**）
    axv = np.asarray(view_axis, dtype=np.float64).ravel()
    axv = axv / max(float(np.linalg.norm(axv)), 1e-12)
    mids = np.arange(-180.0 + 0.5 * float(bin_deg), 180.0, float(bin_deg))
    info: dict = {'bin_deg': float(bin_deg), 'band_mm': float(band_mm), 'r_seed_mm': r0,
                  'slope_tol': float(slope_tol), 'plane_nb_mm': float(plane_nb_mm),
                  'r_tol_mm': float(r_tol_mm), 'h_tol_mm': float(h_tol_mm),
                  'view_axis': np.round(axv, 4).tolist(), 'iter': []}
    rim = np.zeros((len(mids), 3))
    rim_meas = None
    for it in range(max(1, int(n_iter))):
        rel = cloud - c
        rad = np.hypot(rel @ u, rel @ v)
        ang = np.degrees(np.arctan2(rel @ v, rel @ u))
        d_pl = rel @ n                                    # 沿盆口面法向的高度
        band = (rad > r0 - band_mm) & (rad < r0 + band_mm)
        inner = (rad < 0.5 * r0) & (rad > 0.12 * r0)
        if plane is None and int(inner.sum()) > 50 and int(band.sum()) > 50:
            # 口内（碗里）一定比口沿低：用符号把 d 定成"口内为负"，便于判"陡升段"
            # （**指定共用平面时不做**：符号已由盆口边缘那条线定好，再翻会整体取反）
            if float(np.median(d_pl[inner])) < float(np.median(d_pl[band])):
                d_pl = -d_pl
        ok_bin = np.zeros(len(mids), dtype=bool)
        r_prof = np.full(len(mids), np.nan)
        d_prof = np.full(len(mids), np.nan)
        spread = np.full(len(mids), np.nan)
        n_hit = np.zeros(len(mids), dtype=int)
        # 每个角向的"检测半径"（横向可靠；深度才是它跳的地方）
        s_rad = np.hypot((seed - c) @ u, (seed - c) @ v)
        s_ang = np.degrees(np.arctan2((seed - c) @ v, (seed - c) @ u))
        r_ref = np.full(len(mids), np.nan)
        for i, a0 in enumerate(mids):
            mm = (s_ang >= a0 - 0.5 * float(bin_deg)) & (s_ang < a0 + 0.5 * float(bin_deg))
            if int(mm.sum()) >= 3:
                r_ref[i] = float(np.median(s_rad[mm]))
        r_seed_fin = r_ref.copy()
        n_seed_bin = int(np.sum(np.isfinite(r_seed_fin)))   # 检测点覆盖到的角向箱数（完整圈 = n_bin）
        r_ref = circ_interp(mids, r_ref)
        # ---- 青（收缩边）专用：带内**分簇 + 按角向连续性跟踪**选簇 ----------------
        # 收缩半径落到"口沿内唇"附近时，带里会同时有平台(+几 mm)和碗壁(-几十 mm)，
        # "取点数最多的那一层"会在这两者间翻转 → 表现为个别点突然变高/变低。
        # 所以：先把带内高度分簇，再沿角向按高度连续性（线性外推）跟踪挑簇。
        if h_ceil_mm is not None:
            cands: list = []
            for i, a0 in enumerate(mids):
                if not np.isfinite(r_ref[i]):
                    cands.append([])
                    continue
                m = ((ang >= a0 - 0.5 * float(bin_deg)) & (ang < a0 + 0.5 * float(bin_deg))
                     & (np.abs(rad - r_ref[i]) <= band_mm))
                m_all = m
                ceil_i = float(np.asarray(h_ceil_mm, float)[i])
                relaxed = False
                if np.isfinite(ceil_i):
                    m = m_all & (d_pl <= ceil_i + h_ceil_tol_mm)
                    if int(m.sum()) < 3:
                        m, relaxed = m_all, True     # 上限内没点 → 退回全带（下面只留"最高簇"）
                if int(m.sum()) < 3:
                    cands.append([])
                    continue
                hb = d_pl[m]
                step_h = max(1.0, 0.5 * plat_tol_mm)
                edges = np.arange(float(hb.min()), float(hb.max()) + step_h + 1e-6, step_h)
                cnt, _ = np.histogram(hb, bins=edges)
                sm = np.convolve(cnt, np.ones(3), mode='same') if len(cnt) >= 3 else cnt
                pk = [k for k in range(len(sm))
                      if sm[k] > 0 and sm[k] >= sm[max(0, k - 2):k + 3].max() - 1e-9
                      and sm[k] >= 0.15 * sm.max()]
                cl = []
                for k in pk:
                    hc = 0.5 * (edges[k] + edges[k + 1])
                    sel = m & (np.abs(d_pl - hc) <= max(plat_tol_mm, step_h))
                    if int(sel.sum()) < 3:
                        continue
                    h_med = float(np.median(d_pl[sel]))
                    r_med = float(rad[sel][int(np.argmin(np.abs(rad[sel] - r_ref[i])))])
                    cl.append((h_med, r_med, int(sel.sum())))
                if relaxed and cl:                   # 退让时只留最高的那一簇（最接近口沿、且是真实点）
                    cl = [max(cl, key=lambda c: c[0])]
                cands.append(cl)
            # 起点：候选最少的箱（歧义最小）；并列时取点数最多的
            order = [i for i in range(len(mids)) if cands[i]]
            if order:
                start = min(order, key=lambda i: (len(cands[i]), -max(c[2] for c in cands[i])))
                sel_h: list = [np.nan] * len(mids)
                sel_h[start] = cands[start][0][0] if len(cands[start]) == 1 else \
                    max(cands[start], key=lambda c: c[2])[0]
                for direction in (1, -1):                    # 两侧环形跟踪
                    prev1 = sel_h[start]
                    prev2 = sel_h[start]
                    for step_i in range(1, len(mids)):
                        i = (start + direction * step_i) % len(mids)
                        if not cands[i]:
                            sel_h[i] = np.nan
                            continue
                        ref = 2.0 * prev1 - prev2 if np.isfinite(prev2) else prev1
                        best = min(cands[i], key=lambda c: abs(c[0] - ref))
                        sel_h[i] = best[0]
                        prev2, prev1 = prev1, best[0]
            else:
                sel_h = [np.nan] * len(mids)
            # 用选中的簇重算该箱的实测点（半径取簇内最接近检测半径的点 ±3mm 邻域中位）
            for i, a0 in enumerate(mids):
                h_sel = sel_h[i]
                if not np.isfinite(h_sel):
                    continue
                m = ((ang >= a0 - 0.5 * float(bin_deg)) & (ang < a0 + 0.5 * float(bin_deg))
                     & (np.abs(rad - r_ref[i]) <= band_mm)
                     & (np.abs(d_pl - h_sel) <= max(plat_tol_mm, 0.5 * plat_tol_mm) + 3.0))
                if int(m.sum()) < 2:
                    continue
                r_star = float(rad[m][int(np.argmin(np.abs(rad[m] - r_ref[i])))])
                near = m & (np.abs(rad - r_star) <= 3.0)
                if int(near.sum()) < 2:
                    near = m
                rim[i] = np.median(cloud[near], axis=0)
                r_prof[i] = float(np.median(rad[near]))
                d_prof[i] = float(np.median(d_pl[near]))
                spread[i] = float(np.percentile(d_pl[near], 90)
                                  - np.percentile(d_pl[near], 10))
                n_hit[i] = int(near.sum())
                ok_bin[i] = True
        for i, a0 in enumerate(mids):
            if ok_bin[i] or not np.isfinite(r_ref[i]) or h_ceil_mm is not None:
                continue
            m = ((ang >= a0 - 0.5 * float(bin_deg)) & (ang < a0 + 0.5 * float(bin_deg))
                 & (np.abs(rad - r_ref[i]) <= band_mm))
            if int(m.sum()) < min_pts_bin:
                continue
            # 高度上限（青 = 收缩边在口内，不允许高过同角度的红边口沿）
            if h_ceil_mm is not None:
                ceil_i = float(np.asarray(h_ceil_mm, float)[i])
                if np.isfinite(ceil_i):
                    m_lim = m & (d_pl <= ceil_i + h_ceil_tol_mm)
                    if int(m_lim.sum()) >= 3:        # 有 3 个点就够定中位
                        m = m_lim
                    else:
                        continue                      # **不退回无上限集合**（否则平台又被吸回来）
            # ① 带内找**口沿平台**：高度直方图的峰（平台是面积最大的水平面，点数远多于内外壁）
            hb = d_pl[m]
            lo, hi = float(np.min(hb)), float(np.max(hb))
            if hi - lo > 1e-6:
                edges = np.arange(lo, hi + plat_tol_mm + 1e-6, max(1.0, float(plat_tol_mm)))
                cnt, _ = np.histogram(hb, bins=edges)
                if len(cnt) >= 3:                        # 3 桶滑动和，抗单桶抖动
                    sm = np.convolve(cnt, np.ones(3), mode='same')
                else:
                    sm = cnt
                k = int(np.argmax(sm))
                h_mode = 0.5 * (edges[k] + edges[k + 1])
            else:
                h_mode = lo
            plat = m & (np.abs(d_pl - h_mode) <= plat_tol_mm)     # 平台点（厚度 ±plat_tol）
            if int(plat.sum()) < 5:
                plat = m
            # ② 半径：平台上**最接近检测半径**的点，±3mm 邻域取中位（真实点云值，压噪声）
            r_star = float(rad[plat][int(np.argmin(np.abs(rad[plat] - r_ref[i])))])
            near = plat & (np.abs(rad - r_star) <= 3.0)
            if int(near.sum()) < 2:
                near = plat
            rim[i] = np.median(cloud[near], axis=0)          # 该角向真实边缘点（含真实高度）
            r_prof[i] = float(np.median(rad[near]))
            d_prof[i] = float(np.median(d_pl[near]))
            spread[i] = float(np.percentile(d_pl[near], 90) - np.percentile(d_pl[near], 10))
            n_hit[i] = int(near.sum())
            ok_bin[i] = True
        # 环形鲁棒剔点（**多轮**，防成簇尖峰）：与邻域中值差太多 = 串到别的台面，用邻域中值替换
        n_fix = 0
        fixed = np.zeros(len(mids), dtype=bool)
        for _round in range(max(1, int(n_round))):
            if int(ok_bin.sum()) < 8:
                break
            changed = 0
            for arr, tol in ((r_prof, r_tol_mm), (d_prof, h_tol_mm)):
                arr_f = circ_interp(mids, np.where(ok_bin, arr, np.nan))
                if not np.isfinite(arr_f).all():
                    continue
                med = circ_median(arr_f, 2)
                bad = ok_bin & (np.abs(arr_f - med) > tol)
                if int(bad.sum()):
                    changed += int(bad.sum())
                    n_fix += int(bad.sum())
                    fixed |= bad
                    arr[:] = np.where(bad, med, arr_f)
            if changed == 0:
                break

        if int(ok_bin.sum()) < 8:
            break
        r_f = circ_interp(mids, np.where(ok_bin, r_prof, np.nan))
        d_f = circ_interp(mids, np.where(ok_bin, d_prof, np.nan))
        if not (np.isfinite(r_f).all() and np.isfinite(d_f).all()):
            break
        rim_meas = rim.copy()                       # 本轮**实测**的边缘点（含真实高度）
        # 用取到的边缘点重拟合盆口面（中心 = 边缘中位点；法向 = 边缘点 PCA）
        c_old = c
        c = np.median(rim[ok_bin], axis=0)
        _, n_new, u_new, v_new = plane_axes(rim[ok_bin], n_ref=n)
        n, u, v = n_new, u_new, v_new
        # 回到 (r, θ, d)：**d 是实测高度，必须带上**（只重建 r(θ) 会把曲线压成平面）
        rim = (c[None, :] + (r_f * np.cos(np.radians(mids)))[:, None] * u[None, :]
               + (r_f * np.sin(np.radians(mids)))[:, None] * v[None, :]
               + d_f[:, None] * n[None, :])
        info['iter'].append({'iter': it + 1, 'n_ok_bin': int(ok_bin.sum()),
                             'r_median_mm': round(float(np.median(r_f)), 2),
                             'c': np.round(c, 2).tolist(), 'n': np.round(n, 4).tolist(),
                             'dc_mm': round(float(np.linalg.norm(c - c_old)), 2)})


    # (r, d) 环形**中值**滤波：只用于"缺箱/被剔箱"的补齐值（实时点直接输出，不抹平）
    rel = rim - c
    rr = np.hypot(rel @ u, rel @ v)
    dd = rel @ n
    k = max(0, int(smooth_k))
    rr, dd = circ_median(rr, k), circ_median(dd, k)
    rim_rebuild = (c[None, :] + (rr * np.cos(np.radians(mids)))[:, None] * u[None, :]
                   + (rr * np.sin(np.radians(mids)))[:, None] * v[None, :]
                   + dd[:, None] * n[None, :])
    # **输出：能取到实测点的箱直接用实测点云点**（它就躺在点云上 = 肉眼看到的"贴合"）；
    # 缺箱 / 串台面被剔的箱先用重建值补，再**吸附到最近的实测点云点**（补出来的点也得在点云上，
    # 否则会跟邻居差出一大截 → 看起来"个别点突然变高/变低"）。
    n_snap = 0
    if rim_meas is not None:
        fill = ~ok_bin | fixed
        rim = np.where(fill[:, None], rim_rebuild, rim_meas)
        bad_row = ~np.isfinite(rim).all(axis=1)
        rim[bad_row] = rim_rebuild[bad_row]
        if int(fill.sum()):
            snap, sd = nearest_point(cloud, rim[fill], float(snap_mm))
            got = np.isfinite(sd)
            if int(got.sum()):
                idx_fill = np.where(fill)[0][got]
                rim[idx_fill] = snap[got]
                n_snap = int(got.sum())
    else:
        rim = rim_rebuild
    # 高度贴合度：重建高度 dd 与本轮**实测**边缘点在同一法向下的差（只统计没被剔点替换的箱）
    hug = np.zeros(0)
    if rim_meas is not None and int(ok_bin.sum()) >= 8:
        sel_bin = ok_bin & ~fixed
        hug = np.abs(dd[sel_bin] - ((rim_meas[sel_bin] - c) @ n))
    seg = np.linalg.norm(np.diff(np.vstack([rim, rim[:1]]), axis=0), axis=1)
    dep = rim @ axv
    slope = np.abs(np.diff(np.r_[dep, dep[:1]])) / np.maximum(seg, 1e-6)
    info.update({'n_bin': len(mids), 'n_ok_bin': int(np.sum(ok_bin)),
                 'r_min_mm': float(np.min(rr)), 'r_max_mm': float(np.max(rr)),
                 'r_median_mm': float(np.median(rr)),
                 'perimeter_mm': float(seg.sum()), 'depth_span_mm': float(dep.max() - dep.min()),
                 'slope_median': float(np.median(slope)), 'slope_max': float(slope.max()),
                 'step_median_mm': float(np.median(seg)), 'step_max_mm': float(seg.max()),
                 'jump_frac': float(np.mean(seg > 5.0 * max(float(np.median(seg)), 1e-6))),
                 'plane_rms_mm': float(np.sqrt((dd ** 2).mean())),
                 'relief_span_mm': float(dd.max() - dd.min()),
                 'n_fix_bin': int(n_fix), 'n_snap_bin': int(n_snap), 'n_hug_bin': int(len(hug)),
                 'n_seed_bin': int(n_seed_bin),
                 'n_hit_median': float(np.median(n_hit[ok_bin]))
                 if int(ok_bin.sum()) else float('nan'),
                 'spread_median_mm': float(np.nanmedian(spread)) if np.isfinite(spread).any()
                 else float('nan'),
                 'spread_max_mm': float(np.nanmax(spread)) if np.isfinite(spread).any()
                 else float('nan'),
                 'hug_median_mm': float(np.median(hug)) if len(hug) else float('nan'),
                 'hug_max_mm': float(np.max(hug)) if len(hug) else float('nan'),
                 'bin_r_mm': [round(float(x), 2) for x in r_prof],
                 'bin_h_mm': [round(float(x), 2) for x in d_prof],
                 'bin_ok': [int(x) for x in ok_bin],
                 'bin_fixed': [int(x) for x in fixed],
                 'c_o2e': np.round(c, 2).tolist(), 'n_o2e': np.round(n, 4).tolist(),
                 'u_o2e': np.round(u, 5).tolist(), 'v_o2e': np.round(v, 5).tolist()})
    return rim, info


def edge_points(data: dict, want: str) -> np.ndarray:
    """取"检测边缘点"里的一类：
        want='red'  红 = **盆口边缘**（mask 分割边缘线）
        want='cyan' 青 = **边缘收缩后的点**（图像里 ×0.9，往口内收）
    优先用点云里 `hl='edge'` 且染色匹配的点（同一批点、且在密集点云里），
    没有就退回 highlight_points 里对应的分组。
    """
    hl, rgb = data['hl'], data['rgb']
    edge = hl == 'edge'
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if want == 'red':
        m = edge & (r > 150.0) & (g < 100.0) & (b < 100.0)
    else:
        m = edge & (g > 150.0) & (b > 150.0) & (r < 100.0)
    if int(m.sum()) >= 50:
        return data['pts'][m]

    def ok(c):
        return ((c[0] > 150 and c[1] < 100 and c[2] < 100) if want == 'red'
                else (c[0] < 100 and c[1] > 150 and c[2] > 150))

    for gp in data['groups']:
        if gp['kind'] == 'edge' and ok(gp['rgb']) and len(gp['pts']) >= 50:
            return gp['pts']
    return np.zeros((0, 3))


def seed_edges(data: dict) -> np.ndarray:
    """红 + 青一起（自举/兼容旧调用）。"""
    out = [a for a in (edge_points(data, 'red'), edge_points(data, 'cyan')) if len(a)]
    return np.vstack(out) if out else np.zeros((0, 3))


def plane_axes(pts: np.ndarray, n_ref: Optional[np.ndarray] = None):
    """最小二乘平面：返回 (质心 c, 单位法向 n, 面内基 u/v)。n_ref 用来定符号（避免翻转）。"""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    c = p.mean(axis=0)
    _, _, vh = np.linalg.svd(p - c, full_matrices=False)
    n, u, v = vh[2].copy(), vh[0].copy(), vh[1].copy()
    for vec in (n, u, v):
        vec /= max(float(np.linalg.norm(vec)), 1e-12)
    if n_ref is not None and float(n @ np.asarray(n_ref, float).ravel()) < 0.0:
        n = -n
    return c, n, u, v


def circ_median(val: np.ndarray, k: int) -> np.ndarray:
    """环形中值滤波（k=半窗）；比均值滤波更能保住真实的起伏/台肩。"""
    v = np.asarray(val, dtype=np.float64)
    k = int(k)
    if k <= 0 or len(v) < 3:
        return v.copy()
    out = np.empty_like(v)
    for i in range(len(v)):
        out[i] = np.median([v[(i + o) % len(v)] for o in range(-k, k + 1)])
    return out


def circ_interp(ang_deg: np.ndarray, val: np.ndarray) -> np.ndarray:
    """按角度环形插值补缺（val 允许 NaN）。"""
    val = np.asarray(val, dtype=np.float64)
    ok = np.isfinite(val)
    if int(ok.sum()) < 3:
        return np.full_like(val, np.nan)
    a = np.r_[np.asarray(ang_deg, float)[ok], np.asarray(ang_deg, float)[ok][:1] + 360.0]
    x = np.r_[val[ok], val[ok][0]]
    return np.interp(ang_deg, a, x)


# ===========================================================================
# ③ 规划主流程（几何/规划原语全在 python/spray_plan 公共层）
# ===========================================================================

class _UVFrame:
    """只给 make_rim_passes 用的轻量"面内多边形"（它内部只用 .uv）。"""

    def __init__(self, uv):
        self.uv = np.asarray(uv, dtype=np.float64)


def voxel_downsample(pts: np.ndarray, voxel_mm: float) -> np.ndarray:
    """体素抽稀（只用于间隙检查加速；0 = 不抽）。"""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if voxel_mm <= 0 or len(p) == 0:
        return p
    keys = np.floor(p / float(voxel_mm)).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return p[np.sort(first)]


def prune_short_raster_lines(ras, min_pts: int = 2):
    """剔除"被边界裁到只剩孤点"的栅格扫描线 —— 即 S 形轨迹末尾那截**尾巴点**。

    为什么会有: `make_raster_pass` 先在扫描线上按间距采样, 再用
    `dist_to_poly(uv_wp, rim_uv) >= EDGE_INSET_MM` 把贴边点滤掉; 靠近盆口上下两端的
    扫描线 v 跨度本来就短, 滤完可能只剩 **1 个点** —— 它构成不了一条线(没法插值、
    也没有"蛇形往返"的语义), 却会作为**最后一个尾巴点**让机械臂多跑一趟。
        · 实测 212234: n_line=5, line1 采样 2 点全被滤掉(跳过), line5(u=-151.5)
          采样 3 点只留 1 点(注: u 是盆口长轴, 长轴两端 = 盆口上下端);
        · 实测 212257: 同样末线只剩 1 点。
    做法: 按线条号**整条剔除**点数 < min_pts 的线, 并把保留下来的线号重排成 1..K
    (线条号只用于"蛇形左右交替"和渲染分组, 重排后仍连续、交替关系不变)。
    返回 (Pass, 剔除点数, 剔除线条数); 没有可剔的则原样返回。
    """
    if ras is None or ras.line is None or len(ras.pts) == 0 or int(min_pts) <= 1:
        return ras, 0, 0
    line = np.asarray(ras.line, dtype=np.float64)
    keep = np.zeros(len(ras.pts), dtype=bool)
    line_new = np.zeros(len(ras.pts), dtype=np.float64)
    n_drop_pts, n_drop_lines, k = 0, 0, 0
    for ln in np.unique(line):
        m = line == ln
        n_m = int(m.sum())
        if n_m >= int(min_pts):
            k += 1
            keep |= m
            line_new[m] = float(k)
        else:
            n_drop_pts += n_m
            n_drop_lines += 1
    if n_drop_pts == 0:
        return ras, 0, 0
    label = ras.label
    mm = re.search(r'(\d+)\s*条线', label)          # label 里的线数跟着改
    if mm:
        label = label[:mm.start(1)] + str(k) + label[mm.end(1):]
    out = Pass(key=ras.key, label=label, type=ras.type, pts=ras.pts[keep],
               line=line_new[keep], dwell_s=ras.dwell_s, standoff_mm=ras.standoff_mm)
    return out, int(n_drop_pts), int(n_drop_lines)


def run_plan(path: str, args) -> dict:
    t0 = time.time()
    log(SEP)
    log('[输入] %s' % os.path.abspath(path))
    data = load_box_cloud(path)
    hl = data['hl']
    log('[信息] 点云 %d 点：未染色 %d / hl=edge %d / hl=keypoint %d'
        % (len(data['pts']), int((hl == '').sum()), int((hl == 'edge').sum()),
           int((hl == 'keypoint').sum())))
    keep = np.ones(len(hl), dtype=bool)
    if args.drop_hl == 'all':
        keep &= hl == ''
    elif args.drop_hl == 'edge':
        keep &= hl != 'edge'
    elif args.drop_hl == 'keypoint':
        keep &= hl != 'keypoint'
    work = data['pts'][keep]
    log('[信息] 取边工作点云 %d 点（--drop-hl %s；染色点是检测叠加层，不参与取边）'
        % (len(work), args.drop_hl))

    # ---- ② 点云取边（两条：盆口边缘=红，收缩后的边缘=青）------------------
    axv = view_axis_o2e(data['meta']['T_cam_end'])
    log('[信息] 相机视线   : o2e %s（T_cam_end 第三列）' % np.round(axv, 4).tolist())
    curves = {}
    for tag, want, name in (('mouth', str(args.mouth_edge), '盆口边缘'),
                            ('path', str(args.path_edge), '收缩后边缘')):
        seed_pts = edge_points(data, want)
        if len(seed_pts) < 50:
            raise SystemExit('[错误] 找不到「%s」的检测点（hl=edge 且染色=%s）' % (name, want))
        kw: dict = {}
        if tag == 'path':
            # 收缩边（青）与盆口边缘（红）**共用同一个盆口平面**，并用红边高度当**高度上限**
            # （青在口内、不可能高过口沿；否则个别箱会吸到口沿平台上 → 个别点突然变高）
            mi = curves['mouth'][1]
            kw['plane'] = (mi['c_o2e'], mi['n_o2e'], mi['u_o2e'], mi['v_o2e'])
            kw['h_ceil_mm'] = mi['bin_h_mm']
            log('[信息] 收缩边约束 : 共用红边盆口平面 + 高度上限=红边同角度高度（+%.1fmm）'
                % float(args.rim_ceil_tol))
        pts_o2e, info = extract_rim_from_cloud(
            work, seed_pts, axv, bin_deg=float(args.rim_bin_deg),
            band_mm=float(args.rim_band_path if tag == 'path' else args.rim_band),
            r_step_mm=float(args.rim_step), min_pts_bin=int(args.rim_min_pts),
            n_iter=int(args.rim_iter), smooth_k=int(args.rim_smooth), slope_tol=0.3,
            plane_nb_mm=float(args.rim_plane_nb), r_tol_mm=float(args.rim_rtol),
            h_tol_mm=float(args.rim_htol), plat_tol_mm=float(args.rim_plat_tol),
            h_ceil_tol_mm=float(args.rim_ceil_tol), snap_mm=float(args.rim_snap), **kw)
        info['seed'] = want
        info['seed_points'] = int(len(seed_pts))
        info['label'] = name
        curves[tag] = (pts_o2e, info)
        log('[信息] 点云取边-%-6s: %s（%s：%d 点）| 自举半径 %.1f -> 角向 %d 箱（有效 %d，剔点 %d）| '
            '半径 %.1f~%.1f（中位 %.1f）| 周长 %.1f mm | 深度跨度 %.1f mm'
            % (name, '红 = 盆口边缘' if want == 'red' else '青 = 边缘收缩后',
               'hl=edge+染色' if want in ('red', 'cyan') else want, len(seed_pts),
               info['r_seed_mm'], info['n_bin'], info['n_ok_bin'], info['n_fix_bin'],
               info['r_min_mm'], info['r_max_mm'], info['r_median_mm'], info['perimeter_mm'],
               info['depth_span_mm']))
        if info.get('n_snap_bin'):
            log('                   （%d 个缺箱/剔点箱的补齐点已吸附到最近实测点云点）'
                % info['n_snap_bin'])
        log('                   逐点深度连续: 相邻步长 中位 %.2f / 最大 %.2f mm | '
            '坡度 中位 %.3f / 最大 %.3f'
            % (info['step_median_mm'], info['step_max_mm'], info['slope_median'],
               info['slope_max']))
        log('                   高度取自点云: 沿盆口面法向起伏 %.1f mm（RMS %.1f）| '
            '与实测边缘点差(%d 箱) 中位 %.2f / 最大 %.2f mm'
            % (info['relief_span_mm'], info['plane_rms_mm'], info['n_hug_bin'],
               info['hug_median_mm'], info['hug_max_mm']))
        log('                   每箱点数 中位 %.0f；窗口内高度散布(P90-P10) 中位 %.1f / 最大 %.1f mm'
            '（越小越说明窗口落在单一平面上）'
            % (info['n_hit_median'], info['spread_median_mm'], info['spread_max_mm']))
    rim_o2e, ri = curves['mouth']
    path_o2e, pi = curves['path']
    if args.dump_rim:
        base = os.path.splitext(args.dump_rim)[0]
        np.savetxt(base + '_mouth_' + str(args.mouth_edge) + '.txt', rim_o2e, fmt='%.3f',
                   header='mouth rim from cloud points (o2e, mm): x y z')
        np.savetxt(base + '_path_' + str(args.path_edge) + '.txt', path_o2e, fmt='%.3f',
                   header='shrunk path from cloud points (o2e, mm): x y z')
        log('[输出] %s_mouth_*.txt / %s_path_*.txt' % (base, base))

    # ---- 取边质量闸门（不达标直接 FAIL，避免输出看似正常的 MoveS）------------
    problems = []
    for tag in ('mouth', 'path'):
        _, info = curves[tag]
        dist = nearest_dist(work, curves[tag][0], 15.0)
        info['dist_median_mm'] = float(np.median(dist))
        info['dist_p90_mm'] = float(np.percentile(dist, 90))
        info['far_frac'] = float((dist > 10.0).mean())
        log('[信息] 贴合校验-%-6s: 取到的点 到点云最近距离 中位 %.2f / P90 %.2f mm | '
            '离点云 >10mm 的点 %.0f%% | 检测覆盖 %d/%d 角向箱 | 半径 中位 %.1f vs 检测自举 %.1f（差 %.0f%%）'
            ' | 相邻步长 最大 %.1f mm = 中位 %.1f 的 %.1f 倍（成片跳变 %.0f%%）'
            % (info['label'], info['dist_median_mm'], info['dist_p90_mm'],
               100.0 * info['far_frac'], info['n_seed_bin'], info['n_bin'],
               info['r_median_mm'], info['r_seed_mm'],
               100.0 * abs(info['r_median_mm'] - info['r_seed_mm']) / max(info['r_seed_mm'], 1e-6),
               info['step_max_mm'], info['step_median_mm'],
               info['step_max_mm'] / max(info['step_median_mm'], 1e-6),
               100.0 * info['jump_frac']))
        if info['far_frac'] > float(args.max_far_frac):
            problems.append('%s：离点云 >10mm 的点占 %.0f%%（阈值 --max-far-frac %.0f%%）——'
                            '取到的不是点云上的边缘'
                            % (info['label'], 100.0 * info['far_frac'],
                               100.0 * float(args.max_far_frac)))
        if info['n_seed_bin'] < float(args.min_seed_cover) * info['n_bin']:
            problems.append('%s：检测点只覆盖 %d/%d 角向箱（阈值 --min-seed-cover %.0f%%），'
                            '缺口全靠插值补、不可信'
                            % (info['label'], info['n_seed_bin'], info['n_bin'],
                               100.0 * float(args.min_seed_cover)))
        if abs(info['r_median_mm'] - info['r_seed_mm']) > float(args.max_r_dev) * info['r_seed_mm']:
            problems.append('%s：取到的半径中位 %.1f 与检测自举半径 %.1f 差 %.0f%%（阈值 '
                            '--max-r-dev %.0f%%）—— 半径跑偏，多半是平面/自举错了'
                            % (info['label'], info['r_median_mm'], info['r_seed_mm'],
                               100.0 * abs(info['r_median_mm'] - info['r_seed_mm'])
                               / max(info['r_seed_mm'], 1e-6), 100.0 * float(args.max_r_dev)))
        if info['jump_frac'] > float(args.max_jump_frac):
            problems.append('%s：相邻点步长 >5×中位的点占 %.0f%%（阈值 --max-jump-frac %.0f%%）'
                            '—— 曲线成片跳变（单点台阶不算）'
                            % (info['label'], 100.0 * info['jump_frac'],
                               100.0 * float(args.max_jump_frac)))
        if tag == 'mouth' and info['relief_span_mm'] > float(args.max_relief):
            problems.append('%s：沿盆口面的高度起伏 %.1f mm（阈值 --max-relief %.0f mm）'
                            '—— 口沿平台本该基本平，起伏这么大说明有箱串到碗里/别的台面了'
                            % (info['label'], info['relief_span_mm'], float(args.max_relief)))
    if problems and not args.no_fail:
        for s in problems:
            log('[错误] ' + s)
        raise SystemExit('[失败] 取边质量不达标，已中止（不输出轨迹）')
    for s in problems:
        log('[警告] ' + s + '（--no-fail：仅告警）')

    # ---- o2e -> 基座（拍照位姿固定；与生产链路同式）------------------------
    t_ph = np.asarray(args.photo_pose[:3], dtype=np.float64)
    ang_ph = np.asarray(args.photo_pose[3:], dtype=np.float64)
    r_ph = euler_to_R(float(ang_ph[0]), float(ang_ph[1]), float(ang_ph[2]))
    rim_base = o2e_to_base(rim_o2e, t_ph, ang_ph)
    path_base = o2e_to_base(path_o2e, t_ph, ang_ph)
    cloud_base = o2e_to_base(data['pts'], t_ph, ang_ph)          # 全部点（含染色点）
    log('[信息] 拍照位姿   : [%.3f, %.3f, %.3f] mm / [%.2f, %.2f, %.2f]°'
        % tuple(np.r_[t_ph, ang_ph]))


    # ---- ③ 闭合 + 盆口平面 + 弧长 -----------------------------------------
    rim_pts, gap, closed_edge = close_rim_curve(rim_base)
    pf, d_pl, a_open = plane_frame(rim_pts, robot_ref=t_ph)
    s_arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(rim_pts, axis=0), axis=1))])
    L_arc = float(s_arc[-1])
    log('[信息] 盆口平面   : 质心 [%.1f, %.1f, %.1f] mm | 法向 [%.3f, %.3f, %.3f] | '
        '面内 %.0f × %.0f mm | 面积 %.0f cm²'
        % (pf.c0[0], pf.c0[1], pf.c0[2], pf.n[0], pf.n[1], pf.n[2],
           np.ptp(pf.uv[:, 0]), np.ptp(pf.uv[:, 1]), a_open / 100.0))
    log('                 闭合后 %d 点 / 周长 %.1f mm（首尾缝 %.2f mm，%s）'
        % (len(rim_pts), L_arc, gap, '已补闭合边' if closed_edge else '首尾已相接'))
    # 收缩边（青）：唇遍沿它走；同投影到盆口面内
    path_pts, gap_p, closed_p = close_rim_curve(path_base)
    s_arc_p = np.concatenate([[0.0],
                              np.cumsum(np.linalg.norm(np.diff(path_pts, axis=0), axis=1))])
    L_p = float(s_arc_p[-1])
    uv_path = np.c_[path_pts @ pf.u - float(pf.c0 @ pf.u), path_pts @ pf.v - float(pf.c0 @ pf.v)]
    # 轨迹目标用**投影回盆口平面**的点（喷洗目标在口沿平面上；画出来的青线仍保留真实高度用来核对）：
    # 否则唇遍会跟着碗壁扎进口内，工具/夹爪间隙会掉到几 mm。
    path_planar = (pf.c0[None, :] + uv_path[:, 0:1] * pf.u[None, :]
                   + uv_path[:, 1:2] * pf.v[None, :])
    s_arc_plan = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(np.vstack([path_planar, path_planar[:1]]), axis=0), axis=1))])
    L_plan = float(s_arc_plan[-1])
    log('[信息] 收缩边(青) : 闭合后 %d 点 / 周长 %.1f mm（首尾缝 %.2f mm）| 面内等效直径 %.1f mm'
        % (len(path_pts), L_p, gap_p, 2.0 * L_p / (2.0 * math.pi)))

    # ---- ④ 喷距 / 足迹 ----------------------------------------------------
    mouth_short = float(min(np.ptp(pf.uv[:, 0]), np.ptp(pf.uv[:, 1])))
    ax_arr = euler_to_R(*[float(v) for v in args.arrival_angles])[:, 2]
    so_auto = float(abs((pf.c0 - t_ph) @ ax_arr))
    if str(args.standoff).lower() == 'auto':
        so = so_auto
        d_cap = float(args.footprint_frac) * mouth_short
        tan_h = math.tan(math.radians(SPRAY_CONE_DEG / 2.0))
        if args.footprint_frac > 0 and 2.0 * so * tan_h > d_cap:
            so = d_cap / (2.0 * tan_h)
        so_kind = 'auto（轴向 %.0f mm，按足迹 <= %.0f%%×短边 %.0f mm 收紧）' % (
            so_auto, 100.0 * args.footprint_frac, mouth_short)
    else:
        so = float(args.standoff)
        so_kind = '指定'
    D_fp, s_fp = footprint_spacing(so, SPRAY_CONE_DEG, SPRAY_OVERLAP)
    log('[信息] 喷洗距离   : standoff = %.1f mm（%s）；足迹 D = %.1f mm，间距 s = %.1f mm'
        % (so, so_kind, D_fp, s_fp))

    # ---- ⑤ 四遍航点 -------------------------------------------------------
    passes = []
    ras, n_line, n_skip = make_raster_pass(pf, s_fp, so, EDGE_INSET_MM, 'across_short',
                                           RASTER_TOP_TO_BOTTOM)
    ras, n_tail_pt, n_tail_ln = prune_short_raster_lines(ras, int(args.raster_min_pts))
    if len(ras.pts):
        passes.append(ras)
        log('[信息] 遍[栅格]   : %d 条线（跳过 %d 太窄）-> %d 点' % (n_line, n_skip, len(ras.pts)))
    if n_tail_pt:
        log('[信息] 栅格尾巴   : 剔除 %d 条不足 %d 点的扫描线（%d 点，S 形末端的孤点）-> 余 %d 条线'
            % (n_tail_ln, int(args.raster_min_pts), n_tail_pt,
               int(np.max(ras.line)) if len(ras.pts) else 0))
    if args.lip:                       # 唇遍（外唇那两半圈）：默认不生成
        for p in make_rim_passes(_UVFrame(uv_path), path_planar, s_arc_plan, L_plan, s_fp, so,
                                 RIM_SPLIT_HALVES, RIM_DWELL_S):
            passes.append(p)
            log('[信息] 遍[%-9s] : %d 点（沿收缩边=青）' % (p.key, len(p.pts)))
    ring_so = (float(args.ring_standoff) if args.ring_standoff is not None else None)
    ring_off = (None if str(args.ring_offset).lower() in ('auto', 'none')
                else float(args.ring_offset))
    ring, rd = make_ring_pass(pf, rim_pts, pf.uv, s_arc, L_arc, s_fp, D_fp, so, ring_off,
                              ring_so, CORNER_ROUND, CORNER_MAX_TURN_DEG,
                              CORNER_MIN_STEP_MM, RING_MERGE_MM, RING_TURN_SOFT_FINE,
                              RING_ROUND_MAX_IT)
    if ring is not None:
        passes.append(ring)
        log('[信息] 遍[内缩圈]   : 内缩 %.1f mm -> %d 点（越界 %d）'
            % (rd['ring_off'], rd['n_ring'], rd['n_out']))
    asm = assemble_passes(passes)
    log('[信息] 合计航点   : %d（沿口 %d + 栅格 %d）'
        % (asm['n_wp'], int((asm['wp_type'] == 1).sum()), int((asm['wp_type'] == 2).sum())))

    # ---- ⑥ 位姿 / 间隙 / 覆盖率 ------------------------------------------
    poses = compute_poses(asm['wp_pos'], asm['so_wp'], pf.n, args.arrival_angles,
                          NOZZLE_TOOL, GRIPPER_TOOL)
    ds = voxel_downsample(cloud_base, float(args.clear_voxel))
    clr = compute_clearance(poses, ds, CLEAR_WARN_MM)
    cov = compute_coverage(poses, asm, pf.uv, pf.c0, pf.u, pf.v, pf.n, SPRAY_CONE_DEG,
                           COVER_CELL_MM)
    cov_path = compute_coverage(poses, asm, uv_path, pf.c0, pf.u, pf.v, pf.n, SPRAY_CONE_DEG,
                                COVER_CELL_MM)
    dep = (asm['wp_pos'] - pf.c0) @ ax_arr
    step = np.abs(np.diff(dep))
    log('[信息] 位姿/间隙  : 喷嘴口到目标 %.1f~%.1f mm | 喷淋轴与盆口法向夹角 %.1f°'
        % (poses['d_pt'].min(), poses['d_pt'].max(), float(np.median(poses['inc_deg']))))
    log('                 工具轴线到点云（体素 %.1f mm，%d 点）最小间隙: 喷嘴 %.1f / 夹爪 %.1f mm'
        % (args.clear_voxel, len(ds), clr['min_nozzle'], clr['min_gripper']))
    if min(clr['min_nozzle'], clr['min_gripper']) < CLEAR_WARN_MM:
        log('[错误] 最小间隙 %.1f mm < %.0f mm：本帧取边/自举几何不稳（%d 个航点低于告警线）'
            % (min(clr['min_nozzle'], clr['min_gripper']), CLEAR_WARN_MM, clr['risk_points']))
        if not args.no_fail:
            raise SystemExit('[失败] 间隙不达标，已中止（不输出轨迹；可 --no-fail 或调大 --standoff）')
    log('[信息] 轨迹深度   : %.1f ~ %.1f mm（跨度 %.1f）| 相邻航点深度变化 中位 %.2f / 最大 %.2f mm'
        % (dep.min(), dep.max(), float(np.ptp(dep)), float(np.median(step)), float(step.max())))
    log('[信息] 覆盖率     : %.1f%%（栅格 %d 单元 = %.0f cm²，未覆盖 %.1f cm²）| '
        '仅沿口 %.1f%% / 仅栅格 %.1f%%'
        % (cov['pc_all'], cov['n_cell_in'], cov['n_cell_in'] * cov['a_cell_mm2'] / 100.0,
           cov['uncover_cm2'], cov['pc_rim'], cov['pc_raster']))
    log('                 其中「盆口边缘(红)多边形」%.1f%%，「收缩边(青)多边形」%.1f%%（诊断）'
        % (cov['pc_all'], cov_path['pc_all']))
    for k, (lab, pc1, pc2) in enumerate(zip(asm['pass_short'], cov['pc_pass'],
                                            cov['pc_pass_cum']), 1):
        log('    (%d) %-32s 单独 %5.1f%%   累计 %5.1f%%' % (k, lab, pc1, pc2))
    log('  总耗时 %.2f s' % (time.time() - t0))
    log(SEP)
    return {'data': data, 'rim_o2e': rim_o2e, 'rim_info': ri, 'rim_base': rim_base,
            'path_o2e': path_o2e, 'path_info': pi, 'path_pts': path_pts, 'uv_path': uv_path,
            'cov_path': cov_path, 'rim_pts': rim_pts, 'pf': pf, 'asm': asm, 'poses': poses,
            'clr': clr, 'cov': cov, 'so_mm': so, 'D_fp': D_fp, 's_fp': s_fp,
            'cloud_base': cloud_base, 't_ph': t_ph, 'ang_ph': ang_ph, 'a_open': a_open,
            'dep': dep, 'step': step}



# ===========================================================================
# ④ 输出：JSON / CSV / MoveS / PNG
# ===========================================================================

def quat_wxyz(r_mat: np.ndarray) -> List[float]:
    """旋转矩阵 -> 四元数 (w, x, y, z)，模长 1。"""
    m = np.asarray(r_mat, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    return list(q / max(float(np.linalg.norm(q)), 1e-12))


def _outdir(path: str, args) -> str:
    outdir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(path)), 'cloud_plan')
    os.makedirs(outdir, exist_ok=True)
    return outdir


def write_outputs(res: dict, path: str, args) -> List[str]:
    d, pf, asm, poses = res['data'], res['pf'], res['asm'], res['poses']
    clr, cov = res['clr'], res['cov']
    name = os.path.splitext(os.path.basename(path))[0]
    outdir = _outdir(path, args)
    quat = quat_wxyz(poses['r_arr'])
    wps = []
    for k in range(asm['n_wp']):
        wps.append({
            'idx': k + 1, 'pass': int(asm['wp_pass'][k]),
            'pass_key': asm['pass_key'][int(asm['wp_pass'][k]) - 1],
            'type': int(asm['wp_type'][k]), 'line': int(asm['wp_line'][k]),
            'spray': bool(asm['wp_type'][k] > 0),
            'dwell_s': round(float(asm['wp_dwell'][k]), 2),
            'standoff_mm': round(float(asm['so_wp'][k]), 2),
            'target_base_mm': np.round(asm['wp_pos'][k], 3).tolist(),
            'flange_base_mm': np.round(poses['p_flange'][k], 3).tolist(),
            'nozzle_tip_base_mm': np.round(poses['apex'][k], 3).tolist(),
            'gripper_tip_base_mm': np.round(poses['grip_tip'][k], 3).tolist(),
            'quat_wxyz': [round(float(v), 6) for v in quat],
            'incidence_deg': round(float(poses['inc_deg'][k]), 2),
            'depth_mm': round(float(res['dep'][k]), 3),
            'clear_nozzle_mm': round(float(clr['d_noz'][k]), 2),
            'clear_gripper_mm': round(float(clr['d_grip'][k]), 2)})
    js = {
        'generated_by': 'toilet_plan_from_cloud.py（只用点云真实点取盆口边缘）',
        'source_json': os.path.abspath(path),
        'source_timestamp': d['meta'].get('timestamp', ''),
        'unit': 'mm / deg（o2e = 拍照时刻末端法兰坐标系；base = 机器人基座系）',
        'why': '检测输出的边缘折线逐点取深度、噪声大且含误检段；盆口形状改从点云取：'
               '在检测半径带内逐角向取"最靠相机"的点，深度因此逐点连续。',
        'options': {'drop_hl': args.drop_hl, 'rim_bin_deg': float(args.rim_bin_deg),
                    'rim_band_mm': float(args.rim_band), 'rim_step_mm': float(args.rim_step),
                    'rim_iter': int(args.rim_iter), 'rim_smooth': int(args.rim_smooth),
                    'standoff_input': ('auto' if str(args.standoff).lower() == 'auto'
                                       else float(args.standoff)),
                    'standoff_used_mm': round(float(res['so_mm']), 2),
                    'footprint_mm': round(float(res['D_fp']), 2),
                    'spacing_mm': round(float(res['s_fp']), 2),
                    'cone_deg': SPRAY_CONE_DEG, 'overlap': SPRAY_OVERLAP,
                    'clear_voxel_mm': float(args.clear_voxel),
                    'photo_pose': list(map(float, args.photo_pose)),
                    'arrival_angles_deg': list(map(float, args.arrival_angles))},
        'cloud': {'n_total': int(len(d['pts'])), 'n_uncolored': int((d['hl'] == '').sum()),
                  'n_hl_edge': int((d['hl'] == 'edge').sum()),
                  'n_hl_keypoint': int((d['hl'] == 'keypoint').sum()),
                  'n_used_for_rim': int(len(res['rim_o2e']))},
        'rim_from_cloud': {'mouth_edge': str(args.mouth_edge), 'path_edge': str(args.path_edge),
                           'mouth': {k: (round(v, 3) if isinstance(v, float) else v)
                                     for k, v in res['rim_info'].items()},
                           'path': {k: (round(v, 3) if isinstance(v, float) else v)
                                    for k, v in res['path_info'].items()}},
        'mouth_o2e': np.round(res['rim_o2e'], 3).tolist(),
        'path_o2e': np.round(res['path_o2e'], 3).tolist(),
        'rim_o2e': np.round(res['rim_o2e'], 3).tolist(),
        'plane': {'c0_mm': np.round(pf.c0, 3).tolist(), 'normal': np.round(pf.n, 5).tolist(),
                  'area_cm2': round(float(res['a_open']) / 100.0, 2),
                  'span_mm': [round(float(np.ptp(pf.uv[:, 0])), 2),
                              round(float(np.ptp(pf.uv[:, 1])), 2)]},
        'tools_mm': {k: np.round(v, 3).tolist() for k, v in TCP_MM.items()},
        'target_tool': NOZZLE_TOOL, 'gripper_tool': GRIPPER_TOOL,
        'note': '只做几何/覆盖规划，未做逆解、关节限位与速度规划',
        'passes': [{'idx': k + 1, 'key': asm['pass_key'][k], 'label': asm['pass_short'][k],
                    'waypoints': int((asm['wp_pass'] == k + 1).sum()),
                    'coverage_alone_pct': round(float(cov['pc_pass'][k]), 2),
                    'coverage_cum_pct': round(float(cov['pc_pass_cum'][k]), 2)}
                   for k in range(len(asm['pass_key']))],
        'summary': {'total_waypoints': int(asm['n_wp']),
                    'spray_waypoints': int((asm['wp_type'] > 0).sum()),
                    'depth_span_mm': round(float(np.ptp(res['dep'])), 2),
                    'depth_step_median_mm': round(float(np.median(res['step'])), 3),
                    'depth_step_max_mm': round(float(res['step'].max()), 3),
                    'coverage_pct': round(float(cov['pc_all']), 2),
                    'coverage_path_polygon_pct': round(float(res['cov_path']['pc_all']), 2),
                    'uncovered_cm2': round(float(cov['uncover_cm2']), 2),
                    'min_clear_nozzle_mm': round(float(clr['min_nozzle']), 2),
                    'min_clear_gripper_mm': round(float(clr['min_gripper']), 2),
                    'risk_points': int(clr['risk_points'])},
        'waypoints': wps}
    outdir = _outdir(path, args)
    jpath = os.path.join(outdir, name + '_cloud_plan.json')
    with open(jpath, 'w', encoding='utf-8') as fp:
        json.dump(js, fp, ensure_ascii=False, indent=2)
    return jpath, wps, quat, outdir, name

    return outdir


def write_tables(res: dict, wps: List[dict], quat: List[float], outdir: str, name: str,
                 path: str, args) -> Tuple[str, str]:
    """CSV（Excel 直接打开）+ MoveS（TB6 SDK，格式与既有喷洗轨迹一致）。"""
    asm, poses, cov = res['asm'], res['poses'], res['cov']
    cpath = os.path.join(outdir, name + '_cloud_plan.csv')
    with open(cpath, 'w', encoding='utf-8-sig', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['#toilet_plan_from_cloud.py 航点表（基座系 mm；姿态 = 到达姿态四元数 wxyz）'])
        w.writerow(['idx', 'pass', 'pass_key', 'type', 'line', 'spray', 'standoff_mm',
                    'target_x', 'target_y', 'target_z', 'flange_x', 'flange_y', 'flange_z',
                    'quat_w', 'quat_x', 'quat_y', 'quat_z', 'depth_mm', 'incidence_deg',
                    'clear_nozzle_mm', 'clear_gripper_mm', 'dwell_s'])
        for it in wps:
            w.writerow([it['idx'], it['pass'], it['pass_key'], it['type'], it['line'],
                        int(it['spray']), it['standoff_mm']] + it['target_base_mm']
                       + it['flange_base_mm'] + it['quat_wxyz']
                       + [it['depth_mm'], it['incidence_deg'], it['clear_nozzle_mm'],
                          it['clear_gripper_mm'], it['dwell_s']])
    mpath = os.path.join(outdir, name + '_cloud_plan_moves.txt')
    with open(mpath, 'w', encoding='utf-8', newline='\n') as fp:
        fp.write('# toilet_plan_from_cloud.py 生成的喷洗轨迹（TB6 SDK MoveS 指令）\n')
        fp.write('# robottarget_value = {x,y,z,qw,qx,qy,qz}；位置单位 m，姿态为到达姿态四元数\n')
        fp.write('# 来源: %s | 取边: 点云（--drop-hl %s）| 航点 %d | 覆盖率 %.1f%%\n'
                 % (os.path.abspath(path), args.drop_hl, asm['n_wp'], cov['pc_all']))
        fp.write('# 喷洗点（type>0）开喷；过渡/起终点关\n\n')
        fp.write('{Clear}\n{Disable}\n{Recover}\n{Mode}\n{Enable}\n')
        fp.write('{SetUsingSP --state=on}\n{Var --clear}\n{Start}\n')
        fp.write('{MoveS --type=first_insert}\n')
        for k in range(asm['n_wp']):
            p = poses['p_flange'][k] / 1000.0
            tag = '' if asm['wp_type'][k] > 0 else '（喷洗关）'
            fp.write('{MoveS --type=insert --robottarget_value={%.6f, %.6f, %.6f, %.6f, %.6f, '
                     '%.6f, %.6f}}   # %d %s%s\n'
                     % (p[0], p[1], p[2], quat[0], quat[1], quat[2], quat[3], k + 1,
                        asm['pass_key'][int(asm['wp_pass'][k]) - 1], tag))
        fp.write('{MoveS --type=insert --stop}\n')
    return cpath, mpath


def render_png(res: dict, out_png: str, args) -> Optional[str]:
    """open3d 离屏出图：点云（体素抽稀）+ 取到的边缘线 + 四遍轨迹。"""
    try:
        import open3d as o3d
    except Exception as exc:                                    # 没装 open3d 就跳过出图
        log('[提示] 跳过 PNG（open3d 不可用：%s）' % exc)
        return None
    asm, poses = res['asm'], res['poses']
    cloud = voxel_downsample(res['cloud_base'], 3.0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([0.62, 0.66, 0.72], (len(cloud), 1)))

    def line(pts, color):
        pts = np.asarray(pts, dtype=np.float64)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(
            np.c_[np.arange(len(pts) - 1), np.arange(1, len(pts))].astype(np.int32))
        ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts) - 1, 1)))
        return ls

    geoms = [pcd, line(np.vstack([res['rim_pts'], res['rim_pts'][:1]]), [0.05, 0.9, 0.15])]
    palette = {1: [1.0, 0.35, 0.1], 2: [0.15, 0.55, 1.0]}
    for p in range(1, len(asm['pass_key']) + 1):
        idx = np.flatnonzero(asm['wp_pass'] == p)
        if len(idx) < 2:
            continue
        geoms.append(line(poses['apex'][idx],
                          palette.get(int(asm['wp_type'][idx[0]]), [1.0, 1.0, 0.0])))
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=int(args.size), height=int(args.size))
    for g in geoms:
        vis.add_geometry(g)
    vis.get_render_option().background_color = np.array([0.08, 0.08, 0.10])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=80.0)
    frame.translate(res['pf'].c0)
    vis.add_geometry(frame)
    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_png)
    vis.destroy_window()
    log('[输出] %s' % out_png)
    return out_png




def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='只用点云真实点取盆口边缘、再做四遍喷淋规划（检测边线只用于自举）')
    p.add_argument('json', help='box_cloud_*.json（RgbDepth 按 P 键导出）')
    p.add_argument('--drop-hl', choices=('all', 'edge', 'keypoint', 'none'), default='all',
                   help='取边点云丢掉哪些染色点：all（默认，丢全部染色点 = 隐藏高亮后的真实点）/'
                        'edge（只丢边缘线像素）/ keypoint（只丢蓝点）/ none（全留）')
    p.add_argument('--mouth-edge', choices=('red', 'cyan'), default='red',
                   help='**盆口边缘**用哪条检测点：red（默认）= 红 = mask 分割边缘（真实口沿，定平面/'
                        '多边形/覆盖率）；cyan = 青 = 收缩后边缘')
    p.add_argument('--path-edge', choices=('red', 'cyan'), default='cyan',
                   help='**唇遍轨迹**沿哪条走：cyan（默认）= 青 = 边缘收缩后的点（口内一点，喷锥不甩'
                        '到外面）；red = 沿真实口沿')
    p.add_argument('--rim-bin-deg', type=float, default=4.0, metavar='DEG',
                   help='取边角向箱宽，默认 4°')
    p.add_argument('--rim-smooth', type=int, default=1, metavar='K',
                   help='缺箱/被剔箱的补齐值用环形中值半窗 K（实时点直接输出、不参与平滑），默认 1')
    p.add_argument('--rim-band', type=float, default=20.0, metavar='MM',
                   help='盆口边缘(红)的搜索带半宽（带内先找口沿平台，再取平台上最接近检测半径的点），默认 ±20mm')
    p.add_argument('--rim-band-path', type=float, default=8.0, metavar='MM',
                   help='收缩边(青)的搜索带半宽：青线在碗内壁上，带要窄才不会把口沿平台带进来，默认 ±8mm')
    p.add_argument('--rim-plat-tol', type=float, default=3.0, metavar='MM',
                   help='口沿平台厚度容差：高度直方图峰值 ±该值 内的点算平台点，默认 3mm')
    p.add_argument('--rim-ceil-tol', type=float, default=8.0, metavar='MM',
                   help='收缩边的高度上限裕量（上限 = 红边同角度高度 + 该值）：口沿内唇可能比红边高几 mm，'
                        '所以留 8mm，默认 8')
    p.add_argument('--rim-snap', type=float, default=12.0, metavar='MM',
                   help='缺箱/剔点箱的补齐点吸附到最近实测点云点的搜索半径，默认 12mm')
    p.add_argument('--rim-step', type=float, default=2.0, metavar='MM',
                   help='取边半径分桶步长（预留），默认 2mm')
    p.add_argument('--rim-min-pts', type=int, default=20, metavar='N',
                   help='每角向箱最少点数，默认 20')
    p.add_argument('--rim-iter', type=int, default=2, metavar='N',
                   help='用取到的边缘点重拟合盆口面的轮数，默认 2')
    p.add_argument('--rim-plane-nb', type=float, default=25.0, metavar='MM',
                   help='在点云里取检测点附近该半径内的点来求口沿面法向（中心仍用检测环质心），默认 25mm')
    p.add_argument('--rim-rtol', type=float, default=8.0, metavar='MM',
                   help='环形剔点：半径与邻域中值差 > 该值视为串台面，用邻域中值替换，默认 8mm')
    p.add_argument('--rim-htol', type=float, default=6.0, metavar='MM',
                   help='环形剔点：高度与邻域中值差 > 该值视为串台面，用邻域中值替换，默认 6mm')
    p.add_argument('--min-seed-cover', type=float, default=0.70, metavar='F',
                   help='检测点角向覆盖率下限（有 >=3 个检测点的箱占比），低于它判 FAIL，默认 0.70')
    p.add_argument('--max-far-frac', type=float, default=0.10, metavar='F',
                   help='取到的曲线里"离点云 >10mm"的点占比上限，超过判 FAIL，默认 0.10')
    p.add_argument('--max-r-dev', type=float, default=0.20, metavar='F',
                   help='取到的半径中位与检测自举半径的相对偏差上限，超过判 FAIL，默认 0.20')
    p.add_argument('--max-jump-frac', type=float, default=0.10, metavar='F',
                   help='相邻点步长 >5×中位 的点占比上限（成片跳变判据），超过判 FAIL，默认 0.10')
    p.add_argument('--max-relief', type=float, default=25.0, metavar='MM',
                   help='盆口边缘(红)线沿盆口面的高度起伏上限，超过判 FAIL，默认 25mm')
    p.add_argument('--no-fail', action='store_true',
                   help='质量不达标只告警，不中止（默认中止、不输出 MoveS）')
    p.add_argument('--dump-rim', default=None, metavar='PATH',
                   help='把取到的边缘点（o2e mm）另存文本，便于核对')
    p.add_argument('--standoff', default='auto', metavar='auto|MM',
                   help='喷洗距离：auto（默认）或固定值（生产链路喷嘴 96.2mm）')
    p.add_argument('--footprint-frac', type=float, default=0.35, metavar='F',
                   help='auto 时"足迹 <= F × 口短边"上限，默认 0.35；0 = 不限')
    p.add_argument('--clear-voxel', type=float, default=2.0, metavar='MM',
                   help='间隙检查体素抽稀，默认 2mm（0 = 全点，慢）')
    p.add_argument('--lip', action='store_true',
                   help='保留唇遍（rim_left/rim_right，沿收缩边的外唇两半圈）；**默认不生成**')
    p.add_argument('--ring-offset', default='40', metavar='auto|MM',
                   help='内缩圈沿内法向的内缩量，越小内圆越大；默认 40mm（auto = 足迹半径 D_fp/2）')
    p.add_argument('--raster-min-pts', type=int, default=2, metavar='N',
                   help='栅格每条扫描线的最少点数：不足则**整条剔除**（默认 2）——被内缩裁到只剩'
                        '孤点的扫描线（S 形轨迹末尾的"尾巴点"）会被去掉；1 = 不剔')
    p.add_argument('--ring-standoff', type=float, default=None, metavar='MM',
                   help='内缩圈单独喷距；默认 None = 与其它遍相同（生产 urinal 链路用 108.2mm）')
    p.add_argument('--photo-pose', type=float, nargs=6, metavar=('X', 'Y', 'Z', 'RX', 'RY', 'RZ'),
                   default=[float(v) for v in np.r_[CAPTURE_T_EE_MM, CAPTURE_ANGLES_DEG]],
                   help='拍照位姿（mm/deg，Z-Y-X 欧拉），默认标定值')
    p.add_argument('--arrival-angles', type=float, nargs=3, metavar=('RX', 'RY', 'RZ'),
                   default=[float(v) for v in ARRIVAL_ANGLES],
                   help='到达姿态（deg），默认与生产喷淋链路一致')
    p.add_argument('--out-dir', default=None, help='输出目录，默认 <json 同目录>/cloud_plan')
    p.add_argument('--png', action='store_true', help='额外用 open3d 离屏出一张 iso 图')
    p.add_argument('--size', type=int, default=1280, help='出图边长，默认 1280')
    p.add_argument('--no-save', action='store_true', help='不写文件（只打印指标）')
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    res = run_plan(args.json, args)
    if not args.no_save:
        jpath, wps, quat, outdir, name = write_outputs(res, args.json, args)
        log('[输出] %s' % jpath)
        cpath, mpath = write_tables(res, wps, quat, outdir, name, args.json, args)
        log('[输出] %s' % cpath)
        log('[输出] %s' % mpath)
        if args.png:
            render_png(res, os.path.join(outdir, name + '_cloud_plan_iso.png'), args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
