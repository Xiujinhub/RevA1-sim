# -*- coding: utf-8 -*-
"""
喷淋轨迹规划移植 (MATLAB: matlab/plot_spray_path_o2e.m) —— Python 参照实现 (MATLAB -> Python -> C++)

【链路与 MATLAB 段号对应】
    ① 读 o2e JSON                      -> (由调用方/run_spray_plan.py 负责)
    ② o2e -> 基座系(拍照位姿 R/T)      -> :202-212      o2e_to_base()
    ③ 沿口清洗(唇)                     -> clean_rim_contour.py
                                          ⚠ 必须同口径: plane_fit(fix_signs=True) 定型 n/u,
                                            且输出做"起点/绕向规范化"(见该文件与 MATLAB 说明)
    ④ 沿口闭合 + 盆口平面 + 面内坐标   -> :300-355      close_rim_curve() + plane_frame()
    ⑤ 足迹 / 间距                      -> :357-361      footprint_spacing()
    ⑥ 四遍航点: 栅格 / 唇左右 / 内缩圈 -> :366-543      make_raster_pass() / make_rim_passes()
                                                          / make_ring_pass()
    ⑦ 按 PASS_ORDER 组装               -> :545-570      assemble_passes()
    ⑧ 法兰位姿 + 工具线段 + 干涉判据   -> :571-599      compute_poses() / compute_clearance()
    ⑨ 覆盖率                           -> :600-660      compute_coverage()
    ⑩ 导出航点表 / 对账文件            -> run_spray_plan.py

【2026-09-19 同步进来的两个新口径】(与 MATLAB 侧同一次修复)
    · 面内基定型: n 朝机器人(robot_ref) + u 朝基座 +Z —— 否则"起点/绕向"会随 SVD 特征向量的
      任意符号翻转(现象: ring 起点跑到另一端, 看着像上下颠倒);
    · ring 尖角插值圆滑: 用 resample_by_turn() 替换等间距重采样(唇半圈 / 蛇形栅格**不动**),
      原因与实测见 resample_by_turn.py 文件头。

【风格】与 spray_common.py 一致: 显式 for 循环 + dataclass(<=> C++ struct), 不把流程藏进广播。
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from spray_common import (ARRIVAL_ANGLES, CAPTURE_ANGLES_DEG, CAPTURE_T_EE_MM, CLEAR_WARN_MM,
                          COVER_CELL_MM, EDGE_INSET_MM, GRIPPER_TOOL, NOZZLE_TOOL, PASS_ORDER,
                          RASTER_DWELL_S, RASTER_TOP_TO_BOTTOM, RIM_DWELL_S, RIM_SPLIT_HALVES,
                          RING_OFFSET_MM, RING_STANDOFF_MM, SPRAY_CONE_DEG, SPRAY_OVERLAP,
                          TCP_MM, coverage_mask, dist_to_poly, euler_to_R, footprint_on_plane,
                          in_polygon_mask, plane_fit, poly_area, poly_scanline,
                          resample_arclen, round_matlab, seg_point_dist)
from spray_common import PlaneFrame
from resample_by_turn import poly_turns, resample_by_turn


# ==================== 数据结构 (<=> MATLAB passes(k) 结构体 / C++ struct) ====================
@dataclass
class Pass:
    """一遍航点 (MATLAB: passes(k) = struct('key','label','type','line','dwell','so','pts'))"""
    key: str
    label: str
    type: int                            # 1 = 唇/环(轮廓类, 可插值圆滑)  2 = 栅格(直线蛇形, 不动)
    pts: np.ndarray                      # (n,3) 目标点(基座系, mm)
    line: Optional[np.ndarray] = None    # 栅格线条号(栅格遍有效)
    dwell_s: float = 0.0
    standoff_mm: float = 0.0


@dataclass
class Plan:
    """规划结果 (对应 MATLAB 工作区 passes / wp_pos / Pk / ... 的全套量)"""
    passes: List[Pass] = field(default_factory=list)
    wp_pos: np.ndarray = None            # (n_wp,3) 目标点
    wp_pass: np.ndarray = None           # (n_wp,)  所属遍序号(1 基, 按 PASS_ORDER)
    wp_pass_key: List[str] = field(default_factory=list)
    wp_type: np.ndarray = None
    wp_line: np.ndarray = None
    wp_dwell: np.ndarray = None
    wp_standoff: np.ndarray = None
    # 平面/几何摘要(供对账打印)
    c0: np.ndarray = None
    n_pl: np.ndarray = None
    u_long: np.ndarray = None
    v_short: np.ndarray = None
    D_fp: float = 0.0
    s_fp: float = 0.0
    rim_pts: np.ndarray = None           # 闭合后的沿口(清洗后)
    rim_uv: np.ndarray = None
    A_open: float = 0.0


# ==================== ② o2e -> 基座系 ====================
def o2e_to_base(pts_o2e: np.ndarray, pose_xyz_mm: Optional[np.ndarray] = None,
                pose_angles_deg: Optional[np.ndarray] = None) -> np.ndarray:
    """P_base = R_photo · P_o2e + T_EE   (MATLAB :203-212)

    pose 缺省 = 标定拍照位姿(hand_eye_config.CAPTURE_*, 与 MATLAB 逐位一致);
    生产链路(arm_driver /plan_spray)用的是**实时位姿**, 对账时显式传入即可。
    C++: 同式, R = Rz·Ry·Rx (spray_o2e::eulerToR)。
    """
    p = np.asarray(pts_o2e, dtype=float)
    t_ee = CAPTURE_T_EE_MM if pose_xyz_mm is None else np.asarray(pose_xyz_mm, float).ravel()
    ang = CAPTURE_ANGLES_DEG if pose_angles_deg is None else np.asarray(pose_angles_deg, float).ravel()
    r_photo = euler_to_R(float(ang[0]), float(ang[1]), float(ang[2]))
    return (r_photo @ p.T).T + t_ee


# ==================== ④ 沿口闭合 + 盆口平面 + 面内坐标 ====================
def close_rim_curve(rim_base: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """沿口首尾相接: 缝 > 3×相邻间距中位数 -> 补一条闭合边 (MATLAB :288-299)

    返回 (rim_pts, 首尾缝 gap, 是否补了闭合边)。
    """
    p = np.asarray(rim_base, dtype=float)
    dseg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    gap = float(np.linalg.norm(p[0] - p[-1]))
    closed_gap = False
    if gap > 3.0 * float(np.median(dseg)):
        p = np.vstack([p, p[:1]])
        closed_gap = True
    return p, gap, closed_gap


def plane_frame(rim_pts: np.ndarray, robot_ref: Optional[np.ndarray] = None):
    """盆口平面最小二乘 + 面内坐标(**定型**: n 朝机器人 + u 朝基座 +Z) (MATLAB :313-349)

    返回 (PlaneFrame, d_pl, A_open)。robot_ref 缺省 = 标定拍照位姿(等价 MATLAB 的 T_EE_MM)。
    ⚠ "定型"是 2026-09-19 的新口径, 必须与 clean_rim_contour 同口径, 否则重建沿口的
      "起点/绕向"会随 SVD 特征向量的任意符号翻转(现象: ring 起点跑到另一端)。
    """
    p = np.asarray(rim_pts, dtype=float)
    t_ee = CAPTURE_T_EE_MM if robot_ref is None else np.asarray(robot_ref, float).ravel()
    c0 = p.mean(axis=0)
    _, _, vh = np.linalg.svd(p - c0, full_matrices=False)   # vh 的"行" = 右奇异向量(= MATLAB V 的列)
    n_pl = vh[2].copy()
    u_long = vh[0].copy()
    if float(n_pl @ (t_ee - c0)) < 0.0:
        n_pl = -n_pl                                        # 法向朝机器人(盆口外侧)
    if u_long[2] < 0.0:
        u_long = -u_long                                    # 长轴朝上(基座系 +Z)
    v_short = np.cross(n_pl, u_long)
    v_short = v_short / np.linalg.norm(v_short)             # (u, v, n) 右手系
    uv = np.c_[p @ u_long - float(c0 @ u_long), p @ v_short - float(c0 @ v_short)]
    uc, vc = float(uv[:, 0].mean()), float(uv[:, 1].mean())
    ang = np.arctan2(uv[:, 1] - vc, uv[:, 0] - uc)
    rad = np.hypot(uv[:, 0] - uc, uv[:, 1] - vc)
    d_pl = p @ n_pl - float(c0 @ n_pl)
    pf = PlaneFrame(c0=c0, n=n_pl, u=u_long, v=v_short, uc=uc, vc=vc,
                    uv=uv, ang=ang, rad=rad, d_pl=d_pl)
    return pf, d_pl, poly_area(uv)


def footprint_spacing(standoff_mm: float, cone_deg: float = SPRAY_CONE_DEG,
                      overlap: float = SPRAY_OVERLAP) -> Tuple[float, float]:
    """足迹 D = 2·standoff·tan(锥角/2); 间距 s = D·(1-重叠) (MATLAB :358-359)"""
    d_fp = 2.0 * standoff_mm * math.tan(math.radians(cone_deg / 2.0))
    return d_fp, d_fp * (1.0 - overlap)


# ==================== ⑥ 四遍航点 ====================
def make_rim_passes(pf: PlaneFrame, rim_pts: np.ndarray, s_arc: np.ndarray, L_arc: float,
                    s_fp: float, standoff_mm: float, split_halves: bool = RIM_SPLIT_HALVES,
                    dwell_s: float = RIM_DWELL_S) -> List[Pass]:
    """遍"沿口唇": 左/右半圈(各自 最上点->最下点) 或 整圈 (MATLAB :430-477)

    ⚠ 半圈被**强制重排成"最上点->最下点"**, 所以"起点/绕向"与沿口原始顺序无关
      (这也是为什么沿口顺序反了时, 只有 ring 会露馅); 本遍**不做**尖角插值圆滑。
    """
    uv = pf.uv
    out: List[Pass] = []
    if split_halves:
        i_top = int(np.argmax(uv[:, 0]))
        i_bot = int(np.argmin(uv[:, 0]))
        if i_top == i_bot:
            split_halves = False
        else:
            nc = len(rim_pts)
            if i_top <= i_bot:
                idx_a = np.arange(i_top, i_bot + 1)
                idx_b = np.concatenate([np.arange(i_bot, nc), np.arange(0, i_top + 1)])
            else:
                idx_a = np.concatenate([np.arange(i_top, nc), np.arange(0, i_bot + 1)])
                idx_b = np.arange(i_bot, i_top + 1)
            pa = rim_pts[idx_a]
            pb = rim_pts[idx_b][::-1]                       # 翻转 -> 也是 最上点->最下点
            if float(uv[idx_a, 1].mean()) < float(uv[idx_b, 1].mean()):
                pa, pb = pb, pa                             # PA = 左半(v>0, 机器人左手侧)
            halves = ((pa, "rim_left", "唇口左半圈 (v>0, 上->下)"),
                      (pb, "rim_right", "唇口右半圈 (v<0, 上->下)"))
            for ph, key_h, lab_h in halves:
                dh = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ph, axis=0), axis=1))])
                if dh[-1] < 3.0 * s_fp:
                    continue
                nh = max(3, round_matlab(dh[-1] / s_fp))
                ph_s = resample_arclen(ph, dh, (np.arange(1, nh + 1) - 0.5) * dh[-1] / nh)
                out.append(Pass(key=key_h, label=lab_h, type=1, pts=ph_s,
                                line=np.zeros(len(ph_s)), dwell_s=dwell_s,
                                standoff_mm=standoff_mm))
    if not split_halves:
        n_rim = max(8, int(math.ceil(L_arc / s_fp)))
        p_rim = resample_arclen(rim_pts, s_arc, (np.arange(1, n_rim + 1) - 0.5) * L_arc / n_rim)
        out.append(Pass(key="rim", label="沿口唇整圈", type=1, pts=p_rim,
                        line=np.zeros(n_rim), dwell_s=dwell_s, standoff_mm=standoff_mm))
    return out


def make_raster_pass(pf: PlaneFrame, s_fp: float, standoff_mm: float,
                     edge_inset_mm: float = EDGE_INSET_MM, mode: str = "across_short",
                     top_to_bottom: bool = RASTER_TOP_TO_BOTTOM,
                     dwell_s: float = RASTER_DWELL_S) -> Tuple[Pass, int, int]:
    """遍"栅格": 盆口面内蛇形 (MATLAB :366-427)

    mode='across_short'(默认): 栅格线沿**短轴**(v, 水平), 沿长轴(u)阶梯下降(上->下);
    mode='across_long'       : 栅格线沿**长轴**(u), 沿短轴(v)阶梯。
    ⚠ 栅格是直线蛇形, **不做**尖角插值圆滑(线端 ~180° 折返 move_blend 实测可正常执行)。
    """
    uv = pf.uv
    u_lo, u_hi = float(uv[:, 0].min()), float(uv[:, 0].max())
    v_lo, v_hi = float(uv[:, 1].min()), float(uv[:, 1].max())
    uc0, vc0 = (u_lo + u_hi) / 2.0, (v_lo + v_hi) / 2.0
    span_stack = (v_hi - v_lo) if mode == "across_long" else (u_hi - u_lo)
    n_line = max(1, int(math.ceil(span_stack / s_fp)))
    offs = (np.arange(1, n_line + 1, dtype=float) - (n_line + 1) / 2.0) * s_fp
    if top_to_bottom:
        offs = offs[::-1]
    pts_list, line_list = [], []
    n_skip, k_line = 0, 0
    for ii in range(n_line):
        if mode == "across_long":
            i_var, i_fix, fix_val = 0, 1, vc0 + offs[ii]
        else:
            i_var, i_fix, fix_val = 1, 0, uc0 + offs[ii]
        # MATLAB: poly_scanline(rim_uv(:, [i_fix, i_var]), fix_val) -> 沿"阶梯轴"扫描求交
        cr = poly_scanline(uv[:, [i_fix, i_var]], fix_val)
        if cr.size < 2:
            n_skip += 1
            continue
        lo = float(cr.min()) + edge_inset_mm
        hi = float(cr.max()) - edge_inset_mm
        if hi - lo < 0.5 * s_fp:
            n_skip += 1
            continue
        nw = max(2, round_matlab((hi - lo) / s_fp) + 1)
        steps = np.linspace(lo, hi, nw)
        if (k_line % 2) == 1:
            steps = steps[::-1]                     # 蛇形
        uv_wp = np.zeros((nw, 2))
        uv_wp[:, i_fix] = fix_val
        uv_wp[:, i_var] = steps
        keep = dist_to_poly(uv_wp, uv) >= edge_inset_mm
        nk = int(keep.sum())
        if nk == 0:
            n_skip += 1
            continue
        p_line = (pf.c0[None, :] + uv_wp[keep, 0][:, None] * pf.u[None, :]
                  + uv_wp[keep, 1][:, None] * pf.v[None, :])
        pts_list.append(p_line)
        line_list.append(np.full(nk, float(k_line + 1)))
        k_line += 1
    pts = np.vstack(pts_list) if pts_list else np.zeros((0, 3))
    lines = np.concatenate(line_list) if line_list else np.zeros(0)
    dir_txt = "上->下" if top_to_bottom else "下->上"
    return Pass(key="raster", label="栅格 (%s, %d 条线蛇形)" % (dir_txt, k_line), type=2,
                pts=pts, line=lines, dwell_s=dwell_s, standoff_mm=standoff_mm), n_line, n_skip


def make_ring_pass(pf: PlaneFrame, rim_pts: np.ndarray, rim_uv: np.ndarray, s_arc: np.ndarray,
                   L_arc: float, s_fp: float, D_fp: float, standoff_mm: float,
                   ring_offset_mm: Optional[float] = RING_OFFSET_MM,
                   ring_standoff_mm: Optional[float] = RING_STANDOFF_MM,
                   corner_round: bool = True, corner_max_turn_deg: float = 15.0,
                   corner_min_step_mm: float = 5.0,
                   merge_mm: float = 2.0, turn_soft_fine: float = 12.0, max_round_it: int = 200,
                   dwell_s: float = RIM_DWELL_S):
    """遍"内缩一圈"(contour-parallel): 唇线去噪后沿内法向偏移, 补"唇"与"栅格"之间的环带
       (MATLAB :479-543)

    流程与 MATLAB 逐行对应:
      ① 唇线重采样到 ~6mm + 5 点环形滑动平均(去噪; 直接用密集原始点求内法向会有锯齿)
      ② 逐点沿内法向偏移 ring_off; "哪侧是内"用**候选点是否落在盆口多边形内**判断
         (退化时沿用上一个点, 与 MATLAB 同)
      ③ 回到 3D; 用 resample_by_turn 做**曲率自适应重采样**(直线段按 s_fp, 圆角处自动加密)
         —— 2026-09-19 新口径: 等间距重采样会把紧圆角切成 60°~71° 折点 -> -505;
            该函数落点全在原曲线上(几何零改动), 并给出"每步折角"统计。
    返回 (Pass 或 None, 诊断 dict)。
    """
    c0, u_long, v_short = pf.c0, pf.u, pf.v
    ring_off = (D_fp / 2.0) if ring_offset_mm is None else float(ring_offset_mm)
    ring_so = standoff_mm if ring_standoff_mm is None else float(ring_standoff_mm)

    ncr = len(rim_uv)
    if ncr < 3 or len(rim_pts) < 2:
        return None, {}
    n_sm = min(ncr, max(60, round_matlab(L_arc / 6.0)))
    p_sm = resample_arclen(rim_pts, s_arc, (np.arange(1, n_sm + 1) - 0.5) * L_arc / n_sm)
    uv_sm = np.c_[p_sm @ u_long - float(c0 @ u_long), p_sm @ v_short - float(c0 @ v_short)]

    k_ma = 2
    uv_ma = np.zeros_like(uv_sm)
    for i in range(n_sm):
        idx = np.arange(i - k_ma, i + k_ma + 1) % n_sm      # MATLAB: mod((i-k_ma-1):(i+k_ma-1), n_sm)+1
        uv_ma[i, :] = uv_sm[idx, :].mean(axis=0)

    uv_off = np.zeros_like(uv_ma)
    ctr = uv_ma.mean(axis=0)
    rel = uv_ma[0] - ctr
    if float(np.linalg.norm(rel)) < 1e-12:
        rel = np.array([1.0, 0.0])
    uv_last = uv_ma[0] + ring_off * rel / float(np.linalg.norm(rel))
    n_retreat, n_radial = 0, 0
    idx_radial: List[int] = []
    for i in range(n_sm):
        ip = (i + 1) % n_sm
        im = (i - 1) % n_sm
        tv = uv_ma[ip] - uv_ma[im]
        if float(np.linalg.norm(tv)) < 1e-9:
            uv_off[i] = uv_last
            continue
        tv = tv / float(np.linalg.norm(tv))
        nrm = np.array([tv[1], -tv[0]])
        # ---- ① 逐步内缩重试: 全距 -> 半距 -> 1/4 距 ----
        #   盆口"窄尖端"处, 按全距沿法向内推会越过对面墙 => 两侧候选都不在多边形内;
        #   原版此时沿用上一个点(uv_last), 会制造 0.32mm 重复点 + 42.6° 折角(实测;
        #   MATLAB/Python 同源都有), 并把 resample_by_turn 的"粒度下限"顶高, ring 折角降不下来。
        done = False
        for frac in (1.0, 0.5, 0.25):
            off = ring_off * frac
            p1 = uv_ma[i] + off * nrm
            p2 = uv_ma[i] - off * nrm
            if bool(in_polygon_mask(p1[None, :], rim_uv)[0]):
                uv_off[i] = p1
                done = True
                break
            if bool(in_polygon_mask(p2[None, :], rim_uv)[0]):
                uv_off[i] = p2
                done = True
                break
        # ---- ② 仍失败 => 朝形心方向径向内缩(窄尖端最稳妥), 不再沿用上一点 ----
        if not done:
            d = uv_ma[i] - ctr
            nd = float(np.linalg.norm(d))
            uv_off[i] = (uv_ma[i] - ring_off * (d / nd)) if nd > 1e-9 else uv_last
            n_radial += 1
            idx_radial.append(i)
        elif frac < 1.0:
            n_retreat += 1
        uv_last = uv_off[i]

    # ---- ③ 内缩"塌陷"清理 + 局部圆角 (2026-09-19, 修 ring 折角降不到目标值的根因) ----
    #   凸尖角处, 内缩量 ring_off 一旦**超过局部曲率半径**, 内缩曲线会在该处**塌陷成尖点**
    #   (点堆积 / 自交): 实测下护裙尖角处最短边仅 0.32mm、折角 42.6°(MATLAB/Python 同源都有;
    #   已排除兜底分支 —— 兜底触发 0 次)。塌陷会把 resample_by_turn 的"粒度下限"顶高,
    #   于是 ring 折角只能降到 ~37° 而到不了目标。
    #   处理(只动塌陷处几个点, 位置改动 <1mm):
    #     (a) 合并近重复点(阈值 merge_mm);
    #     (b) 对折角超阈值处做**局部环形平滑**(与 clean_rim_contour 的"局部圆角"同一思路),
    #         迭代到折角 <= turn_soft_fine 或达上限。
    uv_off, n_merged = _merge_close_cyclic(uv_off, merge_mm)
    uv_off, n_round, turn_fine_0, turn_fine_1 = _smooth_hot_corners(uv_off, turn_soft_fine,
                                                                     max_round_it)

    p_raw = (c0[None, :] + uv_off[:, 0][:, None] * u_long[None, :]
             + uv_off[:, 1][:, None] * v_short[None, :])
    d_ring = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p_raw, axis=0), axis=1))])
    L_ring = float(d_ring[-1])
    if L_ring <= 3.0 * s_fp:
        return None, {"L_ring": L_ring}
    n_ring_old = max(8, round_matlab(L_ring / s_fp))
    ci_cor = None
    if corner_round:
        p_ring, ci_cor = resample_by_turn(p_raw, s_fp, corner_max_turn_deg,
                                          min_step_mm=corner_min_step_mm, closed=True)
    else:
        p_ring = resample_arclen(p_raw, d_ring,
                                 (np.arange(1, n_ring_old + 1) - 0.5) * L_ring / n_ring_old)
    n_ring = len(p_ring)
    uv_chk = np.c_[p_ring @ u_long - float(c0 @ u_long), p_ring @ v_short - float(c0 @ v_short)]
    n_out = int((~in_polygon_mask(uv_chk, rim_uv)).sum())
    ps = Pass(key="ring", label="边界内缩一圈 (内缩 %.0f mm, standoff %.1f mm)" % (ring_off, ring_so),
              type=1, pts=p_ring, line=np.zeros(n_ring), dwell_s=dwell_s, standoff_mm=ring_so)
    return ps, {"L_ring": L_ring, "n_ring_old": n_ring_old, "n_ring": n_ring,
                "n_out": n_out, "corner": ci_cor, "ring_off": ring_off, "ring_so": ring_so,
                "p_raw": p_raw, "n_retreat": n_retreat, "n_radial": n_radial,
                "idx_radial": idx_radial, "n_merged": n_merged, "n_round": n_round,
                "turn_fine_0": turn_fine_0, "turn_fine_1": turn_fine_1}


# ==================== ⑦ 按 PASS_ORDER 组装执行顺序 (MATLAB :604-640) ====================
def assemble_passes(passes: List[Pass], pass_order: Optional[List[str]] = None) -> dict:
    """按 PASS_ORDER 组装执行顺序; 返回各航点数组(与 MATLAB 的 wp_pos/wp_type/wp_line/
    wp_dwell/wp_pass/so_wp 同名同义) + 逐遍摘要文本

    ⚠ 与 MATLAB 一致: 找不到的 key **跳过不报错**; 遍序号(1 基)按组装顺序排;
      栅格遍沿用其线条号(用于绘图/对账)。
    """
    order = PASS_ORDER if pass_order is None else list(pass_order)
    a = dict(wp_pos=np.zeros((0, 3)), wp_type=np.zeros(0), wp_line=np.zeros(0),
             wp_dwell=np.zeros(0), wp_pass=np.zeros(0), so_wp=np.zeros(0))
    pass_txt: List[str] = []
    pass_short: List[str] = []
    pass_key: List[str] = []
    for key in order:
        hit = [ps for ps in passes if ps.key.lower() == str(key).lower()]
        if not hit:
            continue                                    # MATLAB: warning + 跳过
        p = hit[0]
        n0 = len(a["wp_pos"])
        np_ = len(p.pts)
        a["wp_pos"] = np.vstack([a["wp_pos"], np.asarray(p.pts, float)])
        a["wp_type"] = np.concatenate([a["wp_type"], np.full(np_, float(p.type))])
        a["wp_line"] = np.concatenate([a["wp_line"], np.asarray(p.line, float).ravel()])
        a["wp_dwell"] = np.concatenate([a["wp_dwell"], np.full(np_, float(p.dwell_s))])
        a["wp_pass"] = np.concatenate([a["wp_pass"], np.full(np_, float(len(pass_txt) + 1))])
        a["so_wp"] = np.concatenate([a["so_wp"], np.full(np_, float(p.standoff_mm))])
        pass_key.append(p.key)
        pass_short.append(p.label)
        pass_txt.append("(%d) %s: 航点 %d~%d, 共 %d 个, standoff %.1f mm" % (
            len(pass_txt) + 1, p.label, n0 + 1, len(a["wp_pos"]), np_, p.standoff_mm))
    a["pass_key"], a["pass_short"], a["pass_txt"] = pass_key, pass_short, pass_txt
    a["n_wp"] = len(a["wp_pos"])
    return a


# ==================== ⑧ 逐航点法兰位姿 + 工具线段 (MATLAB :642-667) ====================
def compute_poses(wp_pos: np.ndarray, so_wp: np.ndarray, n_pl: np.ndarray,
                  arrival_angles_deg=ARRIVAL_ANGLES, tool: str = NOZZLE_TOOL,
                  gripper_tool: str = GRIPPER_TOOL) -> dict:
    """法兰公式(与 plot_spray_path.m / plot_nozzle_reach.m 的 pose_metrics 一致, MATLAB :643-667):
        P_flange = P_target - standoff*R(:,3) - R*TCP          (standoff **按遍取**)
        喷嘴线段 = [P_flange + R*[tcp_x, tcp_y, 0],  P_flange + R*TCP]   (后者 = 喷嘴口/锥顶)
        夹爪线段 = 同理(用夹爪 TCP, 尖端为 grip_tip)
        入射角   = acos(|ax_spray · n_pl|)      (喷淋轴与盆口法向夹角)
    返回 dict: p_flange/noz_a/noz_b/grip_a/grip_b/apex/grip_tip/d_pt/inc_deg/r_arr/ax_spray
    C++: 对应 ToolFlange::computeSprayFlangePose + spray_planner::computePoses。
    """
    r_arr = euler_to_R(float(arrival_angles_deg[0]), float(arrival_angles_deg[1]),
                       float(arrival_angles_deg[2]))
    ax_spray = r_arr[:, 2].copy()
    tcp_noz = np.asarray(TCP_MM[tool], float)
    tcp_grip = np.asarray(TCP_MM[gripper_tool], float)
    n_wp = len(wp_pos)
    out = {k: np.zeros((n_wp, 3)) for k in ("p_flange", "noz_a", "noz_b", "grip_a", "grip_b")}
    out["apex"] = np.zeros((n_wp, 3))
    out["grip_tip"] = np.zeros((n_wp, 3))
    out["d_pt"] = np.zeros(n_wp)
    out["inc_deg"] = np.zeros(n_wp)
    out["r_arr"], out["ax_spray"] = r_arr, ax_spray
    for k in range(n_wp):
        pk = np.asarray(wp_pos[k], float)
        pf = pk - float(so_wp[k]) * ax_spray - (r_arr @ tcp_noz)
        out["p_flange"][k] = pf
        out["noz_a"][k] = pf + (r_arr @ np.array([tcp_noz[0], tcp_noz[1], 0.0]))
        out["noz_b"][k] = pf + (r_arr @ tcp_noz)
        out["grip_a"][k] = pf + (r_arr @ np.array([tcp_grip[0], tcp_grip[1], 0.0]))
        out["grip_b"][k] = pf + (r_arr @ tcp_grip)
        out["apex"][k] = out["noz_b"][k]
        out["grip_tip"][k] = out["grip_b"][k]
        out["d_pt"][k] = float(np.linalg.norm(out["noz_b"][k] - pk))
        out["inc_deg"][k] = math.degrees(math.acos(min(1.0, abs(float(ax_spray @ n_pl)))))
    return out


# ==================== ⑨ 干涉判据 (MATLAB :662-663) ====================
def compute_clearance(poses: dict, check_pts: np.ndarray,
                      clear_warn_mm: float = CLEAR_WARN_MM) -> dict:
    """工具轴线线段到点云的最小距离 (MATLAB :662-663)

    ⚠ 判据 = "点到线段的垂距取最小", **与工具半径无关**(与 MATLAB / spray_planner::computeClearance
      同口径); 点云抽稀(stride>0)会让判据偏乐观, 严格结论请用 stride=0 复核。
    返回 dict: d_noz/d_grip(逐航点) + min_nozzle/min_gripper + 对应航点号 + risk_points
    """
    pts = np.asarray(check_pts, float)
    n_wp = len(poses["apex"])
    d_noz = np.zeros(n_wp)
    d_grip = np.zeros(n_wp)
    for k in range(n_wp):
        d_noz[k] = seg_point_dist(poses["noz_a"][k], poses["noz_b"][k], pts)[0]
        d_grip[k] = seg_point_dist(poses["grip_a"][k], poses["grip_b"][k], pts)[0]
    return dict(d_noz=d_noz, d_grip=d_grip,
                min_nozzle=float(d_noz.min()), min_gripper=float(d_grip.min()),
                i_min_nozzle=int(np.argmin(d_noz)) + 1,
                i_min_gripper=int(np.argmin(d_grip)) + 1,
                risk_points=int((np.minimum(d_noz, d_grip) < clear_warn_mm).sum()))


# ==================== ⑨ 覆盖率 (MATLAB :666 / :671-704) ====================
def compute_coverage(poses: dict, pass_info: dict, rim_uv: np.ndarray, c0: np.ndarray,
                     u_long: np.ndarray, v_short: np.ndarray, n_pl: np.ndarray,
                     cone_deg: float = SPRAY_CONE_DEG, cell_mm: float = COVER_CELL_MM,
                     nseg: int = 72) -> dict:
    """覆盖率: 锥体足迹投到盆口平面 + 栅格统计 (MATLAB :666 / :671-704)

    ① 逐航点足迹: footprint_on_plane(喷嘴口, 喷淋轴, 锥角/2, c0, n_pl, nseg)
    ② 盆口面内建栅格(cell_mm), 只统计落在盆口多边形内的单元(in_polygon_mask)
    ③ 全轨迹 / 仅栅格 / 仅沿口唇 的覆盖 + **逐遍单独与累计**
    ⚠ 单元排列与 MATLAB 对齐: MATLAB 用 meshgrid(u_c,v_c) 后 Uc(:),Vc(:)(列优先) =>
      numpy 用 indexing="ij" + ravel(order="F") 才逐单元对应, 否则逐单元对账会错位。
    返回 dict: pc_all/pc_rim/pc_raster/uncover_cm2/only_rim_cm2/only_ras_cm2/both_cm2/
               n_cell_in/a_cell_mm2/pc_pass(逐遍单独)/pc_pass_cum(逐遍累计)/cover_all/cell_uv
    """
    n_wp = len(poses["apex"])
    fp = [footprint_on_plane(poses["apex"][k], poses["ax_spray"], cone_deg / 2.0,
                             c0, n_pl, nseg) for k in range(n_wp)]
    u_lo, u_hi = float(rim_uv[:, 0].min()), float(rim_uv[:, 0].max())
    v_lo, v_hi = float(rim_uv[:, 1].min()), float(rim_uv[:, 1].max())
    u_edge = np.arange(u_lo - cell_mm / 2.0, u_hi + cell_mm / 2.0 + 1e-9, cell_mm)
    v_edge = np.arange(v_lo - cell_mm / 2.0, v_hi + cell_mm / 2.0 + 1e-9, cell_mm)
    u_c = 0.5 * (u_edge[:-1] + u_edge[1:])
    v_c = 0.5 * (v_edge[:-1] + v_edge[1:])
    vv, uu = np.meshgrid(v_c, u_c, indexing="ij")
    cell_uv = np.c_[uu.ravel(order="F"), vv.ravel(order="F")]
    in_open = in_polygon_mask(cell_uv, rim_uv)
    a_cell = cell_mm ** 2
    wp_type = np.asarray(pass_info["wp_type"], float)

    cover_all = coverage_mask(fp, cell_uv, in_open, c0, u_long, v_short)
    cover_rim = coverage_mask([fp[k] for k in range(n_wp) if wp_type[k] == 1],
                              cell_uv, in_open, c0, u_long, v_short)
    cover_ras = coverage_mask([fp[k] for k in range(n_wp) if wp_type[k] == 2],
                              cell_uv, in_open, c0, u_long, v_short)

    def pc(mask):
        return 100.0 * float(np.sum(mask)) / max(1, int(np.sum(in_open)))

    out = dict(cover_all=cover_all, in_open=in_open, cell_uv=cell_uv, fp=fp,
               a_cell_mm2=a_cell, n_cell_in=int(np.sum(in_open)),
               pc_all=pc(cover_all), pc_rim=pc(cover_rim), pc_raster=pc(cover_ras),
               uncover_cm2=float(np.sum(in_open & ~cover_all)) * a_cell / 100.0,
               only_rim_cm2=float(np.sum(cover_rim & ~cover_ras & in_open)) * a_cell / 100.0,
               only_ras_cm2=float(np.sum(cover_ras & ~cover_rim & in_open)) * a_cell / 100.0,
               both_cm2=float(np.sum(cover_rim & cover_ras & in_open)) * a_cell / 100.0)
    wp_pass = np.asarray(pass_info["wp_pass"], float)
    pc_pass, pc_cum = [], []
    cov_cum = np.zeros_like(in_open)
    for k in range(1, len(pass_info["pass_short"]) + 1):
        cov_k = coverage_mask([fp[i] for i in range(n_wp) if wp_pass[i] == k],
                              cell_uv, in_open, c0, u_long, v_short)
        cov_cum = cov_cum | cov_k
        pc_pass.append(pc(cov_k))
        pc_cum.append(pc(cov_cum))
    out["pc_pass"], out["pc_pass_cum"] = pc_pass, pc_cum
    return out


# ==================== 局部工具: 内缩"塌陷"清理 ====================
def _uv3(uv: np.ndarray) -> np.ndarray:
    """(n,2) -> (n,3): 复用 resample_by_turn.poly_turns(它按 3D 点写)算折角"""
    return np.c_[np.asarray(uv, dtype=float), np.zeros(len(uv))]


def _merge_close_cyclic(uv: np.ndarray, min_gap_mm: float) -> Tuple[np.ndarray, int]:
    """合并环形相邻近重复点(< min_gap_mm); 返回 (新点列, 合并掉的个数)

    用途: 清理内缩塌陷造成的"点堆积"(实测盆口下护裙尖角处最短边仅 0.32mm)。
    C++: 顺序遍历, 与"上一个保留点"比距离即可。
    """
    p = np.asarray(uv, dtype=float)
    n = len(p)
    if n < 3 or min_gap_mm <= 0:
        return p.copy(), 0
    keep = [0]
    for i in range(1, n):
        if float(np.linalg.norm(p[i] - p[keep[-1]])) >= min_gap_mm:
            keep.append(i)
    while len(keep) > 2 and float(np.linalg.norm(p[keep[0]] - p[keep[-1]])) < min_gap_mm:
        keep.pop()
    return p[keep], n - len(keep)


def _smooth_hot_corners(uv: np.ndarray, turn_soft_deg: float = 12.0, max_it: int = 200,
                        win: int = 2) -> Tuple[np.ndarray, int, float, float]:
    """对折角 > turn_soft_deg 的顶点做**局部环形平滑**, 迭代到达标或达上限

    每轮把**所有**超阈值顶点同时替换为"窗口内均值"; 但都基于**同一份快照**计算(与顺序无关)。
    (早先是"每轮只动最尖的那一个点", 实测收敛太慢 —— MATLAB 侧 60 轮只从 18.6° 降到 16.5°,
     使得 ring 折角卡在 24.7°; 批量更新后同样轮数即可达标。)
    ⚠ 定性: 这是**防御性**处理(针对"内缩量 > 局部曲率半径"时的塌陷), 不是 -505 的根因修复;
      根因修复是 resample_by_turn 的加密。详见 ring_offset_cleanup.m 文件头。
    返回 (新点列, 迭代次数, 平滑前最大折角, 平滑后最大折角)。
    C++: 同样循环 —— 每轮算折角、收集超阈索引、按快照窗口均值批量替换。
    """
    p = np.asarray(uv, dtype=float).copy()
    n = len(p)
    t0 = float(poly_turns(_uv3(p), True).max())
    it = 0
    for it in range(1, max(0, int(max_it)) + 1):
        tv = poly_turns(_uv3(p), True)
        hot = np.flatnonzero(tv > turn_soft_deg)
        if hot.size == 0:
            break
        snap = p.copy()
        for i in hot:
            idx = np.arange(i - win, i + win + 1) % n
            p[i] = snap[idx, :].mean(axis=0)
    return p, it, t0, float(poly_turns(_uv3(p), True).max())
