# -*- coding: utf-8 -*-
"""
曲率自适应重采样(圆角处自动加密) —— MATLAB: matlab/resample_by_turn.m 的逐行移植

【为什么需要】(2026-09-19, 与 MATLAB 侧同口径)
    ring(内缩圈)是"沿口细曲线沿内法向偏移"后按喷淋间距 s_fp(≈41mm) **等间距**重采样得到的。
    盆口紧圆角(下护裙尖角 / 上翻边, 等效半径 22~35mm)用 41mm 一步走时一步就要转 60°~71°,
    折线在此成了"折点"; 该处姿态又恰好 J5≈-89°(腕部奇异) -> 控制器关节插值出现速度台阶
    (实测 J4 相邻两拍位移 65 µrad -> 14 µrad, 即 0.0325 -> 0.0070 rad/s) -> -505 停机。
    ⚠ 曲线本身是光滑的(细采样每步仅 ~6~17°), 问题纯粹是**采样密度**。

【做法】在**原细曲线**上重采样, 只加密、不改形状:
    沿细曲线走, 累加 (a) 弧长  (b) 自上一个落点以来的"累计转角";
    并额外做"**待定弦自身转弯**"保护 —— 弦相对"上一落点处切向"偏了 0.2×预算就提前落点
    (否则 41mm 的长弦会一路伸进圆角: 41mm 走在 r=35mm 弧上自身就转 67°, 弦-弦交界折角会翻倍);
    预算还会按实测折角**自动收紧**(≤3 轮), **下限 = 细曲线自身折角**(再收紧只会把整条轨迹加密)。
    => 直线段步长 = step_mm(喷淋间距不变); 圆角处自动收缩到 ~6~20mm;
    => **所有落点都取自原细曲线 => 几何零改动(切角深度 = 0), 覆盖率/内缩量完全不变**。

【实测】(urinal_9_o2e_real_safe 的沿口复算, 与 MATLAB 侧同参数)
    现状(等间距 41mm): ring 21 点, 最大步进折角 70.9°      <- -505 触发形态
    本函数:            ring ~40 点(2 批), 最大折角 ~25°(约 2.8 倍改善), 几何零改动

【调用方】python/spray_plan/spray_planner.py 的"内缩圈"块(替换等间距 resample_arclen)。
    唇左/右半圈 与 蛇形(栅格) **不动**: 唇半圈最大步进转角 ~20° 且最尖处在遍的端点(控制器本就停靠);
    蛇形拐点是线端近 180° 折返, move_blend 实测能正常执行。
"""
import math

import numpy as np


def poly_turns(p, closed=False):
    """折线每个顶点处的转角(deg); 开曲线的端点返回 0 (对应 MATLAB 局部函数 poly_turns)"""
    p = np.asarray(p, dtype=float)
    n = len(p)
    tv = np.zeros(n)
    for i in range(n):
        if (not closed) and (i == 0 or i == n - 1):
            continue
        im = (i - 1) % n
        ip = (i + 1) % n
        a = p[i] - p[im]
        b = p[ip] - p[i]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            continue
        c = max(-1.0, min(1.0, float(a @ b) / (na * nb)))
        tv[i] = math.degrees(math.acos(c))
    return tv


def resample_by_turn(p_fine, step_mm, max_turn_deg, min_step_mm=5.0, closed=False):
    """按"弧长 + 累计转角 + 待定弦自身转弯"自适应落点; 返回 (P_out, info)

    p_fine       : (N,3) 细采样有序点(基座系 mm); 闭环时**不要把首点复制到末尾**
    step_mm      : 直线段落点间距(= 喷淋间距 s_fp)
    max_turn_deg : 允许的"每步累计转角"上限(deg)
    min_step_mm  : 相邻落点最小间距(保护)
    closed       : True = 闭环(内缩圈): 末点->首点也算一条边
    """
    p = np.asarray(p_fine, dtype=float)
    n = len(p)
    info = {
        "n_in": n,
        "closed": bool(closed),
        "n_out": n,
        "step_max_mm": 0.0,
        "step_min_mm": 0.0,
        "turn_max_deg": 0.0,
        "budget_used_deg": float(max_turn_deg),
    }
    if n < 3:
        return p.copy(), info

    # ---- 细边长度 / 顶点转角 ----
    dseg = np.zeros(n)
    turn = np.zeros(n)
    for i in range(n):
        ip = (i + 1) % n
        dseg[i] = float(np.linalg.norm(p[ip] - p[i]))
        if (not closed) and (i == 0 or i == n - 1):
            continue
        im = (i - 1) % n
        a = p[i] - p[im]
        b = p[ip] - p[i]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na > 1e-9 and nb > 1e-9:
            c = max(-1.0, min(1.0, float(a @ b) / (na * nb)))
            turn[i] = math.degrees(math.acos(c))

    # ---- 走一遍(闭环含 末点->首点 那条边), 按"弧长 / 累计转角 / 弦自身转弯"落点 ----
    n_edge = n if closed else n - 1
    budget = float(max_turn_deg)
    floor_deg = float(turn.max())          # 细曲线自身折角 = 粒度下限(再收紧只会整条加密)
    if floor_deg > budget:
        budget = floor_deg

    idx = list(range(n))
    for _attempt in range(3):
        idx = [0]
        acc_l = 0.0
        acc_t = 0.0
        for e in range(n_edge):
            ip = (e + 1) % n
            acc_l += dseg[e]
            acc_t += turn[e]
            last = idx[-1]
            ipe = (last + 1) % n
            pend = p[ip] - p[last]
            tan0 = p[ipe] - p[last]
            np_ = float(np.linalg.norm(pend))
            nt_ = float(np.linalg.norm(tan0))
            dchord = 0.0
            if np_ > 1e-9 and nt_ > 1e-9:
                cc = max(-1.0, min(1.0, float(pend @ tan0) / (np_ * nt_)))
                dchord = math.degrees(math.acos(cc))
            if (acc_l >= step_mm or acc_t >= budget or dchord >= 0.2 * budget) and acc_l >= min_step_mm:
                idx.append(ip)
                acc_l = 0.0
                acc_t = 0.0
        if closed and len(idx) > 1 and idx[-1] == 0:
            idx = idx[:-1]                 # 闭环: wrap 回到首点 -> 去掉重复
        elif (not closed) and idx[-1] != n - 1:
            idx.append(n - 1)              # 开曲线: 末尾保留原终点
        info["budget_used_deg"] = budget
        info["turn_max_deg"] = float(poly_turns(p[idx], closed).max())
        if info["turn_max_deg"] <= max_turn_deg or budget <= floor_deg + 1e-9:
            break
        budget = max(budget * max_turn_deg / info["turn_max_deg"] * 0.95, floor_deg)

    p_out = p[idx]
    m = len(p_out)
    info["n_out"] = m
    info["turn_max_deg"] = float(poly_turns(p_out, closed).max())
    if m >= 2:
        st = np.zeros(m)
        for i in range(m):
            ip = (i + 1) % m
            st[i] = float(np.linalg.norm(p_out[ip] - p_out[i]))
        info["step_max_mm"] = float(st.max())
        info["step_min_mm"] = float(st.min())
    return p_out, info
