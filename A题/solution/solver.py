# -*- coding: utf-8 -*-
"""
CUMCM 2026 A题 —— 药材烘干问题 数值求解器
===========================================
模型：一维径向（无限长圆柱）瞬态热传导 + 水分扩散耦合
   热:   rho(C)*cp(C) * dT/dt = (1/R^2)(1/xi) d/dxi( xi * k(C) * dT/dxi )
   质:   dC/dt = (1/R^2)(1/xi) d/dxi( xi * D(C,T) * dC/dxi )
   xi = r/R 为归一化(物质)坐标, xi in [0,1]
边界条件:
   xi=0: 对称  dT/dxi = 0, dC/dxi = 0
   xi=1: -k*(1/R) dT/dxi = h*(T - T_air(t));  -D*(1/R) dC/dxi = hm*(C - C_air(t))
数值方法：有限体积法, 全隐式(向后欧拉), Picard 迭代处理非线性耦合
"""

import numpy as np
from scipy.interpolate import interp1d

# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------

from scipy.linalg import solve_banded, solveh_banded


def solve_tridiag(sub, diag, sup, rhs):
    """求解三对角方程组 sub[i]*x[i-1] + diag[i]*x[i] + sup[i]*x[i+1] = rhs[i]
    sub[0]=0, sup[-1]=0. 返回 x."""
    n = len(rhs)
    ab = np.zeros((3, n))
    ab[0, 1:] = sup[:-1]
    ab[1, :] = diag
    ab[2, :-1] = sub[1:]
    return solve_banded((1, 1), ab, rhs)


def round4(x):
    """四舍五入保留4位小数（half-up）"""
    return np.floor(np.asarray(x, dtype=float) * 1e4 + 0.5) / 1e4


# ----------------------------------------------------------------------
# 物性函数（返回 rho, cp, k, D）
# ----------------------------------------------------------------------

def props_p1(C, T):
    """问题1：附录2 常数物性，D 仅与 C 有关。T 为摄氏度，D 与温度无关。"""
    rho = 820.0
    cp = 2600.0
    k = 0.36
    D = 7e-9 * np.exp(-0.89 * C)
    return rho, cp, k, D


def props_p23(C, T):
    """问题2/3：附录3。T 为摄氏度，D 中温度需用开尔文。"""
    rho = 650.0 + 128.0 * C
    cp = 1450.0 + 2736.0 * C / (C + 1.0)
    k = 0.21 + 0.38 * C / (C + 1.0)
    Tk = T + 273.15
    D = 2.4e-3 * np.exp(-0.45 * C) * np.exp(-3850.0 / Tk)
    return rho, cp, k, D


def props_p4(C, T):
    """问题4：附录4。T 为摄氏度，D 中温度用开尔文。"""
    rho = 760.0 + 90.0 * C
    cp = 1850.0 + 2150.0 * C / (C + 1.0)
    k = 0.12 + 0.20 * C / (C + 1.0)
    Tk = T + 273.15
    D = 4.2e-4 * np.exp(-0.30 * C) * np.exp(-3850.0 / Tk)
    return rho, cp, k, D


# ----------------------------------------------------------------------
# 核心求解器：单个时间步（有限体积，隐式，Picard 迭代）
# ----------------------------------------------------------------------

def one_step(T, C, T_amb, C_amb, R, dt, xi, props_fn, h, hm, n_iter=4,
             tol=1e-10, theta=1.0):
    """
    从时刻 n 推进到 n+1（θ-格式：θ=1 向后欧拉，θ=0.5 Crank-Nicolson）。
    T, C: 节点上的温度(°C)、水分浓度(kg/kg)，长度 N+1
    T_amb, C_amb: 下一时刻的烘房温度、水分浓度
    R: 当前半径 (m); dt: 时间步长 (s); xi: 节点坐标 (0..1)
    props_fn: (C,T)->(rho,cp,k,D); h, hm: 对流换热/传质系数
    """
    N = len(xi) - 1
    dxi = 1.0 / N

    # 控制体积面坐标
    xi_p = np.minimum(1.0, (np.arange(N + 1) + 0.5) * dxi)   # xi_{i+1/2}
    xi_m = np.maximum(0.0, (np.arange(N + 1) - 0.5) * dxi)   # xi_{i-1/2}

    # 控制体积 (m^3)
    V = np.pi * R ** 2 * (xi_p ** 2 - xi_m ** 2)
    # 面通量系数：2*pi*xi_{i+1/2}/dxi（面 i+1/2, i=0..N-1）
    face = 2.0 * np.pi * xi_p[:-1] / dxi

    def to_arr(x):
        a = np.asarray(x, dtype=float)
        return np.full(N + 1, float(a)) if a.ndim == 0 else a

    om = 1.0 - theta

    # Picard 迭代
    T_new = T.copy()
    C_new = C.copy()
    for _ in range(n_iter):
        T_old_iter = T_new.copy()
        C_old_iter = C_new.copy()

        # ---------- 热方程 ----------
        rho, cp, k, _ = props_fn(C_old_iter, T_old_iter)
        rho = to_arr(rho); cp = to_arr(cp); k = to_arr(k)
        beta = rho * cp
        k_face = 0.5 * (k[:-1] + k[1:])

        # 面电导
        fin = face[:-1] * k_face[:-1]      # 面 i-1/2 (i=1..N-1)
        fout = face[1:] * k_face[1:]       # 面 i+1/2 (i=1..N-1)
        f0 = face[0] * k_face[0]           # 面 1/2 (中心)
        finN = face[-1] * k_face[-1]       # 面 N-1/2 (表面内侧)
        surf = 2.0 * np.pi * R * h         # 表面对流电导

        sub = np.zeros(N + 1); diag = np.zeros(N + 1)
        sup = np.zeros(N + 1); rhs = np.zeros(N + 1)

        # 内部节点
        sub[1:N] = -theta * fin
        sup[1:N] = -theta * fout
        diag[1:N] = V[1:N] * beta[1:N] / dt + theta * (fin + fout)
        # 显式部分 L(T^n)
        LT = fin * (T[:-2] - T[1:-1]) + fout * (T[2:] - T[1:-1])
        rhs[1:N] = V[1:N] * beta[1:N] / dt * T[1:-1] + om * LT

        # 中心节点
        diag[0] = V[0] * beta[0] / dt + theta * f0
        sup[0] = -theta * f0
        rhs[0] = V[0] * beta[0] / dt * T[0] + om * f0 * (T[1] - T[0])

        # 表面节点
        sub[N] = -theta * finN
        diag[N] = V[N] * beta[N] / dt + theta * (finN + surf)
        rhs[N] = (V[N] * beta[N] / dt * T[N]
                  + om * (finN * (T[N - 1] - T[N]) - surf * T[N])
                  + surf * T_amb)

        T_new = solve_tridiag(sub, diag, sup, rhs)

        # ---------- 质方程 ----------
        _, _, _, D = props_fn(C_old_iter, T_new)
        D = to_arr(D)
        D_face = 0.5 * (D[:-1] + D[1:])

        fin = face[:-1] * D_face[:-1]
        fout = face[1:] * D_face[1:]
        f0 = face[0] * D_face[0]
        finN = face[-1] * D_face[-1]
        surf = 2.0 * np.pi * R * hm

        sub = np.zeros(N + 1); diag = np.zeros(N + 1)
        sup = np.zeros(N + 1); rhs = np.zeros(N + 1)

        sub[1:N] = -theta * fin
        sup[1:N] = -theta * fout
        diag[1:N] = V[1:N] / dt + theta * (fin + fout)
        LC = fin * (C[:-2] - C[1:-1]) + fout * (C[2:] - C[1:-1])
        rhs[1:N] = V[1:N] / dt * C[1:-1] + om * LC

        diag[0] = V[0] / dt + theta * f0
        sup[0] = -theta * f0
        rhs[0] = V[0] / dt * C[0] + om * f0 * (C[1] - C[0])

        sub[N] = -theta * finN
        diag[N] = V[N] / dt + theta * (finN + surf)
        rhs[N] = (V[N] / dt * C[N]
                  + om * (finN * (C[N - 1] - C[N]) - surf * C[N])
                  + surf * C_amb)

        C_new = solve_tridiag(sub, diag, sup, rhs)

        if (np.max(np.abs(T_new - T_old_iter)) < tol and
                np.max(np.abs(C_new - C_old_iter)) < tol):
            break

    return T_new, C_new


# ----------------------------------------------------------------------
# 主模拟函数
# ----------------------------------------------------------------------

def simulate(T_air_fn, C_air_fn, R_fn, props_fn, N, T0, C0, t_end, dt,
             h, hm, save_times=None, n_iter=4, theta=1.0):
    """
    完整模拟。返回 dict：
      xi: 归一化坐标
      save_times: 保存时刻
      T_save, C_save: 形状 (len(save_times), N+1)
      R_save: 对应时刻半径
    save_times 为 None 时不保存中间结果（仅返回最终状态）。
    """
    xi = np.linspace(0.0, 1.0, N + 1)
    T = np.full(N + 1, T0, dtype=float)
    C = np.full(N + 1, C0, dtype=float)

    if save_times is None:
        save_times = np.array([t_end])
    save_times = np.asarray(save_times, dtype=float)

    T_save = np.zeros((len(save_times), N + 1))
    C_save = np.zeros((len(save_times), N + 1))
    R_save = np.zeros(len(save_times))

    # 若 t=0 需要保存
    n_steps = int(round(t_end / dt))
    # 对每个时间步推进；记录落在 save_times 附近的结果
    # 用指针遍历 save_times
    ptr = 0
    t = 0.0
    # 保存 t=0
    if save_times[0] == 0.0:
        R0 = R_fn(0.0)
        T_save[0] = T
        C_save[0] = C
        R_save[0] = R0
        ptr = 1

    for step in range(1, n_steps + 1):
        t_new = step * dt
        # 目标时刻(下一时刻)的边界条件与半径
        R_new = R_fn(t_new)
        T_amb = T_air_fn(t_new)
        C_amb = C_air_fn(t_new)
        # 半径变化缓慢，用 R_new 作为当前步的 R
        T, C = one_step(T, C, T_amb, C_amb, R_new, dt, xi, props_fn, h, hm, n_iter,
                        theta=theta)

        # 记录
        while ptr < len(save_times) and abs(save_times[ptr] - t_new) < dt * 0.5:
            # 用线性插值到精确 save_time（save_times 均为 dt 的整数倍时直接取）
            T_save[ptr] = T
            C_save[ptr] = C
            R_save[ptr] = R_new
            ptr += 1

    return dict(xi=xi, save_times=save_times, T=T_save, C=C_save, R=R_save,
                T_final=T, C_final=C)


def simulate_until(T_air_fn, C_air_fn, R_fn, props_fn, N, T0, C0, dt,
                   h, hm, stop_when, record_every, t_max, n_iter=4, theta=1.0):
    """
    模拟直到 stop_when(T,C,t) 返回 True 或 t 达到 t_max。
    每隔 record_every 秒记录一次结果。
    返回 dict(save_times, T, C, R, t_end)。t_end 为满足停止条件(或 t_max)的时刻。
    """
    xi = np.linspace(0.0, 1.0, N + 1)
    T = np.full(N + 1, T0, dtype=float)
    C = np.full(N + 1, C0, dtype=float)

    save_times = []
    T_list = []
    C_list = []
    R_list = []

    def record(t):
        save_times.append(t)
        T_list.append(T.copy())
        C_list.append(C.copy())
        R_list.append(R_fn(t))

    n_steps = int(round(t_max / dt))
    next_rec = record_every
    t = 0.0
    t_end = t_max
    for step in range(1, n_steps + 1):
        t_new = step * dt
        R_new = R_fn(t_new)
        T_amb = T_air_fn(t_new)
        C_amb = C_air_fn(t_new)
        T, C = one_step(T, C, T_amb, C_amb, R_new, dt, xi, props_fn, h, hm,
                        n_iter, theta=theta)
        # 记录（每隔 record_every）
        if t_new >= next_rec - dt * 0.5:
            record(t_new)
            next_rec += record_every
        # 停止条件
        if stop_when(T, C, t_new):
            t_end = t_new
            # 若停止时刻未被记录，补记
            if abs(t_new - (save_times[-1] if save_times else -1)) > dt * 0.5:
                record(t_new)
            break

    return dict(xi=xi, save_times=np.array(save_times),
                T=np.array(T_list), C=np.array(C_list),
                R=np.array(R_list), t_end=t_end)


def make_air_functions(t_air, T_air, C_air, T_const=50.0, C_const=0.05):
    """构造烘房温度/水分浓度函数。t>t_air[-1] 时取恒定值。"""
    t_air = np.asarray(t_air, float)
    T_air = np.asarray(T_air, float)
    C_air = np.asarray(C_air, float)
    t_last = t_air[-1]

    def T_fn(t):
        t = np.asarray(t, float)
        return np.where(t <= t_last, np.interp(t, t_air, T_air), T_const)

    def C_fn(t):
        t = np.asarray(t, float)
        return np.where(t <= t_last, np.interp(t, t_air, C_air), C_const)

    return T_fn, C_fn


def make_R_function(t_rad, R_rad):
    """构造半径函数 R(t)（米），线性插值。"""
    t_rad = np.asarray(t_rad, float)
    R_rad = np.asarray(R_rad, float) / 100.0   # cm -> m

    def R_fn(t):
        t = np.asarray(t, float)
        return np.clip(np.interp(t, t_rad, R_rad), R_rad.min(), R_rad.max())
    return R_fn
