# -*- coding: utf-8 -*-
"""验证有限体积求解器 vs 无限长圆柱瞬态导热/扩散解析解"""
import numpy as np
from scipy.special import j0, j1, jn_zeros
from solver import simulate, solve_tridiag

# ---------------- 解析解 ----------------
def analytic_cyl(r, t, R, T0, Tinf, alpha, Bi, nmax=200):
    """无限长圆柱, 对流边界, 初温均匀 T0, 环境 Tinf。
    返回各 (r,t) 点的温度。r 可为数组。"""
    r = np.atleast_1d(np.asarray(r, float))
    # 特征值 λ_n: λ J1(λ) = Bi J0(λ)
    # 用求根：在区间内扫描
    lam = []
    dlam = 0.5
    x = dlam
    while len(lam) < nmax:
        if x > 1e5:
            break
        f = x * j1(x) - Bi * j0(x)
        if len(lam) == 0:
            prev = Bi * j0(0.0) * 1e-6  # dummy
        # 用二分求根（在 [x-dlam, x] 找符号变化）
        a, b = x - dlam, x
        fa = a * j1(a) - Bi * j0(a)
        fb = f
        if fa * fb < 0:
            for _ in range(100):
                m = 0.5 * (a + b)
                fm = m * j1(m) - Bi * j0(m)
                if fa * fm <= 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            root = 0.5 * (a + b)
            if len(lam) == 0 or root - lam[-1] > 1e-6:
                lam.append(root)
        x += dlam
    lam = np.array(lam)

    res = np.zeros((len(r), len(t)))
    for k, tt in enumerate(t):
        Fo = alpha * tt / R ** 2
        for j, rj in enumerate(r):
            s = 0.0
            for ln in lam:
                An = 2.0 * j1(ln) / (ln * (j0(ln) ** 2 + j1(ln) ** 2))
                s += An * np.exp(-ln ** 2 * Fo) * j0(ln * rj / R)
            res[j, k] = Tinf + (T0 - Tinf) * s
    return res


def test_heat():
    print("=" * 60)
    print("验证1：热传导 (常数物性, 恒定环境温度)")
    R = 0.02
    rho, cp, k = 820.0, 2600.0, 0.36
    h = 25.0
    alpha = k / (rho * cp)
    Bi = h * R / k
    T0, Tinf = 28.0, 50.0

    def props(C, T):
        return rho, cp, k, 0.0   # D 无关

    N = 400
    dt = 1.0
    times = [100, 300, 600, 1200, 1800]
    sim = simulate(lambda t: Tinf, lambda t: 0.0, lambda t: R, props,
                   N, T0, 0.0, 1800.0, dt, h, 1e-9, save_times=times)
    r_pts = np.array([0.0, 0.01, 0.02])
    idx = [int(round(rr / R * N)) for rr in r_pts]
    ana = analytic_cyl(r_pts, times, R, T0, Tinf, alpha, Bi)

    print(f"{'t(s)':>6} {'r(cm)':>6} {'数值':>12} {'解析':>12} {'误差':>12}")
    maxerr = 0.0
    for j, tt in enumerate(times):
        for i, rr in enumerate(r_pts):
            num = sim['T'][j, idx[i]]
            ref = ana[i, j]
            err = abs(num - ref)
            maxerr = max(maxerr, err)
            print(f"{tt:>6} {rr*100:>6.2f} {num:>12.5f} {ref:>12.5f} {err:>12.2e}")
    print(f"最大误差: {maxerr:.2e}")
    assert maxerr < 0.05, "热传导验证失败！"


def test_mass():
    print("=" * 60)
    print("验证2：水分扩散 (常数 D, 恒定环境水分)")
    R = 0.02
    D = 7e-9
    hm = 8e-7
    C0, Cinf = 2.55, 0.05
    # 质扩散的 Biot 数
    Bi = hm * R / D
    # 扩散解析解同形式 (alpha -> D)
    def props(C, T):
        return 1.0, 1.0, 0.0, D  # beta=1 用 rho*cp=1 无关；k 无关

    N = 800
    dt = 1.0
    times = [300, 600, 1200, 1800]
    sim = simulate(lambda t: 0.0, lambda t: Cinf, lambda t: R, props,
                   N, 0.0, C0, 1800.0, dt, 1e-9, hm, save_times=times)
    r_pts = np.array([0.0, 0.01, 0.019, 0.02])
    idx = [int(round(rr / R * N)) for rr in r_pts]
    ana = analytic_cyl(r_pts, times, R, C0, Cinf, D, Bi)

    print(f"{'t(s)':>6} {'r(cm)':>6} {'数值':>12} {'解析':>12} {'误差':>12}")
    maxerr = 0.0
    for j, tt in enumerate(times):
        for i, rr in enumerate(r_pts):
            num = sim['C'][j, idx[i]]
            ref = ana[i, j]
            err = abs(num - ref)
            maxerr = max(maxerr, err)
            print(f"{tt:>6} {rr*100:>6.3f} {num:>12.6f} {ref:>12.6f} {err:>12.2e}")
    print(f"最大误差: {maxerr:.2e}")
    assert maxerr < 0.05, "水分扩散验证失败！"


if __name__ == "__main__":
    test_heat()
    test_mass()
    print("=" * 60)
    print("全部验证通过 ✓")
