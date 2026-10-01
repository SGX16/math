# -*- coding: utf-8 -*-
"""生成论文图表"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.8

d = np.load('_results.npz')
R0 = 0.02

def rad_cm(xi, R=R0):
    return xi * R * 100.0   # 归一化坐标 -> 半径 cm

# ---------- 图1: 附件1 烘房空气条件 ----------
fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
t_h = d['t_air'] / 3600.0
ax1.plot(t_h, d['T_air'], color='#C0392B', lw=1.6, label='烘房温度')
ax1.set_xlabel('时间 t / h')
ax1.set_ylabel('温度 T / °C', color='#C0392B')
ax1.tick_params(axis='y', labelcolor='#C0392B')
ax1.set_xlim(0, 4)
ax2 = ax1.twinx()
ax2.plot(t_h, d['C_air'], color='#1F618D', lw=1.6, label='烘房水分浓度')
ax2.set_ylabel('水分浓度 C / (kg/kg)', color='#1F618D')
ax2.tick_params(axis='y', labelcolor='#1F618D')
ax1.set_title('图1  预热平衡阶段烘房温度与水分浓度变化')
lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, loc='center right', frameon=False)
plt.tight_layout()
plt.savefig('figures/fig1_air.png', dpi=300)
plt.close()

# ---------- 图2: 问题1 温度与水分径向分布 ----------
t_sel = [100, 300, 600, 900, 1200, 1800]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(t_sel)))
xi = d['p1_xi']
r = rad_cm(xi)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
for tt, c in zip(t_sel, colors):
    j = tt - 1
    ax1.plot(r, d['p1_T'][j], color=c, lw=1.6, label=f'{tt} s')
    ax2.plot(r, d['p1_C'][j], color=c, lw=1.6, label=f'{tt} s')
ax1.set_xlabel('到中心距离 r / cm'); ax1.set_ylabel('温度 T / °C')
ax2.set_xlabel('到中心距离 r / cm'); ax2.set_ylabel('水分浓度 C / (kg/kg)')
ax1.set_title('(a) 温度分布'); ax2.set_title('(b) 水分浓度分布')
ax1.set_xlim(0, 2); ax2.set_xlim(0, 2)
ax1.legend(frameon=False, fontsize=9); ax2.legend(frameon=False, fontsize=9)
fig.suptitle('图2  问题1：预热平衡阶段药材内部温度与水分浓度分布')
plt.tight_layout()
plt.savefig('figures/fig2_p1.png', dpi=300)
plt.close()

# ---------- 图3: 问题2 温度与水分径向分布 (3h) ----------
t_sel2 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
colors = plt.cm.plasma(np.linspace(0, 0.9, len(t_sel2)))
xi = d['p2_xi']
r = rad_cm(xi)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
for tt, c in zip(t_sel2, colors):
    j = int(tt * 3600) - 1
    ax1.plot(r, d['p2_T'][j], color=c, lw=1.6, label=f'{tt} h')
    ax2.plot(r, d['p2_C'][j], color=c, lw=1.6, label=f'{tt} h')
ax1.set_xlabel('到中心距离 r / cm'); ax1.set_ylabel('温度 T / °C')
ax2.set_xlabel('到中心距离 r / cm'); ax2.set_ylabel('水分浓度 C / (kg/kg)')
ax1.set_title('(a) 温度分布'); ax2.set_title('(b) 水分浓度分布')
ax1.set_xlim(0, 2); ax2.set_xlim(0, 2)
ax1.legend(frameon=False, fontsize=9); ax2.legend(frameon=False, fontsize=9)
fig.suptitle('图3  问题2：烘干前3小时药材内部温度与水分浓度分布')
plt.tight_layout()
plt.savefig('figures/fig3_p2.png', dpi=300)
plt.close()

# ---------- 图4: 问题3 水分分布 + 中心水分衰减 ----------
xi = d['p3_xi']
r = rad_cm(xi)
t_h = d['p3_times'] / 3600.0
# 选取 6,12,16h 附近时刻
tsel = [6, 12]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, 4))
for tt, c in zip(tsel + [int(d['t_dry3']/3600)], colors[:3]):
    j = np.argmin(np.abs(t_h - tt))
    ax1.plot(r, d['p3_C'][j], color=c, lw=1.6, label=f'{tt} h')
ax1.axhline(0.15, color='#E74C3C', ls='--', lw=1.2)
ax1.text(1.4, 0.16, '要求阈值 0.15 kg/kg', color='#E74C3C', fontsize=9)
ax1.set_xlabel('到中心距离 r / cm'); ax1.set_ylabel('水分浓度 C / (kg/kg)')
ax1.set_xlim(0, 2); ax1.set_ylim(0, 1.3)
ax1.set_title('(a) 不同时刻水分浓度分布')
ax1.legend(frameon=False, fontsize=9)
# 中心与表面水分随时间
ax2.plot(t_h, d['p3_C'][:, 0], color='#C0392B', lw=1.8, label='中心 (r=0)')
ax2.plot(t_h, d['p3_C'][:, -1], color='#1F618D', lw=1.8, label='表面 (r=2 cm)')
ax2.axhline(0.15, color='#E74C3C', ls='--', lw=1.2)
ax2.axvline(d['t_dry3']/3600, color='#27AE60', ls=':', lw=1.2)
t_dry3_h = d['t_dry3'] / 3600.0
ax2.plot([t_dry3_h], [0.15], 'o', color='#27AE60', ms=6)
ax2.annotate(f'烘干时长 {t_dry3_h:.2f} h',
             xy=(t_dry3_h, 0.15), xytext=(t_dry3_h + 2, 0.35),
             arrowprops=dict(arrowstyle='->', color='#27AE60'), color='#27AE60')
ax2.set_xlabel('时间 t / h'); ax2.set_ylabel('水分浓度 C / (kg/kg)')
ax2.set_title('(b) 中心与表面水分浓度随时间变化')
ax2.set_xlim(0, 20); ax2.set_ylim(0, 1.4)
ax2.legend(frameon=False, fontsize=9)
fig.suptitle('图4  问题3：烘干过程水分浓度变化与烘干时长确定')
plt.tight_layout()
plt.savefig('figures/fig4_p3.png', dpi=300)
plt.close()

# ---------- 图5: 附件2 半径收缩 ----------
t_h = d['t_rad'] / 3600.0
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(t_h, d['R_rad'], color='#8E44AD', lw=1.8)
ax.set_xlabel('时间 t / h'); ax.set_ylabel('药材半径 R / cm')
ax.set_xlim(0, 72); ax.set_ylim(1.0, 2.1)
ax.set_title('图5  烘干过程药材半径的收缩（附件2）')
plt.tight_layout()
plt.savefig('figures/fig5_radius.png', dpi=300)
plt.close()

# ---------- 图6: 问题4 水分分布(含收缩) + 问题3/4对比 ----------
xi = d['p4_xi']
t_h4 = d['p4_times'] / 3600.0
tsel = [6, 12, 18]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
colors = plt.cm.plasma(np.linspace(0.1, 0.9, 4))
for tt, c in zip(tsel, colors[:3]):
    j = np.argmin(np.abs(t_h4 - tt))
    R_cur = d['p4_R'][j] * 100.0
    r_phys = d['p4_xi'] * R_cur
    ax1.plot(r_phys, d['p4_C'][j], color=c, lw=1.6, label=f'{tt} h')
    ax1.plot([R_cur], [d['p4_C'][j, -1]], 'o', color=c, ms=5)
ax1.axhline(0.15, color='#E74C3C', ls='--', lw=1.2)
ax1.set_xlabel('到中心距离 r / cm'); ax1.set_ylabel('水分浓度 C / (kg/kg)')
ax1.set_xlim(0, 2); ax1.set_ylim(0, 2.2)
ax1.set_title('(a) 考虑收缩的水分浓度分布')
ax1.legend(frameon=False, fontsize=9)
# 对比中心水分
t_h3 = d['p3_times'] / 3600.0
ax2.plot(t_h3, d['p3_C'][:, 0], color='#C0392B', lw=1.8, label='问题3 (无收缩)')
ax2.plot(t_h4, d['p4_C'][:, 0], color='#1F618D', lw=1.8, label='问题4 (有收缩)')
ax2.axhline(0.15, color='#E74C3C', ls='--', lw=1.2)
ax2.set_xlabel('时间 t / h'); ax2.set_ylabel('中心水分浓度 C / (kg/kg)')
ax2.set_xlim(0, 22); ax2.set_ylim(0, 1.4)
ax2.set_title('(b) 中心水分浓度对比')
ax2.legend(frameon=False, fontsize=9)
fig.suptitle('图6  问题4：考虑尺寸收缩的烘干过程')
plt.tight_layout()
plt.savefig('figures/fig6_p4.png', dpi=300)
plt.close()

print('全部图表已生成:')
import os
for f in sorted(os.listdir('figures')):
    if f.endswith('.png'):
        print('  figures/' + f)
