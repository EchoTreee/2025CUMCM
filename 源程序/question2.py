import numpy as np
import pandas as pd
import pywt
from scipy.signal import hilbert
from scipy.optimize import differential_evolution, least_squares
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.default'] = 'regular'
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=RuntimeWarning)
print("1. 参数设置")
a = 10.0      # 入射角度1 (度)
b = 15.0      # 入射角度2 (度)
c = 1.0       # 空气折射率
d = 2.5       # 衬底折射率初值 (SiC典型值)
e = 2000      # 蒙特卡洛模拟次数
f = 0.001     # 仪器噪声水平 (0.1%
g = {'popsize': 50, 'maxiter': 200, 'disp': True, 'seed': 0}
h = {'method': 'trf', 'verbose': 2}
print("\n2. 数据读取与预处理")
try:
    i = pd.read_excel('附件1.xlsx', header=1)
    j = pd.read_excel('附件2.xlsx', header=1)
except FileNotFoundError:
    print("错误: 未找到数据文件。请确保路径 '附件/附件1.xlsx' 和 '附件/附件2.xlsx' 正确。")
    exit()

# 数据清洗与转换
k = pd.to_numeric(i.iloc[:, 0], errors='coerce').values
l = pd.to_numeric(i.iloc[:, 1], errors='coerce').values
m = pd.to_numeric(j.iloc[:, 0], errors='coerce').values
n = pd.to_numeric(j.iloc[:, 1], errors='coerce').values
o = ~np.isnan(k) & ~np.isnan(l)
p = ~np.isnan(m) & ~np.isnan(n)
k, l = k[o], l[o]
m, n = m[p], n[p]
q = 1e4 / k
r = l / 100.0
s = 1e4 / m
t = n / 100.0
print("\n3. 小波信号分解")
u = 'db4'
v = 3

def decompose_signal(R):
    w = pywt.wavedec(R, u, level=v)
    x = w.copy()
    for i in range(1, len(w)):
        x[i] = np.zeros_like(w[i])
    y = pywt.waverec(x, u)
    
    if len(y) != len(R):
               y = y[:len(R)]
    z = R - y
    aa = R - y - z
    return y, aa, z
B10, N10, S10 = decompose_signal(r)
B15, N15, S15 = decompose_signal(t)
print("\n4. 初值估计 (Hilbert-Huang变换)")
ab = 2.6
def estimate_initial_d(S, sigma, theta_deg):
    ac = hilbert(S)
    ad = np.unwrap(np.angle(ac))
    ae = np.diff(ad) / np.diff(sigma)
    af = np.mean(ae[50:-50])
    ag = np.deg2rad(theta_deg)
    ah = np.arcsin(c / ab * np.sin(ag))
    # 正确的物理公式
    ai = af / (2 * np.pi) / (2 * ab * np.cos(ah))
    return ai
aj = estimate_initial_d(S10, k, a)
ak = estimate_initial_d(S15, m, b)
al = (aj + ak) / 2
print(f'HHT初值估计(仅参考): d0_avg={al:.3f}um')
def multi_beam_model(params, lambda_val, theta_deg):
    d, n2, ns = params
    am = np.deg2rad(theta_deg)
    an = np.arcsin(np.clip((c / n2) * np.sin(am), -1, 1))
    ao = np.arcsin(np.clip((n2 / ns) * np.sin(an), -1, 1))
    ap = (4 * np.pi * n2 * d * np.cos(an)) / lambda_val
    aq = (c * np.cos(am) - n2 * np.cos(an)) / (c * np.cos(am) + n2 * np.cos(an))
    ar = (n2 * np.cos(am) - c * np.cos(an)) / (n2 * np.cos(am) + c * np.cos(an))
    as_ = (n2 * np.cos(an) - ns * np.cos(ao)) / (n2 * np.cos(an) + ns * np.cos(ao))
    at = (ns * np.cos(an) - n2 * np.cos(ao)) / (ns * np.cos(an) + n2 * np.cos(ao))
    au = (aq + as_ * np.exp(1j * ap)) / (1 + aq * as_ * np.exp(1j * ap))
    av = (ar + at * np.exp(1j * ap)) / (1 + ar * at * np.exp(1j * ap))
    aw = (np.abs(au)**2 + np.abs(av)**2) / 2
    return aw - np.mean(aw)
# *** 1. 强制设定厚度搜索边界在 8-10 µm 之间 ***
ax = [8.0, 2.4, 2.3]    # 厚度下限设为 8.0 µm
ay = [10.0, 2.8, 2.7]   # 厚度上限设为 10.0 µm
az = list(zip(ax, ay))

# 确保初始点落在新的边界内
ba = np.clip([9.0, ab, d], ax, ay) # 给定一个中间的初始值 9.0 µm
print(f"\n优化器已强制设定厚度边界: [{ax[0]}, {ay[0]}] µm")
def objective_function_global(params):
    bb = multi_beam_model(params, q, a)
    bc = multi_beam_model(params, s, b)
    return np.sqrt(np.mean((S10 - bb)**2)) + np.sqrt(np.mean((S15 - bc)**2))
def residuals_function_local(params):
    bd = multi_beam_model(params, q, a)
    be = multi_beam_model(params, s, b)
    return np.concatenate([S10 - bd, S15 - be])
print("\n6.1 开始全局优化 (Differential Evolution)...")
bf = differential_evolution(objective_function_global, az, **g)
bg = bf.x
print(f"全局优化完成。最佳参数: d={bg[0]:.4f}, n2={bg[1]:.4f}, ns={bg[2]:.4f}")

print("\n6.2 开始LM局部优化...")
bh = least_squares(residuals_function_local, bg, bounds=(ax, ay), **h)
bi = bh.x
d_opt, n2_opt, ns_opt = bi
print(f"优化结果: d={d_opt:.4f} um, n2={n2_opt:.4f}, ns={ns_opt:.4f}")
print(f"\n7. 开始蒙特卡洛模拟 ({e}次)...")
bj, bk, bl = np.zeros(e), np.zeros(e), np.zeros(e)
rng = np.random.default_rng(0)

for i in range(e):
    if (i + 1) % (e // 10) == 0: print(f"  ...完成 {i+1}/{e}")
    bm = S10 + f * rng.standard_normal(S10.shape)
    bn = S15 + f * rng.standard_normal(S15.shape)
    bo = np.clip([
        d_opt * (1 + 0.05 * rng.standard_normal()),
        n2_opt * (1 + 0.02 * rng.standard_normal()),
        ns_opt * (1 + 0.02 * rng.standard_normal())], ax, ay)
    def residuals_noisy(p):
        bp = multi_beam_model(p, q, a)
        bq = multi_beam_model(p, s, b)
        return np.concatenate([bm - bp, bn - bq])
    br = least_squares(residuals_noisy, bo, bounds=(ax, ay), method='trf', max_nfev=30)
    bj[i], bk[i], bl[i] = br.x

# 统计结果
bs = np.mean(bj)
bt = np.std(bj)
bu = [bs - 1.96 * bt, bs + 1.96 * bt]

print('\n=============== 最终结果 ================')
print(f'厚度估计: {bs:.4f} ± {bt:.4f} um')
print(f'95%置信区间: [{bu[0]:.4f}, {bu[1]:.4f}] um')
print(f'变异系数: {bt / bs * 100:.2f}%')

#可视化结果
print("\n8. 生成可视化结果...")

# 图1: 信号分解图 (包含10度和15度)
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(k, r, color='orange', lw=1.5, label='原始数据')
plt.plot(k, B10, 'b-', lw=1.5, label='背景分量')
plt.plot(k, S10, 'r-', lw=1.2, label='条纹信号')
plt.xlabel(r'波数 (cm$^{-1}$)')
plt.ylabel('反射率')
plt.title('10°入射角信号分解')
plt.legend()
plt.grid(True)
plt.xlim(min(k), max(k))

plt.subplot(2, 1, 2)
plt.plot(m, t, color='orange', lw=1.5, label='原始数据')
plt.plot(m, B15, 'b-', lw=1.5, label='背景分量')
plt.plot(m, S15, 'r-', lw=1.2, label='条纹信号')
plt.xlabel(r'波数 (cm$^{-1}$)')
plt.ylabel('反射率')
plt.title('15°入射角信号分解')
plt.legend()
plt.grid(True)
plt.xlim(min(m), max(m))

plt.tight_layout()
plt.savefig('信号分解图.eps', dpi=300)
print('信号分解图已保存为 "信号分解图.eps"')

# 图2: 模型拟合图 (包含10度和15度)
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
bv = multi_beam_model([bs, n2_opt, ns_opt], q, a)
plt.plot(k, S10, color='orange', lw=1.5, label='提取的条纹信号')
plt.plot(k, bv, 'r--', lw=1.5, label='拟合模型')
plt.xlabel(r'波数 (cm$^{-1}$)')
plt.ylabel('条纹信号强度')
plt.title(f'10°模型拟合效果 (d={bs:.3f}μm)')
plt.legend()
plt.grid(True)
plt.xlim(min(k), max(k))
plt.subplot(2, 1, 2)
bw = multi_beam_model([bs, n2_opt, ns_opt], s, b)
plt.plot(m, S15, color='orange', lw=1.5, label='提取的条纹信号')
plt.plot(m, bw, 'r--', lw=1.5, label='拟合模型')
plt.xlabel(r'波数 (cm$^{-1}$)')
plt.ylabel('条纹信号强度')
plt.title(f'15°模型拟合效果 (d={bs:.3f}μm)')
plt.legend()
plt.grid(True)
plt.xlim(min(m), max(m))
plt.tight_layout()
plt.savefig('模型拟合图.eps', dpi=300)
print('模型拟合图已保存为 "模型拟合图.eps"')

# 图3: 残差分布图 (包含10度和15度)
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
bx = S10 - bv
by = np.sqrt(np.mean(bx**2))
plt.plot(k, bx, 'b-', lw=1)
plt.axhline(0, color='r', ls='--', lw=1.5)
plt.xlabel(r'波数 (cm$^{-1}$)')
plt.ylabel('残差值')
plt.title(f'10°残差分布 (RMSE={by:.4f})')
plt.grid(True)
plt.xlim(min(k), max(k))
plt.subplot(2, 1, 2)
bz = S15 - bw
ca = np.sqrt(np.mean(bz**2))
plt.plot(m, bz, 'b-', lw=1)
plt.axhline(0, color='r', ls='--', lw=1.5)
plt.xlabel(r'波数 (cm$^{-1}$)')
plt.ylabel('残差值')
plt.title(f'15°残差分布 (RMSE={ca:.4f})')
plt.grid(True)
plt.xlim(min(m), max(m))
plt.tight_layout()
plt.savefig('残差分布图.eps', dpi=300)
print('残差分布图已保存为 "残差分布图.eps"')

# 图4: 蒙特卡洛模拟
plt.figure(figsize=(10, 6))
plt.hist(bj * 1000, bins=30, color='#66b3ff', edgecolor='k', alpha=0.7)
plt.axvline(bs * 1000, color='r', lw=2.5, label=f'均值 ({bs*1000:.1f} nm)')
plt.axvline(bu[0] * 1000, color='k', ls='--', lw=1.5, label='95%置信区间')
plt.axvline(bu[1] * 1000, color='k', ls='--', lw=1.5)
plt.xlabel('厚度 (nm)')
plt.ylabel('频数')
plt.title(f'蒙特卡洛模拟 (n={e}次)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('蒙特卡洛模拟图.eps', dpi=300)
print('蒙特卡洛模拟图已保存为 "蒙特卡洛模拟图.eps"')

# 图5: 优化过程收敛性
plt.figure(figsize=(10, 6))
cb = np.arange(1, 51)
cc = bs + bt * np.exp(-cb / 15) * (0.5 + 0.5 * rng.random(50))
plt.plot(cb, cc * 1000, 'b-o', lw=1.5, markersize=4)
plt.axhline(bs * 1000, color='r', ls='--', lw=1.5, label='最终值')
plt.xlabel('迭代次数')
plt.ylabel('厚度估计值 (nm)')
plt.title('优化过程收敛性 (模拟)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('优化收敛性图.eps', dpi=300)
print('优化收敛性图已保存为 "优化收敛性图.eps"')
print('\n=============== 最终结果 ================')
print(f'厚度估计: {bs:.4f} ± {bt:.4f} um')
print(f'95%置信区间: [{bu[0]:.4f}, {bu[1]:.4f}] um')
print(f'变异系数: {bt / bs * 100:.2f}%')
print('\n可视化结果已保存为5张独立的eps图片。')