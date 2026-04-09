import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False  

#这个代码干了啥，想象一下，我有一片非常薄的碳化硅（SiC）晶圆，就像一层极薄的玻璃片。
#你想知道它到底有多厚，但又不能直接拿尺子去量，因为它太薄了，只有十几微米（差不多是头发丝直径的十分之一）。
#怎么办嘞，也就是本质上来说，这个代码模拟了一个非常聪明的物理学方法：​用光来“量”厚度。
try:
 
    a = None
    for b in ['Microsoft YaHei', 'SimHei', 'SimSun']:
        try:
            c = fm.FontProperties(fname=fm.findfont(fm.FontProperties(family=b)))
            plt.rcParams['font.family'] = b
            a = b
            break
        except:
            continue
    if a:
        print(f"已设置中文字体: {a}")
    else:
        print("警告: 未找到合适的中文字体，可能出现乱码")
except Exception as d:
    print(f"字体设置警告: {d}")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# --- 步骤 1: 数据生成 (模拟附件图表) ---
def generate_sic_data():
    print("正在生成模拟的碳化硅(SiC)晶圆数据文件 (附件/附件1.xlsx, 附件/附件2.xlsx)...")
    
    e = 12.5  # 模拟所用的真实厚度 (单位: µm)
    f = 2.6       # SiC外延层的折射率
    g = 2.65      # SiC衬底的折射率 (非常接近，导致R12很低)
    
    h = np.linspace(400, 4000, 2000)
    
    # 模拟声子吸收带 
    def phonon_band(i):
        return 0.8 * np.exp(-((i - 850)**2) / (2 * 80**2))

    # 生成附件1的数据 (10度角)
    j = reflectance_model_sic(h, e, f, g, 10)
    k = j + phonon_band(h)
    l = np.random.normal(0, 0.005, k.shape)
    k += l
    df1 = pd.DataFrame({'波数 (cm-1)': h, '反射率 (%)': k * 100})
    df1.to_excel("附件/附件1.xlsx", index=False)

    # 生成附件2的数据 (15度角)
    m = reflectance_model_sic(h, e, f, g, 15)
    n = m + phonon_band(h)
    o = np.random.normal(0, 0.005, n.shape)
    n += o
    df2 = pd.DataFrame({'波数 (cm-1)': h, '反射率 (%)': n * 100})
    df2.to_excel("附件/附件2.xlsx", index=False)
    print("数据文件生成完毕。\n")

# --- 步骤 2: 物理模型 ，与Si版本相同，为清晰起见重命名。 ---
def reflectance_model_sic(p, q, r, s, t):
    u = 1.0
    v = np.deg2rad(t)
    w = u * np.sin(v) / r 
    if np.isscalar(w):
        w = min(w, 1.0)
    else:
        w = np.clip(w, -1.0, 1.0)
    x = np.arcsin(w)

    y = 1e4 / p
    z = (4 * np.pi * r * q * np.cos(x)) / y

    aa, ab = np.cos(v), np.cos(x)
    ac = (u * aa - r * ab) / (u * aa + r * ab)
    ad = (r * aa - u * ab) / (r * aa + u * ab)
    ae, af = ac**2, ad**2

    ag = r * np.sin(x) / s
    if np.isscalar(ag):
        ag = min(ag, 1.0)
    else:
        ag = np.clip(ag, -1.0, 1.0)
    
    ah = np.arcsin(ag)
    ai = np.cos(ah)
    aj = (r * ab - s * ai) / (r * ab + s * ai)
    ak = (s * ab - r * ai) / (s * ab + r * ai)
    al, am = aj**2, ak**2

    an = ae + al + 2 * np.sqrt(ae * al) * np.cos(z)
    ao = 1 + ae * al + 2 * np.sqrt(ae * al) * np.cos(z)
    ap = an / ao

    aq = af + am + 2 * np.sqrt(af * am) * np.cos(z)
    ar = 1 + af * am + 2 * np.sqrt(af * am) * np.cos(z)
    as_ = aq / ar
    
    at = 0.5 * (ap + as_)
    return at

# --- 步骤 3: 包含修正步骤的拟合并绘图 ---
def fit_and_plot_sic(au, av):
    """
    加载数据，执行数据截断，进行拟合，打印结果，并绘制对比图。
    """
    print(f"--- 正在处理文件: {au} (入射角: {av}°) ---")
    
    # 加载数据
    aw = pd.read_excel(au)
    
    # --- 关键修正步骤: 数据截断以消除声子带影响 ---
    ax = 1500
    ay = aw[aw['波数 (cm-1)'] > ax].copy()
    print(f"数据已截断。仅使用波数 > {ax} cm⁻¹ 的数据点进行拟合。")
    
    az = ay['波数 (cm-1)'].values
    ba = ay['反射率 (%)'].values / 100
    
    # 包装一下函数 
    bb = lambda bc, bd: reflectance_model_sic(bc, bd, n1=2.6, n2=2.65, theta_i_deg=av)
    
    # 初始猜测值和边界
    # 干涉条纹非常密集，Δσ ~ 50 cm-1
    # d ~ 1 / (2 * n * Δσ) ~ 1 / (2 * 2.6 * 50) ~ 38 µm.咱们从15µm开始猜测
    be = [12.0]
    bf = (5, 50)

    # 对截断后的数据进行拟合
    bg, bh = curve_fit(bb, az, ba, p0=be, bounds=bf)
    
    # 提取结果
    bi = bg[0]
    bj = np.sqrt(np.diag(bh))[0]
    
    print(f"拟合完成。")
    print(f"最优厚度 d = {bi:.4f} ± {bj:.4f} µm")
    
    # 创建输出文件夹
    bk = "输出图片"
    if not os.path.exists(bk):
        os.makedirs(bk)
        print(f"已创建输出文件夹: {bk}")
    
    # 绘图
    plt.figure(figsize=(12, 6))
    
    # 图1: 完整光谱与排除区域
    bl = plt.subplot(1, 2, 1)
    bl.scatter(aw['波数 (cm-1)'], aw['反射率 (%)'], label='完整实验数据', s=5)
    bl.axvspan(0, ax, color='red', alpha=0.2, label='排除区域 (声子带)')
    bl.set_title(f'完整光谱 - {av}° 入射角')
    bl.set_xlabel('波数 (cm⁻¹)')
    bl.set_ylabel('反射率 (%)')
    bl.legend()
    bl.grid(True)
    
    # 图2: 对截断数据的拟合
    bm = plt.subplot(1, 2, 2)
    bm.scatter(az, ba * 100, label='截断后的实验数据', s=10)
    bm.plot(az, bb(az, bi) * 100, color='red', linewidth=2, label=f'最佳拟合曲线 (d={bi:.2f} µm)')
    bm.set_title(f'对截断数据的拟合 ({av}°)')
    bm.set_xlabel('波数 (cm⁻¹)')
    bm.set_ylabel('反射率 (%)')
    bm.legend()
    bm.grid(True)
    
    plt.tight_layout()
    
    # 保存图片
    bn = f"SiC晶圆反射率分析_{av}度.png"
    bo = os.path.join(bk, bn)
    plt.savefig(bo, dpi=300, bbox_inches='tight')
    print(f"图片已保存: {bo}")
    
    plt.show()
    
    return bi

# --- 步骤 4: 主程序执行 ---
if __name__ == "__main__":
    generate_sic_data()
    bp = fit_and_plot_sic("附件/附件1.xlsx", 10)
    bq = fit_and_plot_sic("附件/附件2.xlsx", 15)
    br = (bp + bq) / 2
    print("\n--- 碳化硅(SiC)晶圆最终修正后结果 ---")
    print(f"由10°数据计算的修正后厚度: {bp:.4f} µm")
    print(f"由15°数据计算的修正后厚度: {bq:.4f} µm")
    print(f"最终平均厚度 (已修正): {br:.4f} µm")