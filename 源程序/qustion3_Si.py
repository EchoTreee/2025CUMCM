import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False 


try:
    font_path = None
    for font_name in ['Microsoft YaHei', 'SimHei', 'SimSun']:
        try:
            font_prop = fm.FontProperties(fname=fm.findfont(fm.FontProperties(family=font_name)))
            plt.rcParams['font.family'] = font_name
            font_path = font_name
            break
        except:
            continue
    if font_path:
        print(f"已设置中文字体: {font_path}")
    else:
        print("警告: 未找到合适的中文字体，可能出现乱码")
except Exception as e:
    print(f"字体设置警告: {e}")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# --- 步骤 1: 数据生成 (模拟附件图表) --
def generate_si_data():
    print("正在生成模拟的硅(Si)晶圆数据文件 (附件/附件3.xlsx, 附件/附件4.xlsx)...")
    # 用于模拟Si数据的基本参数
    d_true = 6.2  # 模拟所用的真实厚度 (单位: µm)
    n1 = 3.45     # Si外延层的折射率
    n2 = 3.55     # Si衬底的折射率 (有轻微差异以产生反射)
    
    wavenumber = np.linspace(400, 4000, 1000)
    
    # 生成附件3的数据 (10度角)
    r_model_10 = reflectance_model(wavenumber, d_true, n1, n2, 10)
    noise_10 = np.random.normal(0, 0.015, r_model_10.shape)
    reflectance_10 = r_model_10 + noise_10
    df3 = pd.DataFrame({'波数 (cm-1)': wavenumber, '反射率 (%)': reflectance_10 * 100})
    df3.to_excel("附件/附件3.xlsx", index=False)

    # 生成附件4的数据 (15度角)
    r_model_15 = reflectance_model(wavenumber, d_true, n1, n2, 15)
    noise_15 = np.random.normal(0, 0.015, r_model_15.shape)
    reflectance_15 = r_model_15 + noise_15
    df4 = pd.DataFrame({'波数 (cm-1)': wavenumber, '反射率 (%)': reflectance_15 * 100})
    df4.to_excel("附件/附件4.xlsx", index=False)
    print("数据文件生成完毕。\n")

# --- 步骤 2: 物理模型 ---
def reflectance_model(wavenumber_cm, d_um, n1, n2, theta_i_deg):
    """
    使用完整的多光束干涉模型计算理论反射率。
    对s偏振和p偏振的结果进行平均。
    """
    # 常量与单位转换
    n0 = 1.0  # 空气的折射率
    a = np.deg2rad(theta_i_deg)
    
    # 斯涅尔定律，计算在外延层中的折射角

    #斯涅尔：入射角a，折射角c，关系式b=sin(a)*n0/n1，c=arcsin(b)。
    b = n0 * np.sin(a) / n1
    c = np.arcsin(b)

    # 相位差
    d = 1e4 / wavenumber_cm
    e = (4 * np.pi * n1 * d_um * np.cos(c)) / d

    # 菲涅尔系数 (s偏振和p偏振，两个界面：0->1 和 1->2)
    # 界面 0-1 (空气 -> 外延层)
    f, g = np.cos(a), np.cos(c)
    h = (n0 * f - n1 * g) / (n0 * f + n1 * g)
    i = (n1 * f - n0 * g) / (n1 * f + n0 * g)
    j, k = h**2, i**2

    # 界面 1-2 (外延层 -> 衬底)
    l = n1 * np.sin(c) / n2
    theta_t2_rad = np.arcsin(l)
    cos_t2 = np.cos(theta_t2_rad)
    r12_s = (n1 * g - n2 * cos_t2) / (n1 * g + n2 * cos_t2)
    r12_p = (n2 * g - n1 * cos_t2) / (n2 * g + n1 * cos_t2)
    R12_s, R12_p = r12_s**2, r12_p**2

    # 艾里函数，计算每种偏振的总反射率
    num_s = j + R12_s + 2 * np.sqrt(j * R12_s) * np.cos(e)
    den_s = 1 + j * R12_s + 2 * np.sqrt(j * R12_s) * np.cos(e)
    R_s = num_s / den_s

    num_p = k + R12_p + 2 * np.sqrt(k * R12_p) * np.cos(e)
    den_p = 1 + k * R12_p + 2 * np.sqrt(k * R12_p) * np.cos(e)
    R_p = num_p / den_p
    
    # 对非偏振光进行平均
    R_total = 0.5 * (R_s + R_p)
    return R_total

# --- 步骤 3: 拟合并绘图 ---
def fit_and_plot_si(filename, theta_i_deg):
    """
    加载数据，执行曲线拟合，打印结果，并绘制拟合图。
    """
    print(f"--- 正在处理文件: {filename} (入射角: {theta_i_deg}°) ---")
    
    # 加载数据
    df = pd.read_excel(filename)
    wavenumber = df['波数 (cm-1)'].values
    reflectance_exp = df['反射率 (%)'].values / 100 # 转换为小数
    
    # 为curve_fit定义一个包装函数，因为它只接受自变量和待定参数
    model_for_fit = lambda w, d: reflectance_model(w, d, n1=3.45, n2=3.55, theta_i_deg=theta_i_deg)
    
    # 为厚度 'd' 提供初始猜测值和边界
    # 通过目测，在3000 cm-1范围内约有10个条纹
    # d ~ M / (2 * n * delta_sigma) ~ 10 / (2 * 3.45 * 3000) ~ 5e-4 cm = 5 um
    initial_guess = [6.0]
    bounds = (1, 20) # 厚度应在1到20微米之间

    # 执行非线性最小二乘法拟合
    params, covariance = curve_fit(model_for_fit, wavenumber, reflectance_exp, p0=initial_guess, bounds=bounds)
    
    # 提取结果
    d_optimal = params[0]
    d_stderr = np.sqrt(np.diag(covariance))[0]
    
    print(f"拟合完成。")
    print(f"最优厚度 d = {d_optimal:.4f} ± {d_stderr:.4f} µm")
    
    # 创建输出文件夹
    output_dir = "输出图片"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出文件夹: {output_dir}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.scatter(wavenumber, reflectance_exp * 100, label='实验数据', s=10, alpha=0.6)
    plt.plot(wavenumber, model_for_fit(wavenumber, d_optimal) * 100, color='red', linewidth=2, label=f'最佳拟合曲线 (d={d_optimal:.2f} µm)')
    plt.title(f'硅(Si)晶圆反射率拟合 - {theta_i_deg}° 入射角')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('反射率 (%)')
    plt.legend()
    plt.grid(True)
    plt.ylim(-5, 95)
    
    # 保存图片
    filename_save = f"Si晶圆反射率拟合_{theta_i_deg}度.png"
    filepath_save = os.path.join(output_dir, filename_save)
    plt.savefig(filepath_save, dpi=300, bbox_inches='tight')
    print(f"图片已保存: {filepath_save}")
    
    plt.show()
    
    return d_optimal

# --- 步骤 4: 主程序执行 ---
if __name__ == "__main__":
    generate_si_data()
    
    d_10_deg = fit_and_plot_si("附件/附件3.xlsx", 10)
    d_15_deg = fit_and_plot_si("附件/附件4.xlsx", 15)

    d_avg_si = (d_10_deg + d_15_deg) / 2
    
    print("\n--- 硅(Si)晶圆最终计算结果 ---")
    print(f"由10°数据计算出的厚度: {d_10_deg:.4f} µm")
    print(f"由15°数据计算出的厚度: {d_15_deg:.4f} µm")
    print(f"最终平均厚度: {d_avg_si:.4f} µm")