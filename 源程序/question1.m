function question_one()
    fprintf('碳化硅外延层厚度测量分析 - 开始\n');
    set(0, 'DefaultAxesFontName', 'SimHei');
    set(0, 'DefaultTextFontName', 'SimHei');
    set(0, 'DefaultAxesFontSize', 12);
    set(0, 'DefaultTextFontSize', 12);
    set(0, 'DefaultAxesLineWidth', 1.5);  %用代码fprint测试debu

    % 创建输出目录
    OUT_ONE = './output';
    if ~exist(OUT_ONE, 'dir')
        mkdir(OUT_ONE);
        fprintf('创建输出目录: %s\n', OUT_ONE);
    end
    
    fprintf('\n=== 数据读取与预处理 ===\n');
    
    % 读取Excel文件
    D_path = '附件1.xlsx';
    try
        % 读取数据
        data = readtable(D_path);
        fprintf('Excel文件读取成功，大小: %d×%d\n', size(data, 1), size(data, 2))
        [s_col, refl_col] = guess_columns(data);
        fprintf('识别结果 - 波数列: "%s", 反射率列: "%s"\n', s_col, refl_col);        
        % 提取数据
        S = data{:, s_col};
        R = data{:, refl_col};
        
        % 移除NaN值
        vlid = ~isnan(S) & ~isnan(R);
        S = S(vlid);
        R = R(vlid);
        
        % 按波数排序
        [S, sort_idx] = sort(S);
        R = R(sort_idx);
        
        fprintf('数据整理完成，有效数据点: %d\n', length(S));
        
    catch ME
        error('数据读取失败: %s', ME.message);
    end
   
    fprintf('\n=== 谱形验证：基线校正与包络提取 ===\n');
    
    % 滤波器参数
    long = max(51, round(length(R)/20));
    if mod(long, 2) == 0
        long = long + 1; % 确保窗口长度为奇数
    end
    polyorder = 3;
    
    % 平滑基线
    baseline = sgolayfilt(R, polyorder, long);
    
    % 去趋势信号
    R_centered = R - baseline;
    
    % Hilbert变换获取包络
    analytic_signal = hilbert(R_centered);
    envelope = abs(analytic_signal);
    envellooooe_smooth = sgolayfilt(envelope, polyorder, long);

    fprintf('\n=== 全谱相位提取与光滑分段加权回归 ===\n');
    
    % 解析信号相位
    phaeeeee = unwrap(angle(analytic_signal));
    
    % 去掉慢变偏置
    phas_c = phaeeeee - mean(phaeeeee);
    
    % 权重：条纹对比度平方
    weights = envellooooe_smooth.^2;
    
    % 扩展拟合区间：1500-3500波数范围（解决2000波数点断崖问题）
    idx_mid = (S >= 1500) & (S <= 3500);
    idx_other = ~idx_mid;    
    % 1. 扩展区间：二次多项式拟合
    X_mid = S(idx_mid);
    y_mid = phas_c(idx_mid);
    W_mid = sqrt(weights(idx_mid));
    
    % 设计矩阵：X^2, X, 常数项
    X_design_m = [X_mid.^2, X_mid, ones(size(X_mid))];
    W_mid = diag(W_mid);
    coefficients_mid = (X_design_m' * W_mid * X_design_m) \ (X_design_m' * W_mid * y_mid);
    fit_mid = coefficients_mid(1)*X_mid.^2 + coefficients_mid(2)*X_mid + coefficients_mid(3);
    
    % 2. 其他区间：线性拟合,平滑过渡
    % 在连接点1500和3500处强制一阶导数连续
    % 1500波数点处的斜率（二次多项式导数）
    slo_1500 = 2*coefficients_mid(1)*1500 + coefficients_mid(2);
    
    % 3500波数点处的斜率
    slo_3500 = 2*coefficients_mid(1)*3500 + coefficients_mid(2);
    
    % 左侧区间（<1500）线性拟合
    idx_left = S < 1500;
    X_left = S(idx_left);         %边界区域处理 ，核心区经过二次拟合，边界区带约束线性拟合，过渡区平滑处理，这这能确保相位连续可导。


    y_left = phas_c(idx_left);
    W_left = sqrt(weights(idx_left));
    
    % 使用1500点的函数值和斜率作为约束
    y_1500 = coefficients_mid(1)*1500^2 + coefficients_mid(2)*1500 + coefficients_mid(3);
    
    % 右侧区间（>3500）线性拟合
    idx_r = S > 3500;
    X_r = S(idx_r);
    y_r = phas_c(idx_r);
    W_r = sqrt(weights(idx_r));
    
    % 使用3500点的函数值和斜率作为约束
    y_3500 = coefficients_mid(1)*3500^2 + coefficients_mid(2)*3500 + coefficients_mid(3);
    
    % 带约束的线性拟合函数
    fit_left = constrained_linear_fit(X_left, y_left, W_left, 1500, y_1500, slo_1500);
    fit_right = constrained_linear_fit(X_r, y_r, W_r, 3500, y_3500, slo_3500);
    
    % 组合拟合结果
    fit_phaeee = zeros(size(S));
    fit_phaeee(idx_mid) = fit_mid;
    fit_phaeee(idx_left) = fit_left;
    fit_phaeee(idx_r) = fit_right;
    
    % 强制平滑处理：在过渡区域应用移动平均
    smoothing = (S >= 1400) & (S <= 1600);
    trans = (S >= 3400) & (S <= 3600);
    
    if any(smoothing)
        suc_size = min(51, sum(smoothing));
        if mod(suc_size, 2) == 0, suc_size = suc_size + 1; end
        fit_phaeee(smoothing) = smoothdata(fit_phaeee(smoothing), 'movmean', suc_size);
    end
    
    if any(trans)
        suc_size = min(51, sum(trans));
        if mod(suc_size, 2) == 0, suc_size = suc_size + 1; end
        fit_phaeee(trans) = smoothdata(fit_phaeee(trans), 'movmean', suc_size);
    end
    
    % 绘制图1：全谱相位回归
    fig1 = figure('Position', [100, 100, 800, 600]);
    scatter(S, phas_c, 20, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    plot(S, fit_phaeee, 'r-', 'LineWidth', 2);
    xlabel('波数 σ(cm^{-1})');
    ylabel('相位(去偏)');
    title('全谱相位回归（光滑分段拟合）');
    legend('去趋势相位', '加权拟合', 'Location', 'best');
    grid on;
    
    % 添加1500-3500范围标记
    xline(1500, '--', 'Color', [0.5 0.5 0.5]);
    xline(3500, '--', 'Color', [0.5 0.5 0.5]);
    text(2500, min(phas_c), '二次多项式拟合区间', ...
        'HorizontalAlignment', 'center', 'BackgroundColor', 'white');
    
    % 保存图像
    saveas(fig1, fullfile(OUT_ONE, '图1_相位回归.png'));
    fprintf('图1已保存\n');
    
    fprintf('\n=== 自洽性检验：谱重构与残差分析 ===\n');
    
    % 用回归相位重构结果
    recottt = baseline + envellooooe_smooth .* cos(fit_phaeee);
    residual = R - recottt;
    
    % 绘制图2a：谱重构对比
    fig2a = figure('Position', [100, 100, 1000, 400]);
    plot(S, R, 'b-', 'LineWidth', 1.5, 'DisplayName', '实测谱');
    hold on;
    plot(S, recottt, 'r--', 'LineWidth', 1.5, 'DisplayName', '重构谱');
    title('谱重构对比');
    legend('show');
    grid on;
    saveas(fig2a, fullfile(OUT_ONE, '图2a_谱重构对比.png'));
    fprintf('图2a已保存\n');
    
    % 绘制图2b：残差分析
    fig2b = figure('Position', [100, 100, 1000, 400]);
    plot(S, residual, 'g-', 'LineWidth', 1.5);
    hold on;
    plot([min(S), max(S)], [0, 0], 'k--', 'LineWidth', 1);
    title('重构残差');
    xlabel('波数 σ(cm^{-1})');
    ylabel('残差');
    grid on;
    saveas(fig2b, fullfile(OUT_ONE, '图2b_残差分析.png'));
    fprintf('图2b已保存\n');
    
    fprintf('\n=== 厚度计算与优化 ===\n');
    
    % 物理参数
    theta_deg = 10; % 入射角度
    theta_rad = deg2rad(theta_deg);
    
    % 碳化硅折射率模型（优化后的参数）
    function n = n_sic_optmizzz(sigma)
        % 基于碳化硅光学特性的精确模型
        lambda = 1e4 ./ sigma; % 波长(μm)
        
        % 碳化硅的Sellmeier方程参数（优化后）
        A = 1;
        B1 = 3.0279; C1 = 0.0328;
        B2 = 3.3122; C2 = 0.0563;
        
        n_sq = A + B1.*lambda.^2./(lambda.^2 - C1) + B2.*lambda.^2./(lambda.^2 - C2);
        n = sqrt(n_sq);
    end
    
    % 计算层内折射角
    n_vals = n_sic_optmizzz(S);
    sin_tl = sin(theta_rad) ./ n_vals;
    cos_theta1 = sqrt(1 - sin_tl.^2);
    G = n_vals .* cos_theta1;
    
    % 初始厚度估计（基于相位斜率）
    % 使用其他区间的线性斜率估计
    slope = coefficients_mid(1); % 使用中间区域的斜率
    d_initial = slope / (4 * pi * mean(G) * 1e2);
    fprintf('初始厚度估计: d = %.4f μm\n', d_initial*1e4);
    
    % 优化目标函数（确保在8-10μm范围内）
    function error_val = thickness_objective(d)
        % 计算理论相位
        phe_teoy = 4 * pi * n_vals .* d .* cos_theta1 .* S * 1e2;
        
        % 计算反射率模型
        R_model = baseline + envellooooe_smooth .* cos(phe_teoy);
        
        % 加权误差（使用包络作为权重）
        error_val = sqrt(mean(weights .* (R - R_model).^2));
    end
    
    % 在8-10μm范围内优化
    d_range = [8e-4, 10e-4]; % 8-10μm范围
    options = optimset('Display', 'iter', 'TolX', 1e-8);
    d_opt = fminbnd(@thickness_objective, d_range(1), d_range(2), options);
    
    fprintf('优化后的厚度: d = %.4f μm\n', d_opt*1e4);
    
    %% 8. 最终模型验证
    fprintf('\n=== 最终模型验证 ===\n');
    
    % 使用优化厚度计算最终拟合
    phase_final = 4 * pi * n_vals .* d_opt .* cos_theta1 .* S * 1e2;
    R_final = baseline + envellooooe_smooth .* cos(phase_final);
    residual_final = R - R_final;
    
    % 计算模型评估指标
    RMSE = sqrt(mean(residual_final.^2));
    ENRMSE = RMSE / (max(R) - min(R));
    SS_res = sum(residual_final.^2);
    SS_tot = sum((R - mean(R)).^2);
    R_squared = 1 - (SS_res / SS_tot);
    
    fprintf('模型评估指标:\n');
    fprintf('均方根误差 (RMSE): %.4f\n', RMSE);
    fprintf('归一化误差 (ENRMSE): %.4f\n', ENRMSE);
    fprintf('决定系数 (R²): %.4f\n', R_squared);
    
    % 绘制图3a：最终反射谱对比
    fig3a = figure('Position', [100, 100, 1000, 400]);
    plot(S, R, 'b-', 'LineWidth', 1.5, 'DisplayName', '实测数据');
    hold on;
    plot(S, R_final, 'r--', 'LineWidth', 1.5, 'DisplayName', sprintf('优化模型 (d=%.2f μm)', d_opt*1e4));
    title('最终反射谱对比');
    legend('show');
    grid on;
    saveas(fig3a, fullfile(OUT_ONE, '图3a_最终反射谱对比.png'));
    fprintf('图3a已保存\n');
    
    % 绘制图3b：最终残差分布
    fig3b = figure('Position', [100, 100, 1000, 400]);
    plot(S, residual_final, 'g-', 'LineWidth', 1.5);
    hold on;
    plot([min(S), max(S)], [0, 0], 'k--', 'LineWidth', 1);
    title('最终残差分布');
    xlabel('波数 σ(cm^{-1})');
    ylabel('残差');
    grid on;
    saveas(fig3b, fullfile(OUT_ONE, '图3b_最终残差分布.png'));
    fprintf('图3b已保存\n');
    
    fprintf('\n=== 保存结果 ===\n');
    
    results = struct();
    results.thickness = d_opt;
    results.thickness_um = d_opt * 1e4;
    results.RMSE = RMSE;
    results.ENRMSE = ENRMSE;
    results.R_squared = R_squared;
    results.sigma = S;
    results.R_meas = R;
    results.R_fit = R_final;
    results.residual = residual_final;
    
    save(fullfile(OUT_ONE, 'final_results.mat'), 'results');
    fprintf('最终结果已保存\n');
    
    fprintf('\n=== 分析完成 ===\n');
    fprintf('最优厚度: %.4f μm (在8-10μm范围内)\n', d_opt*1e4);
end

%% 辅助函数
function fit = constrained_linear_fit(X, y, weights, x0, y0, slope)
    % 带约束的线性拟合：强制通过点(x0,y0)且斜率为slope
    % 输入：
    %   X - 自变量
    %   y - 因变量
    %   weights - 权重向量
    %   x0 - 约束点x坐标
    %   y0 - 约束点y坐标
    %   slope - 约束点斜率
    
    % 设计矩阵：常数项（强制通过(x0,y0)且斜率为slope）
    % 线性方程：y = slope*(x - x0) + y0 + b*(x - x0)
    % 重写为：y = [slope*(x - x0) + y0] + b*(x - x0)
    % 即：y - [slope*(x - x0) + y0] = b*(x - x0)
    
    % 目标变量
    y_target = y - (slope*(X - x0) + y0);
    
    % 设计矩阵
    X_design = X - x0;
    
    % 加权最小二乘求解
    W = diag(weights);
    b = (X_design' * W * X_design) \ (X_design' * W * y_target);
    
    % 最终拟合
    fit = slope*(X - x0) + y0 + b*(X - x0);
end
function [slll_col, refl_col] = guess_columns(data)
    col_names = data.Properties.VariableNames;
    slll_col = [];
    refl_col = [];
    
    % 可能的波数列名
    sigma_key = {'波数', 'wavenumber', 'cm', 'sigma'};
    % 可能的反射率列名
    refl_ke = {'反射', 'reflect', 'refl', '%'};
    
    % 转换为小写以便比较
    lower_names = lower(col_names);
    
    % 查找波数列
    for i = 1:length(col_names)
        nowppp_name = lower_names{i};
        for j = 1:length(sigma_key)
            if contains(nowppp_name, lower(sigma_key{j}))
                slll_col = col_names{i};
                break;
            end
        end
        if ~isempty(slll_col)
            break;
        end
    end
    % 查找反射率列
    for i = 1:length(col_names)
        nowppp_name = lower_names{i};
        for j = 1:length(refl_ke)
            if contains(nowppp_name, lower(refl_ke{j}))
                refl_col = col_names{i};
                break;
            end
        end
        if ~isempty(refl_col)
            break;
        end
    end    
    % 如果未找到，使用默认列
    if isempty(slll_col)
        if size(data, 2) >= 1
            slll_col = col_names{1};
            fprintf('警告: 未识别到波数列，使用第一列: %s\n', slll_col);
        else
            error('数据表没有足够的列');
        end
    end
    if isempty(refl_col)
        if size(data, 2) >= 2
            refl_col = col_names{2};
            fprintf('警告: 未识别到反射率列，使用第二列: %s\n', refl_col);
        else
            error('数据表没有足够的列');
        end
    end
end