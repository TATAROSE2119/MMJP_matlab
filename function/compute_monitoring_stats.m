function [T2_stats, Q_stats, T2_limits, Q_limits, Lambda] = compute_monitoring_stats(X_train_modes, X_test, P, mode_labels, alpha)
    C = length(X_train_modes); % 模式数
    n_samples = size(X_test, 1);
    d = size(P, 2); % 低维投影维度
    D = size(P, 1); % 数据维度 (52)
    Lambda = cell(C, 1);
    Y_train = cell(C, 1);
    
    % 为每个模式计算低维投影和协方差矩阵
    for c = 1:C
        Xc = X_train_modes{c}; % 模式 c 的训练数据
        nc = size(Xc, 1); % 模式 c 的样本数
        Y_train{c} = Xc * P; % 低维投影
        Lambda{c} = (Y_train{c}' * Y_train{c}) / (nc - 1); % 协方差矩阵
    end
    
    % 初始化统计量
    T2_stats = zeros(n_samples, 1);
    Q_stats = zeros(n_samples, 1);
    T2_limits = zeros(C, 1);
    Q_limits = zeros(C, 1);
    
    % 计算每个测试样本的 T2 和 Q 统计量
    for i = 1:n_samples
        x_new = X_test(i, :);
        c = mode_labels(i); % 确定样本所属模式
        y_new = P' * x_new'; % 低维投影
        T2_stats(i) = y_new' * inv(Lambda{c}) * y_new; % T^2 统计量
        residual = (eye(D) - P * P') * x_new'; % 残差
        Q_stats(i) = residual' * residual; % Q 统计量
    end
    
    % 为每个模式计算控制限
    for c = 1:C
        Xc = X_train_modes{c};
        nc = size(Xc, 1);
        
        % T^2 控制限 (基于 F 分布)
        T2_limits(c) = (d * (nc - 1)) / (nc - d) * finv(alpha, d, nc - d); % 修正为 1 - alpha
        
        % Q 控制限 (基于加权 chi^2 分布)
        residuals = (eye(D) - P * P') * Xc';
        Q_train = sum(residuals.^2, 1)'; % 每个样本的 Q 值
        theta1 = mean(Q_train); % 均值
        theta2 = var(Q_train); % 方差
        theta3 = sum((Q_train - theta1).^3) / nc / (theta2^(3/2)); % 三阶矩
        h = 1 - (2 * theta1 * theta3) / (3 * theta2^2); % 自由度修正
        g = theta2 / (2 * theta1); % 尺度参数
        Q_limits(c) = g * chi2inv(alpha, h); % Q 控制限
    end
end