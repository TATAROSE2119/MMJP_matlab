%% 主程序
clear; clc; close all;
rng(42); % 设置随机种子

%% 加载 TE 数据集
% 加载训练数据 (d00.dat: 正常运行模式, 500 个样本)
train_data = load('d00.dat'); % 假设 53 列（第 1 列为时间戳）
train_data=train_data';
%train_data = train_data(:, 2:53); % 去除时间戳，保留 52 个变量

% 加载测试数据 (d02_te.dat: 故障 2 数据, 960 个样本)
test_data = load('d06_te.dat'); % 假设 53 列（第 1 列为时间戳）
%test_data = test_data(:, 2:53); % 去除时间戳，保留 52 个变量

% 使用所有 52 个变量
X = train_data; % 训练数据 (500 x 52)
X_test = test_data; % 测试数据 (960 x 52)

% 单模态数据，假设训练集为单一模式
X_modes = {X}; % 单一模式，训练数据

%% 主程序执行 - 代表点计算（单模态简化）
% 由于是单模态，跳过多模态谱聚类，直接使用训练数据计算代表点
true_labels = ones(500, 1); % 单一模式标签
exemplars = compute_exemplars(X, true_labels);

% 在线模式识别（测试数据）
plot_mode_identification(X_test, 2, exemplars); % 故障 2

%% 训练 W_intra 和 P
% 参数设置
K = 10; t1 = 1.0; t2 = 2.0; t3 = 1.0; d = 9; % d 设为 9
mu = 10; lambda = 0.01; max_iter = 70; tol = 1e-5; eta = 0.1; % 调整后的参数
alpha = 1; beta = 1; % 固定值

% 初始化
W_intra = cell(1, 1); % 单模态，仅一个 W_intra
[W_D, W_A] = compute_intra_adjacency_matrix(X_modes{1}, K, t1, t2);
W_intra{1} = alpha * W_D + beta * W_A;

W_global = compute_global_affinity_dynamic(X_modes, W_intra, K, t3);
L = compute_laplacian(W_global);
D = size(X, 2); % 数据维度 (D = 52)
P = randn(D, d); % 随机初始 P (52 x 9)
P = orth(P); % 正交化

% 记录收敛过程
J_history = zeros(max_iter, 1);
J_fit_history = zeros(max_iter, 1);
J_mmjp_history = zeros(max_iter, 1);
J_hsic_history = zeros(max_iter, 1);

% 迭代优化
J_prev = inf;
for iter = 1:max_iter
    % 步骤 1：优化 W_intra
    W_intra_new = cell(1, 1);
    [W_D, W_A] = compute_intra_adjacency_matrix(X_modes{1}, K, t1, t2);
    W_fit = alpha * W_D + beta * W_A;
    n = size(X_modes{1}, 1);
    H = eye(n) - ones(n, n) / n;
    K_D = W_D' * W_D; K_A = W_A' * W_A;
    K = H * K_D * H + H * K_A * H;
    A = eye(n); B = lambda * K;
    C = W_fit + mu * (X_modes{1} * P) * (X_modes{1} * P)';
    W_intra_new{1} = lyap(A, B, -C);
    W_intra_new{1} = (W_intra_new{1} + W_intra_new{1}') / 2;
    W_intra{1} = (1 - eta) * W_intra{1} + eta * W_intra_new{1};
    
    % 步骤 2：优化 P
    W_global = compute_global_affinity_dynamic(X_modes, W_intra, K, t3);
    L = compute_laplacian(W_global);
    M = X' * L * X;
    [V, D] = eig(M);
    [eigenvalues, idx] = sort(diag(D));
    P_new = V(:, idx(1:d));
    P_new = orth(P_new);
    
    % 计算目标函数并记录
    [J, J_fit, J_mmjp, J_hsic] = objective_function(X_modes, W_intra, P_new, mu, lambda, K, t1, t2, t3);
    J_history(iter) = J;
    J_fit_history(iter) = J_fit;
    J_mmjp_history(iter) = J_mmjp;
    J_hsic_history(iter) = J_hsic;
    
    % 检查收敛
    if abs(J_prev - J) < tol
        fprintf('Converged at iteration %d\n', iter);
        break;
    end
    P = P_new; J_prev = J;
    fprintf('Iteration %d: J = %.4f, J_fit = %.4f, J_mmjp = %.4f, J_hsic = %.4f\n', iter, J, J_fit, J_mmjp, J_hsic);
end

% 输出结果
disp('Optimized P:'); disp(P);

% 绘制收敛过程
iters = 1:iter; % 实际迭代次数
figure;
subplot(2, 2, 1);
plot(iters, J_history(1:iter), 'b-', 'LineWidth', 1.5);
title('Total Objective Function J'); xlabel('Iteration'); ylabel('J'); grid on;

subplot(2, 2, 2);
plot(iters, J_fit_history(1:iter), 'r-', 'LineWidth', 1.5);
title('J_{fit}'); xlabel('Iteration'); ylabel('J_{fit}'); grid on;

subplot(2, 2, 3);
plot(iters, J_mmjp_history(1:iter), 'g-', 'LineWidth', 1.5);
title('J_{mmjp}'); xlabel('Iteration'); ylabel('J_{mmjp}'); grid on;

subplot(2, 2, 4);
plot(iters, J_hsic_history(1:iter), 'm-', 'LineWidth', 1.5);
title('J_{hsic}'); xlabel('Iteration'); ylabel('J_{hsic}'); grid on;

% 调整图形布局
sgtitle('Convergence Process of Objective Function Components');

%% 可视化结果
figure; imagesc(W_global); colorbar; title('W_{global}');
figure; imagesc(L); colorbar; title('Laplacian L');

% 低维投影（仅显示前两个维度）
Y = X * P; % 500 x 9
figure;
scatter(Y(:, 1), Y(:, 2), 50, 'r+', 'DisplayName', 'Normal Mode');
title('Low-dimensional Projection (First Two Dimensions)'); xlabel('Dim 1'); ylabel('Dim 2'); legend; grid on;

%% 测试
X_train_modes = {X}; % 单模态
mode_labels_test = ones(960, 1); % 单一模式标签
alpha_conf = 0.01;
[T2_test, Q_test, T2_limits, Q_limits, Lambda] = compute_monitoring_stats(X_train_modes, X_test, P, mode_labels_test, alpha_conf);

figure;
subplot(2, 1, 1); plot(T2_test, 'b-', 'LineWidth', 1.5); hold on;
plot([1 960], [T2_limits(1) T2_limits(1)], 'r--', 'LineWidth', 1.5);
plot([160 160], ylim, 'k--'); % 故障 2 在样本 161 开始
title('T^2 (Fault 2)'); ylabel('T^2'); xlabel('Sample Index'); legend('T^2', 'Control Limit'); grid on; hold off;
subplot(2, 1, 2); plot(Q_test, 'b-', 'LineWidth', 1.5); hold on;
plot([1 960], [Q_limits(1) Q_limits(1)], 'r--', 'LineWidth', 1.5);
plot([160 160], ylim, 'k--'); % 故障 2 在样本 161 开始
title('Q (Fault 2)'); ylabel('Q'); xlabel('Sample Index'); legend('Q', 'Control Limit'); grid on; hold off;

%% 函数定义
function exemplars = compute_exemplars(X, labels)
    unique_labels = unique(labels);
    exemplars = [];
    for mode = unique_labels'
        mode_data = X(labels == mode, :);
        distances = pdist2(mode_data, mode_data).^2;
        distance_sums = sum(distances, 2);
        [~, idx] = min(distance_sums);
        exemplars = [exemplars; mode_data(idx, :)];
    end
end

function W_global = compute_global_affinity_dynamic(X_modes, W_intra, K, t3)
    C = length(X_modes);
    N = sum(cellfun(@(x) size(x, 1), X_modes));
    W_global = zeros(N, N);
    offsets = [0 cumsum(cellfun(@(x) size(x, 1), X_modes(1:end-1)))];
    for c = 1:C
        rc = offsets(c) + 1 : offsets(c) + size(X_modes{c}, 1);
        W_global(rc, rc) = W_intra{c};
        for p = 1:C
            if c ~= p
                rp = offsets(p) + 1 : offsets(p) + size(X_modes{p}, 1);
                W_inter = compute_intermode_adjacency(X_modes{c}, X_modes{p}, K, t3);
                W_global(rc, rp) = W_inter;
                W_global(rp, rc) = W_inter';
            end
        end
    end
end

function W_inter = compute_intermode_adjacency(Xc, Xp, K, t3)
    nc = size(Xc, 1);
    np = size(Xp, 1);
    W_inter = zeros(nc, np);
    dist_matrix = pdist2(Xc, Xp, 'euclidean');
    [~, idx_cp] = sort(dist_matrix, 2);
    knn_cp = idx_cp(:, 1:K);
    [~, idx_pc] = sort(dist_matrix', 2);
    knn_pc = idx_pc(:, 1:K);
    for i = 1:nc
        neighbors_cp = knn_cp(i, :);
        for j = neighbors_cp
            W_inter(i, j) = exp(-(dist_matrix(i, j)^2) / t3);
        end
    end
    for j = 1:np
        neighbors_pc = knn_pc(j, :);
        for i = neighbors_pc'
            if W_inter(i, j) == 0
                W_inter(i, j) = exp(-(dist_matrix(i, j).^2) / t3);
            end
        end
    end
end

function [W_D, W_A] = compute_intra_adjacency_matrix(X, K, t1, t2)
    [n_samples, ~] = size(X);
    distance_matrix = pdist2(X, X, 'euclidean');
    W_D = zeros(n_samples, n_samples);
    W_A = zeros(n_samples, n_samples);
    [~, idx] = sort(distance_matrix, 2);
    knn_idx = idx(:, 2:K+1);
    for i = 1:n_samples
        neighbors = knn_idx(i, :);
        for j = 1:n_samples
            if any(j == neighbors) || any(i == knn_idx(j, :))
                W_D(i, j) = exp(-(distance_matrix(i, j)^2) / t1);
            elseif ~any(j == neighbors) && ~any(i == knn_idx(j, :))
                W_D(i, j) = exp(-(distance_matrix(i, j)^2) / t2);
            end
        end
    end
    for k = 1:n_samples
        neighbors_k = knn_idx(k, :);
        for i = 1:length(neighbors_k)
            for j = i+1:length(neighbors_k)
                ni = neighbors_k(i);
                nj = neighbors_k(j);
                vec_ik = X(ni, :) - X(k, :);
                vec_jk = X(nj, :) - X(k, :);
                norm_ik = norm(vec_ik);
                norm_jk = norm(vec_jk);
                if norm_ik > 0 && norm_jk > 0
                    cos_angle = dot(vec_ik, vec_jk) / (norm_ik * norm_jk);
                    W_A(ni, nj) = cos_angle;
                    W_A(nj, ni) = cos_angle;
                end
            end
        end
    end
end

function L = compute_laplacian(W_global)
    D = diag(sum(W_global, 2));
    L = D - W_global;
end

function [T2_stats, Q_stats, T2_limits, Q_limits, Lambda] = compute_monitoring_stats(X_train_modes, X_test, P, mode_labels, alpha)
    C = length(X_train_modes);
    n_samples = size(X_test, 1);
    d = size(P, 2);
    D = size(P, 1);
    Lambda = cell(C, 1);
    Y_train = cell(C, 1);
    for c = 1:C
        Xc = X_train_modes{c};
        nc = size(Xc, 1);
        Y_train{c} = Xc * P;
        Lambda{c} = (Y_train{c}' * Y_train{c}) / (nc - 1);
    end
    T2_stats = zeros(n_samples, 1);
    Q_stats = zeros(n_samples, 1);
    T2_limits = zeros(C, 1);
    Q_limits = zeros(C, 1);
    for i = 1:n_samples
        x_new = X_test(i, :);
        c = mode_labels(i);
        y_new = P' * x_new';
        T2_stats(i) = y_new' * inv(Lambda{c}) * y_new;
        residual = (eye(D) - P * P') * x_new';
        Q_stats(i) = residual' * residual;
    end
    for c = 1:C
        nc = size(X_train_modes{c}, 1);
        T2_limits(c) = (d * (nc - 1)) / (nc - d) * finv(1 - alpha, d, nc - d);
        residuals = (eye(D) - P * P') * X_train_modes{c}';
        Q_train = sum(residuals.^2, 1)';
        theta1 = mean(Q_train);
        theta2 = var(Q_train);
        theta3 = sum((Q_train - theta1).^3) / nc / (theta2^(3/2));
        h = 1 - (2 * theta1 * theta3) / (3 * theta2^2);
        g = theta2 / (2 * theta1);
        Q_limits(c) = g * chi2inv(1 - alpha, h);
    end
end

function [J, J_fit, J_mmjp, J_hsic] = objective_function(X_modes, W_intra, P, mu, lambda, K, t1, t2, t3)
    C = length(X_modes);
    X = vertcat(X_modes{:});
    W_global = compute_global_affinity_dynamic(X_modes, W_intra, K, t3);
    L = compute_laplacian(W_global);
    J_mmjp = trace(P' * X' * L * X * P);
    J_fit = 0; J_hsic = 0;
    for c = 1:C
        [W_D, W_A] = compute_intra_adjacency_matrix(X_modes{c}, K, t1, t2);
        W_fit = W_D + W_A; % alpha = 1, beta = 1
        J_fit = J_fit + norm(W_intra{c} - W_fit, 'fro')^2;
        H = eye(size(W_intra{c}, 1)) - ones(size(W_intra{c}, 1)) / size(W_intra{c}, 1);
        K_D = W_D' * W_D; K_A = W_A' * W_A;
        J_hsic = J_hsic + trace(W_intra{c} * H * K_D * H * W_intra{c}') + trace(W_intra{c} * H * K_A * H * W_intra{c}');
    end
    J = J_fit + mu * J_mmjp + lambda * J_hsic;
end

function plot_mode_identification(test_case, case_num, exemplars)
    n_samples = size(test_case, 1);
    mode_ids = zeros(n_samples, 1);
    for i = 1:n_samples
        x_new = test_case(i, :); % 使用所有 52 个变量
        distances = sum((exemplars - x_new).^2, 2);
        [~, mode] = min(distances);
        mode_ids(i) = mode;
    end
    figure;
    plot(1:n_samples, mode_ids, 'Color', [0.5 0 0.5], 'LineWidth', 2);
    title(['Online Identification (Fault ' num2str(case_num) ')']);
    xlabel('Sample index'); ylabel('Mode'); ylim([0 2]); grid on;
end