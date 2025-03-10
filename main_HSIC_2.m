%% 主程序
clear; clc; close all;
rng(42); % 设置随机种子

%% 生成训练和测试数据
train_mode1 = generate_data(1, 200);
train_mode2 = generate_data(2, 200);
train_mode3 = generate_data(3, 200);
train_data = [train_mode1; train_mode2; train_mode3];

test_mode1 = generate_data(1, 200);
test_mode2 = generate_data(2, 200);
test_mode2(:,1) = test_mode2(:,1) + 0.08;
test_case1 = [test_mode1; test_mode2];

test_case2 = [];
test_case2 = [test_case2; generate_data(1, 200)];
for i = 201:400
    x = generate_data(2, 1);
    x(1) = x(1) + 0.002*(i-100);
    test_case2 = [test_case2; x];
end
test_case2 = [test_case2; generate_data(3, 100)];
test_mode3_fault = generate_data(3, 100);
test_mode3_fault(:,1) = test_mode3_fault(:,1) + 0.08;
test_case2 = [test_case2; test_mode3_fault];

%% 主程序执行 - 聚类
X_vis = train_data(:,1:3);
true_labels = [ones(200,1); 2*ones(200,1); 3*ones(200,1)];
clusters = multimanifold_spectral_clustering(X_vis, 3, 10);
plot_3d_clustering(X_vis, clusters, true_labels);
exemplars = compute_exemplars(X_vis, true_labels);
plot_mode_identification(test_case1(:,1:3), 1, exemplars);
plot_mode_identification(test_case2(:,1:3), 2, exemplars);

%% 训练 W_intra,c 和 P
X_vis_1 = train_mode1(:, 1:3);
X_vis_2 = train_mode2(:, 1:3);
X_vis_3 = train_mode3(:, 1:3);
X_modes = {X_vis_1, X_vis_2, X_vis_3};
X = [X_vis_1; X_vis_2; X_vis_3];

% 参数设置
K = 10; t1 = 1.0; t2 = 2.0; t3 = 1.0; d = 2;
mu = 10; lambda = 0.01; max_iter = 50; tol = 100000; eta = 0.1; % 调整后的参数
alpha = 1; beta = 1; % 固定值

% 初始化
W_intra = cell(3, 1);
for c = 1:3
    [W_D, W_A] = compute_intra_adjacency_matrix(X_modes{c}, K, t1, t2);
    W_intra{c} = alpha * W_D + beta * W_A;
end
W_global = compute_global_affinity_dynamic(X_modes, W_intra, K, t3);
L = compute_laplacian(W_global);
D = size(X, 2); % 数据维度 (D = 3)
P = randn(D, d); % 随机初始 P
P = orth(P); % 正交化

% 记录收敛过程
J_history = zeros(max_iter, 1);
J_fit_history = zeros(max_iter, 1);
J_mmjp_history = zeros(max_iter, 1);
J_hsic_history = zeros(max_iter, 1);

% 迭代优化
J_prev = inf;
for iter = 1:max_iter
    % 步骤 1：优化 W_intra,c
    W_intra_new = cell(3, 1);
    for c = 1:3
        [W_D, W_A] = compute_intra_adjacency_matrix(X_modes{c}, K, t1, t2);
        W_fit = alpha * W_D + beta * W_A;
        n = size(X_modes{c}, 1);
        H = eye(n) - ones(n, n) / n;
        K_D = W_D' * W_D; K_A = W_A' * W_A;
        K = H * K_D * H + H * K_A * H;
        A = eye(n); B = lambda * K;
        C = W_fit + mu * (X_modes{c} * P) * (X_modes{c} * P)';
        W_intra_new{c} = lyap(A, B, -C);
        W_intra_new{c} = (W_intra_new{c} + W_intra_new{c}') / 2;
        W_intra{c} = (1 - eta) * W_intra{c} + eta * W_intra_new{c};
    end
    
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
Y = X * P;
figure;
scatter(Y(1:200, 1), Y(1:200, 2), 50, 'r+', 'DisplayName', 'Mode 1'); hold on;
scatter(Y(201:400, 1), Y(201:400, 2), 50, 'b*', 'DisplayName', 'Mode 2');
scatter(Y(401:600, 1), Y(401:600, 2), 50, 'go', 'DisplayName', 'Mode 3');
title('Low-dimensional Projection (DiMSC-inspired)'); xlabel('Dim 1'); ylabel('Dim 2'); legend; grid on; hold off;

%% 测试
X_train_modes = {X_vis_1, X_vis_2, X_vis_3};
mode_labels_case1 = [ones(200, 1); 2 * ones(200, 1)];
mode_labels_case2 = [ones(200, 1); 2 * ones(200, 1); 3 * ones(200, 1)];
alpha_conf = 0.01;
[T2_case1, Q_case1, T2_limits, Q_limits, Lambda] = compute_monitoring_stats(X_train_modes, test_case1(:,1:3), P, mode_labels_case1, alpha_conf);
[T2_case2, Q_case2, ~, ~, ~] = compute_monitoring_stats(X_train_modes, test_case2(:,1:3), P, mode_labels_case2, alpha_conf);

figure;
subplot(2, 1, 1); plot(T2_case1, 'b-', 'LineWidth', 1.5); hold on;
plot([1 200], [T2_limits(1) T2_limits(1)], 'r--', 'LineWidth', 1.5);
plot([200 200], ylim, 'k--');
plot(201:400, T2_limits(2) * ones(200, 1), 'r--', 'LineWidth', 1.5);
title('T^2 (Case 1)'); ylabel('T^2'); xlabel('Sample Index'); legend('T^2', 'Control Limit'); grid on; hold off;
subplot(2, 1, 2); plot(Q_case1, 'b-', 'LineWidth', 1.5); hold on;
plot([1 200], [Q_limits(1) Q_limits(1)], 'r--', 'LineWidth', 1.5);
plot([200 200], ylim, 'k--');
plot(201:400, Q_limits(2) * ones(200, 1), 'r--', 'LineWidth', 1.5);
title('Q (Case 1)'); ylabel('Q'); xlabel('Sample Index'); legend('Q', 'Control Limit'); grid on; hold off;

figure;
subplot(2, 1, 1); plot(T2_case2, 'b-', 'LineWidth', 1.5); hold on;
plot([1 200], [T2_limits(1) T2_limits(1)], 'r--', 'LineWidth', 1.5);
plot([200 200], ylim, 'k--');
plot(201:400, T2_limits(2) * ones(200, 1), 'r--', 'LineWidth', 1.5);
plot([400 400], ylim, 'k--');
plot(401:600, T2_limits(3) * ones(200, 1), 'r--', 'LineWidth', 1.5);
title('T^2 (Case 2)'); ylabel('T^2'); xlabel('Sample Index'); legend('T^2', 'Control Limit'); grid on; hold off;
subplot(2, 1, 2); plot(Q_case2, 'b-', 'LineWidth', 1.5); hold on;
plot([1 200], [Q_limits(1) Q_limits(1)], 'r--', 'LineWidth', 1.5);
plot([200 200], ylim, 'k--');
plot(201:400, Q_limits(2) * ones(200, 1), 'r--', 'LineWidth', 1.5);
plot([400 400], ylim, 'k--');
plot(401:600, Q_limits(3) * ones(200, 1), 'r--', 'LineWidth', 1.5);
title('Q (Case 2)'); ylabel('Q'); xlabel('Sample Index'); legend('Q', 'Control Limit'); grid on; hold off;

%% 函数定义



















