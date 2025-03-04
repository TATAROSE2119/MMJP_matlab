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
function X = generate_data(mode, num_samples)
    if mode == 1
        s1 = unifrnd(-10, -7, num_samples, 1);
        s2 = normrnd(-5, 1, num_samples, 1);
    elseif mode == 2
        s1 = unifrnd(-3, -1, num_samples, 1);
        s2 = normrnd(2, 1, num_samples, 1);
    elseif mode == 3
        s1 = unifrnd(2, 5, num_samples, 1);
        s2 = normrnd(7, 1, num_samples, 1);
    end
    A = [0.5768, 0.3766; 0.7382, 0.0566; 0.8291, 0.4009; 0.6519, 0.2070; 0.3972, 0.8045];
    S = [s1, s2];
    noise = normrnd(0, 0.01, num_samples, 5);
    X = S * A' + noise;
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

function L = compute_laplacian(W_global)
    D = diag(sum(W_global, 2));
    L = D - W_global;
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

function labels = multimanifold_spectral_clustering(X_vis, n_clusters, n_neighbors)
    distance_matrix = pdist2(X_vis, X_vis);
    adj_matrix = zeros(size(distance_matrix));
    for i = 1:size(X_vis,1)
        [~, idx] = sort(distance_matrix(i,:));
        neighbors = idx(2:n_neighbors+1);
        sigma = mean(distance_matrix(i, neighbors));
        adj_matrix(i, neighbors) = exp(-distance_matrix(i, neighbors).^2/(2*sigma^2));
        adj_matrix(neighbors, i) = adj_matrix(i, neighbors);
    end
    labels = spectralcluster(adj_matrix, n_clusters, 'Distance', 'precomputed');
end

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

function plot_3d_clustering(X_vis, clusters, true_labels)
    figure; hold on;
    markers = {'+', '*', 'o'};
    colors = {'r', 'b', 'g'};
    for label = 1:3
        mask = (clusters == label);
        scatter3(X_vis(mask,1), X_vis(mask,2), X_vis(mask,3), 50, colors{label}, markers{label});
    end
    for label = 1:3
        mask = (true_labels == label);
        data = X_vis(mask,:);
        mu = mean(data);
        cov_mat = cov(data);
        scale = sqrt(chi2inv(0.99, 3));
        [x,y,z] = ellipsoid3(mu, cov_mat, scale);
        surf(x, y, z, 'FaceAlpha',0.1, 'EdgeColor','none', 'FaceColor', 'blue');
    end
    title('3D Clustering Result (x1-x3)'); xlabel('x1'); ylabel('x2'); zlabel('x3');
    legend('Mode1', 'Mode2', 'Mode3'); hold off;
end

function [x,y,z] = ellipsoid3(mu, cov_mat, scale)
    [V,D] = eig(cov_mat);
    [x,y,z] = sphere(50);
    ap = [x(:) y(:) z(:)] * V * sqrt(D) * scale;
    x = reshape(ap(:,1), size(x)) + mu(1);
    y = reshape(ap(:,2), size(y)) + mu(2);
    z = reshape(ap(:,3), size(z)) + mu(3);
end

function plot_mode_identification(test_case, case_num, exemplars)
    n_samples = size(test_case,1);
    mode_ids = zeros(n_samples,1);
    for i = 1:n_samples
        x_new = test_case(i,1:3);
        distances = sum((exemplars - x_new).^2, 2);
        [~, mode] = min(distances);
        mode_ids(i) = mode;
    end
    figure;
    plot(1:n_samples, mode_ids, 'Color', [0.5 0 0.5], 'LineWidth', 2);
    title(['Online Identification (Case ' num2str(case_num) ')']);
    xlabel('Sample index'); ylabel('Mode'); ylim([0 4]); grid on;
end