%% 主程序
clear; clc; close all;
rng(42); % 设置随机种子

%% 2. 生成训练和测试数据
% 训练数据
train_mode1 = generate_data(1, 200);
train_mode2 = generate_data(2, 200);
train_mode3 = generate_data(3, 200);
train_data = [train_mode1; train_mode2; train_mode3];

% 测试数据案例1
test_mode1 = generate_data(1, 200);
test_mode2 = generate_data(2, 200);
test_mode2(:,1) = test_mode2(:,1) + 0.08;
test_case1 = [test_mode1; test_mode2];

% 测试数据案例2
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

%% 训练 P
mode1_data = generate_data(1, 200);
mode2_data = generate_data(2, 200);
mode3_data = generate_data(3, 200);
X_vis_1 = mode1_data(:, 1:3);
X_vis_2 = mode2_data(:, 1:3);
X_vis_3 = mode3_data(:, 1:3);

K = 10; % KNN 邻居数
t1 = 1.0; % 热核参数 t1
t2 = 0.5; % 热核参数 t2
t3 = 1.0;

% 计算初始模态内邻接矩阵
[W_intra_mode1_D, W_intra_mode1_A] = compute_intra_adjacency_matrix(X_vis_1, K, t1, t2);
[W_intra_mode2_D, W_intra_mode2_A] = compute_intra_adjacency_matrix(X_vis_2, K, t1, t2);
[W_intra_mode3_D, W_intra_mode3_A] = compute_intra_adjacency_matrix(X_vis_3, K, t1, t2);

% 计算模态间邻接矩阵
W_inter_12 = compute_intermode_adjacency(X_vis_1, X_vis_2, K, t3);
W_inter_13 = compute_intermode_adjacency(X_vis_1, X_vis_3, K, t3);
W_inter_23 = compute_intermode_adjacency(X_vis_2, X_vis_3, K, t3);

% 可视化（可选）
figure; imagesc(W_intra_mode1_D + W_intra_mode1_A); colorbar; title('W_{intra,1}');
figure; imagesc(W_intra_mode2_D + W_intra_mode2_A); colorbar; title('W_{intra,2}');
figure; imagesc(W_intra_mode3_D + W_intra_mode3_A); colorbar; title('W_{intra,3}');
figure; imagesc(W_inter_12); colorbar; title('W_{inter,12}');
figure; imagesc(W_inter_13); colorbar; title('W_{inter,13}');
figure; imagesc(W_inter_23); colorbar; title('W_{inter,23}');

X_modes = {X_vis_1, X_vis_2, X_vis_3};
X = [X_vis_1; X_vis_2; X_vis_3];
K = 10;
t1 = 1.0;
t2 = 2.0;
t3 = 1.0;
d = 2;
lambda = 0.1;
max_iter = 50;
tol = 1e-6;
eta = 0.1; % 梯度下降步长

% 初始化
alpha = 1;
beta = 1;
W_global = compute_global_affinity(X_modes, alpha, beta, K, t1, t2, t3);
L = compute_laplacian(W_global);
[V, D] = eig(X' * L * X);
[eigenvalues, idx] = sort(diag(D));
P = V(:, idx(1:d));
P = orth(P);

% 迭代优化
J_prev = inf;
for iter = 1:max_iter
    % 步骤 1：固定 P，优化 alpha 和 beta
    A = [];
    b = [];
    for c = 1:length(X_modes)
        [W_D, W_A] = compute_intra_adjacency_matrix(X_modes{c}, K, t1, t2);
        W_intra_c = W_D + W_A; % 动态更新时可优化，这里仍用初始值
        A = [A; W_D(:) W_A(:)];
        b = [b; W_intra_c(:)];
    end
    coeffs = (A' * A) \ (A' * b);
    alpha = coeffs(1);
    beta = coeffs(2);
    
    % 步骤 2：固定 alpha 和 beta，优化 P（添加 HSIC 梯度）
    W_global = compute_global_affinity(X_modes, alpha, beta, K, t1, t2, t3);
    L = compute_laplacian(W_global);
    M = X' * L * X;
    grad_mmjp = 2 * M * P; % MMJP 部分的梯度
    
    % 计算 HSIC 梯度
    grad_hsic = zeros(size(P));
    for c = 1:length(X_modes)
        Y_c = X_modes{c} * P; % n x d
        [n, d] = size(Y_c);
        H = eye(n) - ones(n, n) / n;
        for k = 1:d-1
            for l = k+1:d
                dist_k = pdist2(Y_c(:, k), Y_c(:, k), 'euclidean');
                dist_l = pdist2(Y_c(:, l), Y_c(:, l), 'euclidean');
                sigma_k = median(dist_k(:));
                sigma_l = median(dist_l(:));
                K_k = exp(-dist_k.^2 / (2 * sigma_k^2));
                K_l = exp(-dist_l.^2 / (2 * sigma_l^2));
                % 近似梯度（对 P 的偏导数）
                for i = 1:n
                    for j = 1:n
                        diff_k = Y_c(i,k) - Y_c(j,k);
                        diff_l = Y_c(i,l) - Y_c(j,l);
                        grad_k = -K_k(i,j) * diff_k / sigma_k^2 * (X_modes{c}(i,:) - X_modes{c}(j,:))';
                        grad_l = -K_l(i,j) * diff_l / sigma_l^2 * (X_modes{c}(i,:) - X_modes{c}(j,:))';
                        grad_hsic(:,k) = grad_hsic(:,k) + grad_k * trace(K_l * H) / ((n-1)^2);
                        grad_hsic(:,l) = grad_hsic(:,l) + grad_l * trace(K_k * H) / ((n-1)^2);
                    end
                end
            end
        end
    end
    grad = grad_mmjp + lambda * grad_hsic;
    
    % 正交约束梯度下降
    P_new = P - eta * (grad - P * (grad' * P));
    P_new = orth(P_new);
    
    % 步骤 3：动态更新 W_intra,c（可选）
    % 这里假设 W_intra,c 通过优化目标动态调整，当前仍用 W_D + W_A
    % 若需完全动态，可添加一步优化 W_intra,c 的过程
    
    % 计算目标函数
    J = objective_function(X_modes, alpha, beta, P_new, lambda, K, t1, t2, t3);
    
    % 检查收敛
    if abs(J_prev - J) < tol
        break;
    end
    P = P_new;
    J_prev = J;
    
    fprintf('Iteration %d: J = %.4f, alpha = %.4f, beta = %.4f\n', iter, J, alpha, beta);
end

% 输出结果
disp('Optimized alpha:'); disp(alpha);
disp('Optimized beta:'); disp(beta);
disp('Optimized P:'); disp(P);

%% 可视化
figure; imagesc(W_global); colorbar; title('W_{global}');
figure; imagesc(L); colorbar; title('Laplacian L');
Y = X * P;
figure;
scatter(Y(1:200, 1), Y(1:200, 2), 50, 'r+', 'DisplayName', 'Mode 1'); hold on;
scatter(Y(201:400, 1), Y(201:400, 2), 50, 'b*', 'DisplayName', 'Mode 2');
scatter(Y(401:600, 1), Y(401:600, 2), 50, 'go', 'DisplayName', 'Mode 3');
title('低维投影结果 (MMJP with HSIC)'); xlabel('Dim 1'); ylabel('Dim 2'); legend; grid on; hold off;

%% 测试
X_train = [X_vis_1; X_vis_2; X_vis_3];
X_train_modes = {X_vis_1, X_vis_2, X_vis_3};
mode_labels_case1 = [ones(200, 1); 2 * ones(200, 1)];
mode_labels_case2 = [ones(200, 1); 2 * ones(200, 1); 3 * ones(200, 1)];
alpha_conf = 0.01; % 置信水平 99%
[T2_case1, Q_case1, T2_limits, Q_limits, Lambda] = compute_monitoring_stats(X_train_modes, test_case1(:,1:3), P, mode_labels_case1, alpha_conf);
[T2_case2, Q_case2, ~, ~, ~] = compute_monitoring_stats(X_train_modes, test_case2(:,1:3), P, mode_labels_case2, alpha_conf);

% 可视化 Case 1
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

% 可视化 Case 2
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

%% 函数定义（以下保持不变或更新）
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

function W_global = compute_global_affinity(X_modes, alpha, beta, K, t1, t2, t3)
    C = length(X_modes);
    N = sum(cellfun(@(x) size(x, 1), X_modes));
    W_global = zeros(N, N);
    W_intra = cell(C, 1);
    W_inter = cell(C, C);
    offsets = [0 cumsum(cellfun(@(x) size(x, 1), X_modes(1:end-1)))];
    for c = 1:C
        [W_D, W_A] = compute_intra_adjacency_matrix(X_modes{c}, K, t1, t2);
        W_intra{c} = alpha * W_D + beta * W_A;
        W_intra{c} = (W_intra{c} + W_intra{c}') / 2;
        for p = 1:C
            if c ~= p
                W_inter{c, p} = compute_intermode_adjacency(X_modes{c}, X_modes{p}, K, t3);
            end
        end
    end
    for c = 1:C
        rc = offsets(c) + 1 : offsets(c) + size(X_modes{c}, 1);
        W_global(rc, rc) = W_intra{c};
        for p = 1:C
            if c ~= p
                rp = offsets(p) + 1 : offsets(p) + size(X_modes{p}, 1);
                W_global(rc, rp) = W_inter{c, p};
                W_global(rp, rc) = W_inter{c, p}';
            end
        end
    end
end

function L = compute_laplacian(W_global)
    D = diag(sum(W_global, 2));
    L = D - W_global;
end

function P = compute_projection_matrix(X, L, d)
    M = X' * L * X;
    [V, D] = eig(M);
    eigenvalues = diag(D);
    [~, idx] = sort(eigenvalues);
    P = V(:, idx(1:d));
    P = orth(P);
end

function hsic_val = compute_hsic(Y)
    [n, d] = size(Y);
    H = eye(n) - ones(n, n) / n;
    hsic_val = 0;
    for k = 1:d-1
        for l = k+1:d
            dist_k = pdist2(Y(:, k), Y(:, k), 'euclidean');
            dist_l = pdist2(Y(:, l), Y(:, l), 'euclidean');
            sigma_k = median(dist_k(:));
            sigma_l = median(dist_l(:));
            K_k = exp(-dist_k.^2 / (2 * sigma_k^2));
            K_l = exp(-dist_l.^2 / (2 * sigma_l^2));
            hsic_val = hsic_val + trace(K_k * H * K_l * H) / ((n-1)^2);
        end
    end
end

function J = objective_function(X_modes, alpha, beta, P, lambda, K, t1, t2, t3)
    % 输入：
    % X_modes - 各模态数据单元数组
    % alpha, beta - W_D 和 W_A 的权重
    % P - 投影矩阵 (D x d)
    % lambda - HSIC 惩罚参数
    % K, t1, t2, t3 - 邻接矩阵参数
    % 输出：
    % J - 总目标函数值
    
    C = length(X_modes); % 模态数量
    W_global = compute_global_affinity(X_modes, alpha, beta, K, t1, t2, t3); % 计算全局亲和矩阵
    L = compute_laplacian(W_global); % 计算拉普拉斯矩阵 L = D - W_global
    X = vertcat(X_modes{:}); % 合并所有模态数据，N x D
    
    % MMJP 项：保留几何结构的损失
    J_mmjp = trace(P' * X' * L * X * P); % tr(P' * X' * L * X * P)
    
    % HSIC 项：投影维度的独立性惩罚
    J_hsic = 0;
    for c = 1:C
        Y_c = X_modes{c} * P; % 投影到低维空间，n_c x d
        J_hsic = J_hsic + compute_hsic(Y_c); % 累加每个模态的 HSIC
    end
    
    % 拟合项：W_intra,c 与加权组合的差异
    J_fit = 0;
    for c = 1:C
        [W_D, W_A] = compute_intra_adjacency_matrix(X_modes{c}, K, t1, t2);
        W_intra_c = W_D + W_A; % 初始假设的 W_intra,c
        W_fit = alpha * W_D + beta * W_A; % 加权组合
        J_fit = J_fit + norm(W_intra_c - W_fit, 'fro')^2; % Frobenius 范数的平方
    end
    
    % 总目标函数：拟合项 + MMJP 项 + HSIC 惩罚项
    J = J_fit + J_mmjp + lambda * J_hsic; % 综合三部分
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