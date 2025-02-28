%% 主程序
clear; clc; close all;
rng(4); % 设置随机种子


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
% 模式1数据
test_case2 = [test_case2; generate_data(1, 200)];
% 模式2斜坡故障
for i = 201:400
    x = generate_data(2, 1);
    x(1) = x(1) + 0.002*(i-100);
    test_case2 = [test_case2; x];
end
% 模式3数据
test_case2 = [test_case2; generate_data(3, 100)];
% 模式3阶跃故障
test_mode3_fault = generate_data(3, 100);
test_mode3_fault(:,1) = test_mode3_fault(:,1) + 0.08;
test_case2 = [test_case2; test_mode3_fault];


%% 主程序执行
% 使用三维特征进行聚类
X_vis = train_data(:,1:3); % 取前三个维度
true_labels = [ones(200,1); 2*ones(200,1); 3*ones(200,1)];
clusters = multimanifold_spectral_clustering(X_vis, 3, 10);

% 绘制三维聚类结果
plot_3d_clustering(X_vis, clusters, true_labels);

% 计算三维代表点
exemplars = compute_exemplars(X_vis, true_labels);

% 在线模式识别（使用三维测试数据）
plot_mode_identification(test_case1(:,1:3), 1, exemplars);
plot_mode_identification(test_case2(:,1:3), 2, exemplars);



%%---------------------------训练P----------------------------------


% 2. 生成模式数据（示例：仅生成 Mode 1 的 200 个样本）
mode1_data = generate_data(1, 200); % 假设计算 Mode 1 的邻接矩阵
mode2_data=generate_data(2,200);
mode3_data=generate_data(3,200);
X_vis_1 = mode1_data(:, 1:3); % 使用 x1 和 x2 x3
X_vis_2 = mode2_data(:, 1:3); % 使用 x1 和 x2 x3
X_vis_3 = mode3_data(:, 1:3); % 使用 x1 和 x2 x3
% 4. 主程序：调用函数计算模态内邻接矩阵并可视化（可选）
K = 10; % KNN 邻居数（根据论文第6页）
t1 = 1.0; % 热核参数 t1（经验值，可优化）
t2 = 0.5; % 热核参数 t2（经验值，可优化）
t3=1.0;

% 计算 Mode 1,2,3 的模态内邻接矩阵
W_intra_mode1 = compute_intra_adjacency_matrix(X_vis_1, K, t1, t2);
W_intra_mode2 = compute_intra_adjacency_matrix(X_vis_2, K, t1, t2);
W_intra_mode3 = compute_intra_adjacency_matrix(X_vis_3, K, t1, t2);


% 计算模态间邻接矩阵
W_inter_12 = compute_intermode_adjacency(X_vis_1, X_vis_2, K, t3); % Mode 1 到 Mode 2
W_inter_13 = compute_intermode_adjacency(X_vis_1, X_vis_3, K, t3); % Mode 1 到 Mode 3
W_inter_23 = compute_intermode_adjacency(X_vis_2, X_vis_3, K, t3); % Mode 2 到 Mode 3

% 显示结果（示例：W_inter_12 前5x5块）
disp('W_inter,12 (Mode 1 到 Mode 2) 的前5x5块：');
disp(W_inter_12(1:5, 1:5));

% 可视化 W_inter_12
figure;
imagesc(W_inter_12);
colorbar;
title('模态间邻接矩阵 W_{inter,12} (Mode 1 到 Mode 2)');
xlabel('Mode 2 样本索引');
ylabel('Mode 1 样本索引');
% 可视化 W_inter_13
figure;
imagesc(W_inter_13);
colorbar;
title('模态间邻接矩阵 W_{inter,13} (Mode 1 到 Mode 3)');
xlabel('Mode 3 样本索引');
ylabel('Mode 1 样本索引');
% 可视化 W_inter_23
figure;
imagesc(W_inter_23);
colorbar;
title('模态间邻接矩阵 W_{inter,23} (Mode 2 到 Mode 3)');
xlabel('Mode 3 样本索引');
ylabel('Mode 2 样本索引');

% 可视化邻接矩阵（可选，用于调试）
figure;
imagesc(W_intra_mode1);
colorbar;
title('Intramode Adjacency Matrix for Mode 1');
xlabel('Sample Index');
ylabel('Sample Index');

figure;
imagesc(W_intra_mode2);
colorbar;
title('Intramode Adjacency Matrix for Mode 1');
xlabel('Sample Index');
ylabel('Sample Index');

figure;
imagesc(W_intra_mode3);
colorbar;
title('Intramode Adjacency Matrix for Mode 1');
xlabel('Sample Index');
ylabel('Sample Index');

X_modes = {X_vis_1, X_vis_2, X_vis_3};

% 参数设置
K = 10; % 最近邻数量
t1 = 1.0; % 模态内邻居热核参数
t2 = 2.0; % 模态内非邻居热核参数
t3 = 1.0; % 模态间热核参数

% 计算全局亲和矩阵
W_global = compute_global_affinity(X_modes, K, t1, t2, t3);

% 显示部分矩阵
disp('W_global 的前5x5块：');
disp(W_global(1:5, 1:5));

% 可视化
figure;
imagesc(W_global);
colorbar;
title('多流形亲和矩阵 W_{global}');
xlabel('样本索引');
ylabel('样本索引');






%% 计算全局亲和矩阵函数
function W_global = compute_global_affinity(X_modes, K, t1, t2, t3)
    C = length(X_modes); % 模态数量
    N = sum(cellfun(@(x) size(x, 1), X_modes)); % 总样本数
    W_global = zeros(N, N);
    
    % 计算所有模态内和模态间矩阵
    W_intra = cell(C, 1);
    W_inter = cell(C, C);
    offsets = [0 cumsum(cellfun(@(x) size(x, 1), X_modes(1:end-1)))];
    
    for c = 1:C
        % 模态内矩阵
        W_intra{c} = compute_intra_adjacency_matrix(X_modes{c}, K, t1, t2);
        
        % 模态间矩阵
        for p = 1:C
            if c ~= p
                W_inter{c, p} = compute_intermode_adjacency(X_modes{c}, X_modes{p}, K, t3);
            end
        end
    end
    
    % 填充 W_global
    for c = 1:C
        rc = offsets(c) + 1 : offsets(c) + size(X_modes{c}, 1);
        W_global(rc, rc) = W_intra{c}; % 对角块
        for p = 1:C
            if c ~= p
                rp = offsets(p) + 1 : offsets(p) + size(X_modes{p}, 1);
                W_global(rc, rp) = W_inter{c, p}; % 非对角块
                W_global(rp, rc) = W_inter{c, p}'; % 对称
            end
        end
    end
end




%% 封装的模态间邻接矩阵计算函数
function W_inter = compute_intermode_adjacency(Xc, Xp, K, t3)
    % 输入：
    % Xc - 第c模态的数据矩阵 (nc x d)
    % Xp - 第p模态的数据矩阵 (np x d)
    % K - 最近邻数量
    % t3 - 热核参数
    % 输出：
    % W_inter - 模态间邻接矩阵 (nc x np)
    
    % 获取样本数和维度
    nc = size(Xc, 1); % 第c模态样本数
    np = size(Xp, 1); % 第p模态样本数
    
    % 初始化模态间邻接矩阵
    W_inter = zeros(nc, np);
    
    % 计算两模态间的欧几里得距离矩阵
    dist_matrix = pdist2(Xc, Xp, 'euclidean');
    
    % 计算Xc到Xp的KNN
    [~, idx_cp] = sort(dist_matrix, 2); % 按行排序
    knn_cp = idx_cp(:, 1:K); % Xc每个点的K个最近邻在Xp中的索引
    
    % 计算Xp到Xc的KNN
    [~, idx_pc] = sort(dist_matrix', 2); % 按行排序
    knn_pc = idx_pc(:, 1:K); % Xp每个点的K个最近邻在Xc中的索引
    
    % 填充W_inter (Xc到Xp方向)
    for i = 1:nc
        neighbors_cp = knn_cp(i, :);
        for j = neighbors_cp
            W_inter(i, j) = exp(-dist_matrix(i, j)^2 / t3);
        end
    end
    
    % 填充W_inter (Xp到Xc方向，确保双向邻居关系)
    for j = 1:np
        neighbors_pc = knn_pc(j, :);
        for i = neighbors_pc'
            if W_inter(i, j) == 0 % 避免覆盖已有的双向邻居
                W_inter(i, j) = exp(-dist_matrix(i, j).^2 / t3);
            end
        end
    end
end
% 3. 计算模态内邻接矩阵
function W_intra = compute_intra_adjacency_matrix(X, K, t1, t2)
    % 计算模态内邻接矩阵
    % 输入：X（模式数据矩阵，n_samples x d），K（KNN 邻居数），t1, t2（热核参数）
    % 输出：W_intra（模态内邻接矩阵，n_samples x n_samples）
    
    [n_samples, ~] = size(X);
    
    % 计算欧几里得距离矩阵（简化测地线距离）
    distance_matrix = pdist(X, 'euclidean');
    distance_matrix = squareform(distance_matrix);
    
    % 1. 构建距离-based 邻接矩阵 W_D,c
    W_D = zeros(n_samples, n_samples);
    for i = 1:n_samples
        % 找到 K 最近邻
        [~, neighbors] = sort(distance_matrix(i, :));
        neighbor_set = neighbors(2:K+1); % 排除自身（i=1 对应自身）
        
        for j = 1:n_samples
            if any(neighbor_set == j) || any(neighbors(2:K+1) == i) % 如果 i 或 j 是对方的邻居
                W_D(i, j) = exp(-distance_matrix(i, j)^2 / t1);
            else % 非邻居
                W_D(i, j) = exp(-distance_matrix(i, j)^2 / t2);
            end
        end
    end
    
    % 2. 构建角度-based 邻接矩阵 W_A,c
    W_A = zeros(n_samples, n_samples);
    for k = 1:n_samples
        % 找到 k 的 K 最近邻
        [~, neighbors] = sort(distance_matrix(k, :));
        neighbor_set = neighbors(2:K+1); % 排除自身
        
        for i = 1:length(neighbor_set)
            for j = i+1:length(neighbor_set)
                idx_i = neighbor_set(i);
                idx_j = neighbor_set(j);
                % 计算余弦相似性（角度信息）
                vector_i = X(idx_i, :) - X(k, :);
                vector_j = X(idx_j, :) - X(k, :);
                norm_i = norm(vector_i);
                norm_j = norm(vector_j);
                if norm_i > 0 && norm_j > 0 % 避免除以零
                    W_A(idx_i, idx_j) = dot(vector_i, vector_j) / (norm_i * norm_j);
                    W_A(idx_j, idx_i) = W_A(idx_i, idx_j); % 确保对称
                end
            end
        end
    end
    
    % 3. 融合距离和角度矩阵
    W_intra = W_D + W_A;
end

%% 三维椭球生成函数（保存为ellipsoid3.m）
function [x,y,z] = ellipsoid3(mu, cov_mat, scale)
    [V,D] = eig(cov_mat);
    [x,y,z] = sphere(50);
    ap = [x(:) y(:) z(:)] * V * sqrt(D) * scale;
    x = reshape(ap(:,1), size(x)) + mu(1);
    y = reshape(ap(:,2), size(y)) + mu(2);
    z = reshape(ap(:,3), size(z)) + mu(3);
end

%% 1. 数据生成函数
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
    
    A = [0.5768, 0.3766;
         0.7382, 0.0566;
         0.8291, 0.4009;
         0.6519, 0.2070;
         0.3972, 0.8045];
    
    S = [s1, s2];
    noise = normrnd(0, 0.01, num_samples, 5);
    X = S * A' + noise;
end
%% 3. 多流形谱聚类（三维版本）
function labels = multimanifold_spectral_clustering(X_vis, n_clusters, n_neighbors)
    distance_matrix = pdist2(X_vis, X_vis); % 三维欧氏距离
    adj_matrix = zeros(size(distance_matrix));
    
    for i = 1:size(X_vis,1)
        [~, idx] = sort(distance_matrix(i,:));
        neighbors = idx(2:n_neighbors+1); % 排除自身
        sigma = mean(distance_matrix(i, neighbors));
        adj_matrix(i, neighbors) = exp(-distance_matrix(i, neighbors).^2/(2*sigma^2));
        adj_matrix(neighbors, i) = adj_matrix(i, neighbors);
    end
    
    labels = spectralcluster(adj_matrix, n_clusters, 'Distance', 'precomputed');
end

%% 4. 计算代表点（三维版本）
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

%% 5. 绘制三维聚类结果
function plot_3d_clustering(X_vis, clusters, true_labels)
    figure;
    hold on;
    markers = {'+', '*', 'o'};
    colors = {'r', 'b', 'g'};
    
    % 绘制三维散点图
    for label = 1:3
        mask = (clusters == label);
        scatter3(X_vis(mask,1), X_vis(mask,2), X_vis(mask,3),...
                 50, colors{label}, markers{label});
    end
    
    % 绘制三维椭球
    for label = 1:3
        mask = (true_labels == label);
        data = X_vis(mask,:);
        mu = mean(data);
        cov_mat = cov(data);
        [V, D] = eig(cov_mat);
        scale = sqrt(chi2inv(0.99, 3));
        
        % 生成椭球参数
        [x,y,z] = ellipsoid3(mu, cov_mat, scale);
        surf(x, y, z, 'FaceAlpha',0.1, 'EdgeColor','none', 'FaceColor', 'blue');
    end
    
    title('3D Clustering Result (x1-x3)');
    xlabel('x1'); ylabel('x2'); zlabel('x3');
    legend('Mode1','Mode2','Mode3');
    hold off;
end

%% 6. 在线模式识别（三维版本）
function plot_mode_identification(test_case, case_num, exemplars)
    n_samples = size(test_case,1);
    mode_ids = zeros(n_samples,1);
    
    for i = 1:n_samples
        x_new = test_case(i,1:3); % 使用三维特征
        distances = sum((exemplars - x_new).^2, 2);
        [~, mode] = min(distances);
        mode_ids(i) = mode;
    end
    
    figure;
    plot(1:n_samples, mode_ids, 'Color', [0.5 0 0.5], 'LineWidth',2);
    title(['Online Identification (Case ' num2str(case_num) ')']);
    xlabel('Sample index'); ylabel('Mode');
    ylim([0 4]);
    grid on;
end
