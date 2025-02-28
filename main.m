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


%% 5. 绘制聚类结果
X_vis = train_data(:,1:2);
true_labels = [ones(200,1); 2*ones(200,1); 3*ones(200,1)];
clusters = multimanifold_spectral_clustering(X_vis, 3, 10);

figure;
hold on;
markers = {'+', '*', 'o'};
colors = {'r', 'b', 'g'};
for label = 1:3
    mask = (clusters == label);
    scatter(X_vis(mask,1), X_vis(mask,2), 50, colors{label}, markers{label});
end

% 绘制椭圆
for label = 1:3
    mask = (true_labels == label);
    data = X_vis(mask,:);
    mu = mean(data);
    cov_mat = cov(data);
    [V, D] = eig(cov_mat);
    theta = atan2(V(2,1), V(1,1));
    scale = sqrt(chi2inv(0.99, 2));
    width = 2*sqrt(D(1,1))*scale;
    height = 2*sqrt(D(2,2))*scale;
    ellipse(mu, width, height, theta, 'b--');
end
title('Clustering Result (x1 vs x2)');
xlabel('x1'); ylabel('x2');
legend('Mode1','Mode2','Mode3');
hold off;


%% 执行主程序
% 计算代表点
exemplars = compute_exemplars(X_vis, true_labels);

% 绘制Case1
plot_mode_identification(test_case1(:,1:2), 1, exemplars);

% 绘制Case2
plot_mode_identification(test_case2(:,1:2), 2, exemplars);


%% 3. 多流形谱聚类
function labels = multimanifold_spectral_clustering(X_vis, n_clusters, n_neighbors)
    distance_matrix = pdist2(X_vis, X_vis);
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

%% 4. 计算代表点
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

%% 6. 在线模式识别
function plot_mode_identification(test_case, case_num, exemplars)
    n_samples = size(test_case,1);
    mode_ids = zeros(n_samples,1);
    
    for i = 1:n_samples
        x_new = test_case(i,1:2);
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
%% 椭圆绘制函数（需单独保存为ellipse.m）
function h = ellipse(mu, width, height, theta, linestyle)
    t = linspace(0, 2*pi, 100);
    xy = [width/2 * cos(t); height/2 * sin(t)];
    R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    xy = R * xy;
    x = xy(1,:) + mu(1);
    y = xy(2,:) + mu(2);
    h = plot(x, y, linestyle, 'LineWidth', 1.5);
end