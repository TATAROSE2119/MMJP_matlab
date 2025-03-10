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