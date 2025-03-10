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
