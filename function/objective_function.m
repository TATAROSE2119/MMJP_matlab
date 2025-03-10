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

