function L = compute_laplacian(W_global)
    D = diag(sum(W_global, 2));
    L = D - W_global;
end

