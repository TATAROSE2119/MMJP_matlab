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
