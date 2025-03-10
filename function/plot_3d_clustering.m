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