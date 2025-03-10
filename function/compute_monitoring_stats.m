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