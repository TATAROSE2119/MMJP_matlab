function [x,y,z] = ellipsoid3(mu, cov_mat, scale)
    [V,D] = eig(cov_mat);
    [x,y,z] = sphere(50);
    ap = [x(:) y(:) z(:)] * V * sqrt(D) * scale;
    x = reshape(ap(:,1), size(x)) + mu(1);
    y = reshape(ap(:,2), size(y)) + mu(2);
    z = reshape(ap(:,3), size(z)) + mu(3);
end
