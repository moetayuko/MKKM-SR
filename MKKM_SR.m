function [finalInd, alpha, obj, allind] = MKKM_SR(K, Y, lambda, NITER, early_stop)
% MKKM_SR  Multiple Kernel K-Means Clustering with Simultaneous Spectral Rotation
%   [finalInd, alpha, obj, allind] = MKKM_SR(K, Yinit, lambda, NITER, early_stop)
%   K: n*n*v kernel matrices.
%   Y: n*c initial label indicator matrix.
%   lambda: hyperparameter
%   NITER: max number of iterations
%   early_stop: whether to break loop when objective no longer decrease
%
%   J. Lu, Y. Lu, R. Wang, F. Nie and X. Li, "Multiple Kernel K-Means
%   Clustering with Simultaneous Spectral Rotation," IEEE International
%   Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp.
%   4143-4147, doi: 10.1109/ICASSP43922.2022.9746905.
%
%   SPDX-FileCopyrightText: 2021-2022 Jitao Lu <dianlujitao@gmail.com>
%   SPDX-License-Identifier: MIT

if nargin < 4
    NITER = 30;
end

if nargin < 5
    early_stop = true;
end

[num, ~, numker] = size(K);
c = size(Y, 2);

alpha = ones(numker,1)/numker;

R = orth(rand(c,c));
F = orth(rand(num,c));

allind = [];

for iter = 1:NITER
    % Update F
    K_alpha = calculate_kernel_theta(K,alpha);
    G = Y ./ sqrt(sum(Y)) * R';
    for it_gpi = 1:15
        M = 2*(K_alpha*F + lambda*G);
        [Um,~,Vm] = svd(M,'econ');
        F = Um*Vm';
    end

    % update R
    N = F' * Y ./ sqrt(sum(Y));
    [Un,~,Vn] = svd(N,'econ');
    R = Un*Vn';

    %  update Y
    P = F*R;
    [Y, obj_cd] = CD_SR(P,Y);

    allind = [allind; vec2ind(Y')];
    obj(iter) = trace(K_alpha) - sum(F .* (K_alpha * F), 'all') ...
        + lambda * (sum(P .^ 2, 'all') + c - 2 * sum(P .* Y ./ sqrt(sum(Y)), 'all'));
    if early_stop && iter > 2 && ...
            abs((obj(iter) - obj(iter - 1)) / obj(iter - 1)) < 1e-6
        break;
    end

    % Update alpha
    f = reshape(K, [], numker)' * reshape(eye(num)-F*F', [], 1);
    h = sqrt(f);
    alpha = h/sum(h);
end

finalInd = vec2ind(Y')';

end

function K_alpha = calculate_kernel_theta(K,alpha)
    [num, ~, numker] = size(K);
    KK = reshape(K, [], numker);
    K_alpha = reshape(KK * (1 ./ alpha), num, num);
end

function [Y, obj]  = CD_SR(P, Y, NITER, early_stop)

if nargin < 3
    NITER = 30;
end

if nargin < 4
    early_stop = true;
end

n = size(Y, 1);

%% Initialize
TempPY = diag(P'*Y)';
TempYY = sum(Y);
m_all = vec2ind(Y');

%%
for iter = 1:NITER
    obj(iter) = sum(TempPY ./ sqrt(TempYY));
    if early_stop && iter > 2 && ...
            abs((obj(iter) - obj(iter - 1)) / obj(iter - 1)) < 1e-3
        break;
    end

    for row = 1:n
        m = m_all(row);
        % avoid generating empty cluster
        if TempYY(m) == 1
            continue;
        end

        Temp = P(row,:);
        y0 = TempPY./sqrt(TempYY);
        yk = (TempPY+Temp)./sqrt(TempYY+1);
        y0(m)= (TempPY(m)-Temp(m))/sqrt(TempYY(m)-1);
        yk(m)= TempPY(m)/sqrt(TempYY(m));
        delta = yk-y0;

        [~,p] = max(delta);
        if p ~= m
            Y(row,m) = 0;
            Y(row,p) = 1;
            m_all(row) = p;

            TempPY(m) = TempPY(m)-Temp(m);
            TempYY(m) = TempYY(m)-1;
            TempPY(p) = TempPY(p)+Temp(p);
            TempYY(p) = TempYY(p)+1;
        end
    end
end

end
