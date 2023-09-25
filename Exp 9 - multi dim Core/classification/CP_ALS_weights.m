function [W, norms] = CP_ALS_weights(X, y, M, R, lambda, maxIte)
    %rng(8)
    [N, D] = size(X);
    W = cell(1,D);
    ZW = 1;
    reg = 1;
    norms = ones(R, 1) ;

    % Initialization
    for d = D:-1:1
        W{d} = randn(M,R);
        W{d} = W{d}/norm(W{d});
        reg = reg.*(W{d}'*W{d});
        Z = features(X(:,d),M);
        ZW = (Z*W{d}).*ZW;
    end

    % Main loop
    itemax = maxIte*(2*(D-1))+1;
    for ite = 1:itemax
        loopind = mod(ite-1,2*(D-1))+1;
        if loopind <= D
            d = loopind;
        else
            d = 2*D-loopind;
        end
        Z = features(X(:,d),M);
        reg = reg./(W{d}'*W{d}); 
        ZW = ZW./(Z*W{d});
        [CC,Cy,~] = C_matrices(Z,ZW,y);
        x = (CC+lambda*N*kron(reg,eye(M)))\Cy;
        clear CC Cy
        W{d} = reshape(x,M,R);
        reg = reg.*(W{d}'*W{d});
        ZW = ZW.*(Z*W{d});
        % Normalization
        % [W, norms] = normalization_cpd(W,norms) ;
        norms = ones(R,1) ;
    end
end
