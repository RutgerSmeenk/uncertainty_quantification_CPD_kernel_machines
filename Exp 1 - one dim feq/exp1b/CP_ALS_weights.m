function [W, norms, reg] = CP_ALS_weights(X, y, M, R, lambda, maxIte)

    D = length(1:log2(M)) ;
    N = size(X,1) ;
    W = cell(1,D);
    ZW = 1;
    reg = 1;
    norms = ones(R, 1) ;

    % Initialization
    for m = D:-1:1
        W{m} = randn(2,R);
        W{m} = W{m}/norm(W{m});
        reg = reg.*(W{m}'*W{m});
        %reg = reg.*vecnorm(W{m},2,1).^2;
        Z = features(X,m,M);
        ZW = (Z*W{m}).*ZW;
    end
    
    % Main loop
    itemax = maxIte*(2*(D-1))+1;
    for ite = 1:itemax
        loopind = mod(ite-1,2*(D-1))+1;
        if loopind <= D
            m = loopind;
        else
            m = 2*D-loopind;
        end
        Z = features(X,m,M);
        reg = reg./(W{m}'*W{m});
        %reg = reg./vecnorm(W{m},2,1).^2;
        ZW = ZW./(Z*W{m}); 
        [CC,Cy,~] = C_matrices(Z,ZW,y);
        x = (CC+lambda*N*kron(reg,eye(2)))\Cy;
        %x = (CC+lambda*N*diag(kron(reg,ones(1,2))))\Cy;
        clear CC Cy
        W{m} = reshape(x,2,R);
        reg = reg.*(W{m}'*W{m});
        %reg = reg.*vecnorm(W{m},2,1).^2;
        ZW = ZW.*(Z*W{m});
        % Normalization
        %[W, norms] = normalization_cpd(W,norms) ;
    end
end

