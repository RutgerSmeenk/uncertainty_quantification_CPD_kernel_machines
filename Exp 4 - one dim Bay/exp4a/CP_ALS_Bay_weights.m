function [W_mean, W_cov, norms] = CP_ALS_Bay_weights(X, y, M, R, maxIte, sigma, sigmae)

    D = length(1:log2(M)) ;
    m0 = cell(1,D);
    P0 = cell(1,D);
    W_mean = cell(1,D);
    W_cov = cell(1,D);

    ZW = 1;
    norms = ones(R, 1) ;

    % Initialization
    for m = D:-1:1
        W_mean{m} = randn(2,R);
        W_mean{m} = W_mean{m}/norm(W_mean{m});
        %reg = reg.*vecnorm(W{m},2,1).^2;
        %reg = reg.*(W_mean{m}'*W_mean{m});
        Z = features(X,m,M);
        ZW = (Z*W_mean{m}).*ZW;

        % Compute prior mean and prior covariance
        num = numel(W_mean{m}) ;
        m0{m} = reshape(W_mean{m},[num 1]) ;
        P0_inv{m} = inv(sigma*eye(num)) ;
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
        ZW = ZW./(Z*W_mean{m}); 

        [CC,Cy,~] = C_matrices(Z,ZW,y) ;
        %P0_inv = inv(P0{m}) ;
        z = (CC./sigmae + P0_inv{m}) ;

        m0{m} = reshape(W_mean{m},[num 1]) ;
        x = (z \ (Cy./sigmae + P0_inv{m}*m0{m})) ;
        W_cov{m} = z ;
        clear CC Cy
        W_mean{m} = reshape(x,2,R) ;
        ZW = ZW.*(Z*W_mean{m}) ;
        % Normalization
        %[W_mean, m0, norms] = normalization_cpd(W_mean, m0, norms) ;
        norms = ones(R,1) ;
    end
        for m = 1:D
            W_cov{m} = inv(W_cov{m}) ;
        end
end

