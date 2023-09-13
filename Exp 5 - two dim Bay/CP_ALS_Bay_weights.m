function [W_mean, W_cov, norms] = CP_ALS_Bay_weights(X, y, M, R, maxIte, sigma, sigmae)

    [~, D] = size(X);
    m0 = cell(1,D);
    P0 = cell(1,D);
    W_mean = cell(1,D);
    W_cov = cell(1,D);

    ZW = 1;
    norms = ones(R, 1) ;

    % Initialization
    for d = D:-1:1
        W_mean{d} = randn(M,R);
        W_mean{d} = W_mean{d}/norm(W_mean{d});

        Z = features(X(:,d),M);
        ZW = (Z*W_mean{d}).*ZW;

        % Compute prior mean and prior covariance
        num = numel(W_mean{d}) ;
        m0{d} = reshape(W_mean{d},[num 1]) ;
        P0_inv{d} = inv(sigma*eye(num)) ;
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
        ZW = ZW./(Z*W_mean{d}); 

        [CC,Cy,~] = C_matrices(Z,ZW,y) ;
        %P0_inv = inv(P0{d}) ;
        z = (CC./sigmae + P0_inv{d}) ;

        m0{d} = reshape(W_mean{d},[num 1]) ;
        x = (z \ (Cy./sigmae + P0_inv{d}*m0{d})) ;
        W_cov{d} = z ;
        clear CC Cy
        W_mean{d} = reshape(x,M,R) ;
        ZW = ZW.*(Z*W_mean{d}) ;
        % Normalization
        %[W_mean, m0] = normalization_cpd(W_mean, norms) ;
        norms = ones(R,1) ;
    end
        for d = 1:D
            W_cov{d} = inv(W_cov{d}) ;
        end
end
