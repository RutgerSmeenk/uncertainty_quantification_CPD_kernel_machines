function [pred, var_CI, var_PI] = CP_ALS_Core_predict(XTest, W, norms, M, sigma, sto_core, Y, X)
    [N_test,~] = size(XTest);
    [N,~] = size(X);
    D = length(1:log2(M)) ;
    YTest = ones(N_test,1);
    pred = ones(N_test,1);
    pred_train = ones(N,1);
    for m = 1:D
        feat{m} = features(XTest,m, M) ;
        pred = pred.*(feat{m} * W{m}) ;
        pred_train = pred_train.*(features(X, m, M) * W{m}) ;
    end

        m = sto_core ;
        ZW_train = pred_train./(features(X,m, M) * W{m});
        ZW_test = pred./(features(XTest,m, M) * W{m});
        [CC_train, ~, ~] = C_matrices(features(X,m, M) , ZW_train, Y) ;
        [~, ~, c_test] = C_matrices(feat{m}, ZW_test, YTest) ;

    pred = pred * norms ;
    pred_train = pred_train*norms ;
    
    sigmae = var(Y - pred_train, 1) ;
    

    num = numel(W{m}) ;
    P0 = sigma*eye(num) ;
    P0_inv = inv(P0) ;
    W_cov = pinv(CC_train./sigmae + P0_inv) ;

    var_CI = diag(c_test * W_cov * c_test')  ;

    var_PI = var_CI + sigmae ;
end