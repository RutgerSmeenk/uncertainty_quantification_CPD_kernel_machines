function [pred, C_train, var_PI, sigmae] = CP_ALS_Core_predict(XTest, W, norms, M, sigma, sto_core, Y, X)
    [N_test,~] = size(XTest);
    [N,D] = size(X);
    %Y = ones(N,1) ;
    YTest = ones(N_test,1);
    pred = ones(N_test,1);
    pred_train = ones(N,1);
    for d = 1:D
        feat{d} = features(XTest(:,d), M) ;
        pred = pred.*(feat{d} * W{d}) ;
        pred_train = pred_train.*(features(X(:,d), M) * W{d}) ;
    end

        d = sto_core ;
        ZW_train = pred_train./(features(X(:,d), M) * W{d});
        ZW_test = pred./(features(XTest(:,d), M) * W{d});
        [CC_train, ~, C_train] = C_matrices(features(X(:,d), M) , ZW_train, Y) ;
        [~, ~, c_test] = C_matrices(feat{d}, ZW_test, YTest) ;


    pred = pred * norms ;
    pred_train = pred_train * norms ;

    sigmae = var(Y - pred_train, 1) ;


    num = numel(W{d}) ;
    P0 = sigma*eye(num) ;
    P0_inv = inv(P0) ;
    W_cov = pinv(CC_train./sigmae + P0_inv) ;

    var_CI = diag(c_test * W_cov * c_test')  ;

    var_PI = var_CI + sigmae ;
end