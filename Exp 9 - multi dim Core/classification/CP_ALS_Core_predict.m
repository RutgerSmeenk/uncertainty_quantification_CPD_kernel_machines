function [pred, var_CI, var_PI, sigmae] = CP_ALS_Core_predict(XTest, W, norms, M, sigma, sto_core, Y, X)
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
        [CC_train, ~, ~] = C_matrices(features(X(:,d), M) , ZW_train, Y) ;
        [~, ~, c_test] = C_matrices(feat{d}, ZW_test, YTest) ;
        %C_test = c_test ;
        %C_train = c_train ;
        %regu = reg./(W{d}'*W{d}) ;

    pred = pred * norms ;
    pred_train = pred_train*norms ;

    sigmae = var(Y - pred_train, 1) ;

    % d = sto_core ;
    % Z_test = feat{d} ;
    % ZW_test = ZW./(feat{d}*W_mean{d}); 
    % [~,~, c_test] = C_matrices(Z_test,ZW_test, YTest) ;

    num = numel(W{d}) ;
    P0 = sigma*eye(num) ;
    P0_inv = inv(P0) ;
    W_cov = pinv(CC_train./sigmae + P0_inv) ;%\ eye(num);

    var_CI = diag(c_test * W_cov * c_test')  ;

    var_PI = var_CI ;
end