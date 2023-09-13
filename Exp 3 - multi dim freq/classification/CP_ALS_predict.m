function [pred, pred_train, C_test, C_train, regu] = CP_ALS_predict(XTest, W, norms, M, X, reg)
    [N_test,~] = size(XTest);
    [N,D] = size(X);
    Y = ones(N,1) ;
    YTest = ones(N_test,1) ;
    pred = ones(N_test,1);
    pred_train = ones(N,1);
    for d = 1:D
        feat{d} = features(XTest(:,d), M) ;
        pred = pred.*(feat{d} * W{d}) ;
        pred_train = pred_train.*(features(X(:,d), M) * W{d}) ;
    end

    for d = 1:D
        ZW_train = pred_train./(features(X(:,d), M) * W{d});
        ZW_test = pred./(features(XTest(:,d), M) * W{d});
        [~, ~, c_train] = C_matrices(features(X(:,d), M) , ZW_train, Y) ;
        [~, ~, c_test] = C_matrices(feat{d}, ZW_test, YTest) ;
        C_test{d} = c_test ;
        C_train{d} = c_train ;
        regu{d} = kron(reg./(W{d}'*W{d}), eye(M)) ;
    end

    pred = pred * norms ;
    pred_train = pred_train * norms ;

    
end
