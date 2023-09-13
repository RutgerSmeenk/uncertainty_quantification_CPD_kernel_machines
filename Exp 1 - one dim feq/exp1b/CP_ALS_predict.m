% function score = CPPredict(X, W, hyperparameters)
%     [N,D] = size(X);
%     M = size(W{1},1);
%     score = ones(N,1);
%     for d = 1:D
%         score = score.*(features(X(:,d),M,hyperparameters)*W{d});
%     end
%     score = sum(score,2);
% end

function [pred, C_test, C_train, regu] = CP_ALS_predict(XTest, W, norms, M, X, reg)
    [N_test,~] = size(XTest);
    [N,~] = size(X);
    D = length(1:log2(M)) ;
    Y = ones(N,1) ;
    YTest = ones(N_test,1) ;
    pred = ones(N_test,1);
    pred_train = ones(N,1);
    for m = 1:D
        feat{m} = features(XTest,m, M) ;
        pred = pred.*(feat{m} * W{m}) ;
        pred_train = pred_train.*(features(X,m, M) * W{m}) ;
    end

    for m = 1:D
        ZW_train = pred_train./(features(X,m, M) * W{m});
        ZW_test = pred./(features(XTest,m, M) * W{m});
        [~, ~, c_train] = C_matrices(features(X,m, M), ZW_train, Y) ;
        [~, ~, c_test] = C_matrices(feat{m}, ZW_test, YTest) ;
        C_test{m} = c_test ;
        C_train{m} = c_train ;
        regu{m} = kron(reg./(W{m}'*W{m}), eye(2)) ;
    end

    pred = pred * norms ;

    
end
