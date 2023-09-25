function pred = CP_data_gen(XTest, W)
    [N_test,~] = size(XTest);
    pred = ones(N_test,1);
    D = length(W) ;
    [M,R] = size(W{1}) ;
    for d = 1:D
        feat{d} = features(XTest(:,d), M) ;
        pred = pred.*(feat{d} * W{d}) ;
    end

    pred = pred * ones(R,1) ;
    
end
