function [mean, var_CI, var_PI] = CP_ALS_Bay_predict(X, W_mean, W_cov, norms, M, sigmae)
    [N,~] = size(X);
    %R = size(W_mean{m},2) ;
    D = length(1:log2(M)) ;
    mean = ones(N,1); 
    feat = cell(1,D) ;

    feat{1} = features(X,1,M) ;
    mean = mean.*(feat{1}*W_mean{1});
    test_input = feat{1}' ;

    for m = 2:D
        feat{m} = features(X,m,M) ;
        mean = mean.*(feat{m}*W_mean{m});
        test_input = khatri_rao(test_input, feat{m}');
    end
    mean = real(mean * norms) ;


    % Unscented covariance
    [mUT, PUT] = uns(W_mean,W_cov,norms) ;
    
    var_CI = diag(test_input' * PUT * test_input) ;

    % for n = 1:N
    % var_CI(n) = f(n,:) * PUT * f(n,:)' ;
    % end


    var_PI = sigmae + var_CI ;
end