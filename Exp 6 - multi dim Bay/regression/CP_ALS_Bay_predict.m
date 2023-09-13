function [mean, var_CI, var_PI] = CP_ALS_Bay_predict(X, W_mean, W_cov, norms, M, sigmae)
    [N,~] = size(X);
    %R = size(W_mean{m},2) ;
    [~, D] = size(X);
    mean = ones(N,1); 
    feat = cell(1,D) ;
    %f = cell(N,1) ;

    feat{1} = features(X(:,1),M) ;
    mean = mean.*(feat{1}*W_mean{1});
    test_input = feat{1}' ;

    for d = 2:D
        feat{d} = features(X(:,d),M) ;
        mean = mean.*(feat{d}*W_mean{d});
        test_input = khatri_rao(test_input, feat{d}');
    end
    mean = real(mean * norms) ;

    % Unscented covariance
    %W_cov{1} = W_cov{1}.*0.05;

    [mUT, PUT] = uns(W_mean,W_cov,norms) ;
    
    %hoi = test_input * mUT ;
    var_CI = diag(test_input' * PUT * test_input) ;

    % for n = 1:N
    % var_CI(n) = f(n,:) * PUT * f(n,:)' ;
    % end

    var_PI = sigmae + var_CI ;
end