%% Pure power features FULL
% function Z = features(X,M,lenthscale)
% Z = X.^(0:M-1);
% end

%% Pure power features
function Z = features(X,m,M)
Z = [ones(size(X)), X.^(2.^(m-1))];
end

%% Fourier features FULL
% function Z = features(X,M,lengthscale)
% P=1;
% Z = exp(1j*2*pi*X.*(-M/2:(M/2-1))/P);
% end

%% Fourier features
% function Z = features(X,m,M)
% P = 1 ;
% Z = [exp(-1i*pi*X*M/(log2(M)*P)), exp(1i*pi*(-X*M/log2(M)+2*X.*(2^(m-1)))/P)];
% end


