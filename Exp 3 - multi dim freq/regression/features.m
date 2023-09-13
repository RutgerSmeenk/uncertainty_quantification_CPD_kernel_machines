%% Pure power features FULL
function Mati = features(X,M)
Mati = X.^(0:M-1);
end

%% Pure power features
% function Mati = features(X,m,M)
% Mati = [ones(size(X)), X.^(2.^(m-1))];
% end

%% Fourier features FULL
% function Mati = features(X,M)
% P=1;
% Mati = exp(1j*2*pi*X.*(-M/2:(M/2-1))/P);
% end

%% Fourier features
% function Mati = features(X,m,M)
% P = 1 ;
% Mati = [exp(-1i*pi*X*M/(log2(M)*P)), exp(1i*pi*(-X*M/log2(M)+2*X.*(2^(m-1)))/P)];
% end


