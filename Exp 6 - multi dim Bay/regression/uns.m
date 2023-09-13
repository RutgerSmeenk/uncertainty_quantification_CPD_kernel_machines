function [mUT, PUT] = uns(W_mean, W_cov, norms)

%% Computing sigma points

% W_mean = W_mean_2 ;
% W_cov = W_cov_2 ;


% Stacking mean and covariance
D = length(W_mean) ;
[M,R] = size(W_mean{1}) ;


m = vec(W_mean{1}) ;
P = W_cov{1} ;
M_fac = numel(W_mean{1}) * D ;

alpha = 0.001 ;
beta = 2 ;
kappa = 3 - M_fac ;

lambdaa = alpha^2 * (M_fac + kappa) - M_fac ;
fac = sqrt(M_fac + lambdaa) ;



for n = 2:D
%W_cov{n} = diag(diag(W_cov{n})) ;
m = [m; vec(W_mean{n})] ;
P = real(blkdiag(P,W_cov{n})) ;
end

% sigma point 0
x_0 = m ;

% sigma points 1...M_fac
% sqrt_P = sqrt(P) ;
sqrt_P = chol(P,'lower') ;

for i = 1:M_fac
x_i{i} = m + (fac * sqrt_P(:,i)) ;
end 

% sigma points M+1...2M+1

for i = 1:M_fac
x_i_M{i} = m - (fac * sqrt_P(:,i)) ;
end 



%% Transformed sigma points

% Reshaping sigma points

x_0 = reshape(x_0,M*R,[]) ;

for i = 1:M_fac
x_i{i} = reshape(x_i{i},M*R,[]) ;
x_i_M{i} = reshape(x_i_M{i},M*R,[]) ;

for d = 1:D
x_0_res{d} = reshape(x_0(:,d), [M R]) ;

x_i_res{i}{d} = reshape(x_i{i}(:,d), [M R]) ;

x_i_M_res{i}{d} = reshape(x_i_M{i}(:,d), [M R]) ;

end
end

% Applying the transformation to the reshaped sigma points

W_x_0 = x_0_res{1};

for i = 1:M_fac
    W_x_i{i} = x_i_res{i}{1} ;
    W_x_i_M{i} = x_i_M_res{i}{1} ;

for d = 1:D-1
    if i == 1
W_x_0 = khatri_rao(W_x_0, x_0_res{d+1});
    end
W_x_i{i} = khatri_rao(W_x_i{i}, x_i_res{i}{d+1});
W_x_i_M{i} = khatri_rao(W_x_i_M{i}, x_i_M_res{i}{d+1});
end
W_x_i{i} = W_x_i{i} * ones(R,1) ;
W_x_i_M{i} = W_x_i_M{i} * ones(R,1) ;
end

W_x_0 = W_x_0 * ones(R,1) ;

%% Computing unscented mean and covariance

% Setting the weights

w_0_m = lambdaa / (M_fac + lambdaa) ;
w_0_P = w_0_m + (1 - alpha^2 + beta) ;
w_i_m = 1 / (2 * (M_fac + lambdaa)) ;
w_i_P = w_i_m ;

% Mean and covariance
matrix_second = cell2mat(W_x_i) ;
matrix_third = cell2mat(W_x_i_M) ;

m_first = w_0_m * W_x_0 ;
m_second = sum(w_i_m * matrix_second,2) ;
m_third = sum(w_i_m * matrix_third,2) ;

mUT = m_first + m_second + m_third ;

mUT_stacked = repmat(mUT, 1, M_fac) ;
dif_sec = matrix_second - mUT_stacked ;
dif_third = matrix_third - mUT_stacked ;

P_first = w_0_P * (W_x_0 - mUT) * (W_x_0 - mUT)' ;
P_second =  w_i_P * (dif_sec * dif_sec');
P_third = w_i_P * (dif_third * dif_third');

PUT = P_first + P_second + P_third ;



% P_second = zeros(M_fac,M_fac) ;
% P_second = zeros(M^D,M^D) ;
% for i = 1:M_fac
% P_second = P_second + (w_i_P * (W_x_i{i} - mUT) * (W_x_i{i} - mUT)') ;              
% end
% 
% % P_third = zeros(M_fac,M_fac) ;
% P_third = zeros(M^D,M^D) ;
% for i = 1:M_fac
% P_third = P_third + (w_i_P * (W_x_i_M{i} - mUT) * (W_x_i_M{i} - mUT)') ;              
% end

% PUT = 1/(2*M_fac + 1) * (P_first + P_second + P_third) ;
% PUT = P_first + P_second + P_third ;












% Setting the weights


% % Mean and covariance
% m_second = cell2mat(W_x_i) ;
% m_third = cell2mat(W_x_i_M) ;
% 
% m_first = W_x_0 ;
% m_second = sum(m_second,2) ;
% m_third = sum(m_third,2) ;
% 
% mUT = 1/(2*M_fac + 1) * (m_first + m_second + m_third) ;
% 
% 
% P_first = (W_x_0 - mUT) * (W_x_0 - mUT)' ;
% 
% % P_second = zeros(M_fac,M_fac) ;
% P_second = zeros(2^D,2^D) ;
% for i = 1:M_fac
% P_second = P_second + ((W_x_i{i} - mUT) * (W_x_i{i} - mUT)') ;              
% end
% 
% % P_third = zeros(M_fac,M_fac) ;
% P_third = zeros(2^D,2^D) ;
% for i = 1:M_fac
% P_third = P_third + ((W_x_i_M{i} - mUT) * (W_x_i_M{i} - mUT)') ;              
% end
% 
% PUT = 1/(2*M_fac + 1) * (P_first + P_second + P_third) ;



end









% aa = zeros(4,4) ;
% for i = 1:16
% aa = aa + (bb - mUT) * (bb - mUT)' ;              
% end
% 
% 
% aa = zeros(4,4) ;
% for i = 1:16
% aa = aa + (bb - mUT) * (bb - mUT)' ;              
% end


