function [var_CI, var_PI] = delta_method(C_train, C_test, lambda, sigmae, regu)

% Parameter covariance

regularization = 2 * lambda * blkdiag(regu{:}) ;
V_grad_tot = cell2mat(C_train) ;
g = cell2mat(C_test) ;

jj = V_grad_tot' * V_grad_tot ;
inv_F = pinv(2*(jj) + regularization) ;
inv_F_gen = sigmae * inv_F * 2*jj * inv_F;

% Prediction error variance

var_CI = diag(g * inv_F_gen * g') ;

var_PI = sigmae + var_CI ;


end
