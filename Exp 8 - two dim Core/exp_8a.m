%% Ripley & Four class Dataset Plots

rng(5);
M = 10;
R = 20;
maxIte = 10;
sto_core = 2;
%lambda = 1e-7;

%% Processing data

load("X.mat","XTest.mat","Y.mat","YTest.mat","XPlot.mat")

%% K fold cross validation
[mean_error_all_Core, conf_error_all_Core, sigma_folds_Core, lambda_folds_Core] = cross_Bay_dat(X, Y, M, R, maxIte, sto_core) ;

% Find the minimum element in the matrix
minValue = min(mean_error_all_Core);

% Calculate the threshold for the difference
threshold = 0.5 * minValue;

% Find the indices of the minimum elements that meet the condition
index_conf = find(mean_error_all_Core > minValue + threshold);

conf_error_all_Core(index_conf) = 10e4 ;

matrix_conf = reshape(conf_error_all_Core,[length(sigma_folds_Core), length(lambda_folds_Core)]) ;

min_value = min(matrix_conf(:)) ;

% Find the indices of the minimum value using find
[index_sigma, index_lambda] = find(matrix_conf == min_value);

% Extract elements
sigma_Core = sigma_folds_Core(index_sigma);
lambda_Core = lambda_folds_Core(index_lambda);

%% Making predictions

% Obtaining the weights
YPlot = ones(size(XPlot,1),1) ;

[W, norms] = CP_ALS_weights(X, Y, M, R, lambda_Core, maxIte);
[prediction, var_CI, var_PI] = CP_ALS_Core_predict(XPlot, W, norms, M, sigma_Core, sto_core, Y, X);

prediction = real(prediction) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

pred_Core = sign(prediction);

% Uncertainty intervals
CI_lower_Core = sign(prediction - (2*sqrt(var_PI))) ;
CI_upper_Core = sign(prediction + (2*sqrt(var_PI))) ;

%% Plotting

    figure(3);
%    fig = gcf;
    hold on
    % s1 = scatter(X(Y==1,1),X(Y==1,2),36,[238, 28, 37]/255,'filled');
    % s2 = scatter(X(Y==-1,1),X(Y==-1,2),36,[1, 90, 162]/255,'filled');
    s1 = scatter(X(Y==1,1),X(Y==1,2),36,[238, 28, 37]/255,'filled');
    s2 = scatter(X(Y==-1,1),X(Y==-1,2),36,[1, 90, 162]/255,'filled');

    s1.MarkerFaceAlpha = 0.25;
    s2.MarkerFaceAlpha = 0.25;
    f1 = contour(X1Plot,X2Plot,reshape(pred_Core,size(X1Plot)),[0 0],'black','LineWidth',1.5);
    %c1 = contour(X1Plot,X2Plot,reshape(scorePlotCP,size(XPlot)),[0 0],'black','LineWidth',1.5);
    %c2 = contour(X1Plot,X2Plot,reshape(scorePlotHilbert,size(X1Plot)),[0 0],'black','LineWidth',1.5,'LineStyle','--');
    %plot(c1(1,2:end)',c1(2,2:end)')
    f2 = contour(X1Plot,X2Plot,reshape(CI_lower_Core,size(X1Plot)),[0 0],'black','LineWidth',1.5,'LineStyle','--');
    f3 = contour(X1Plot,X2Plot,reshape(CI_upper_Core,size(X1Plot)),[0 0],'black','LineWidth',1.5,'LineStyle','--');

    xticks(-1:0.2:1);
    yticks(-1:0.2:1);
    %legend('Class A','Class B','Classifier','Confidence interval','Location','southeast','interpreter','latex')
    axis equal
    axis off
    hold off
    %filename = 'banana'+string(M^2)+'frequencies'+string(R)+'rank'+'.pdf';


%% Evaluating results


% %[W, norms] = CP_ALS_Bay_weights(X, Y, M, R, lambda, maxIte, sto_core);
% [prediction, var_CI, var_PI] = CP_ALS_Bay_predict(XTest, W, norms, M, sigma, sto_core, Y, X);
% 
% prediction = real(prediction) ;
% var_CI = real(var_CI) ;
% var_PI = real(var_PI) ;
% 
% CI_low = real(prediction - (2*sqrt(var_PI))) ;
% CI_up = real(prediction + (2*sqrt(var_PI))) ;
% 
% CI_lower_Core = sign(CI_low) ;
% CI_upper_Core = sign(CI_up) ;
% 
% clear c
% N_Test = length(YTest) ;
% for n = 1:N_Test
%     if (YTest(n) ~= CI_lower_Core(n)) && (YTest(n) ~= CI_upper_Core(n))
%     c(n) = 1 ;
%     else
%     c(n) = 0 ;
%     end
% end
% 
% 
% PICP = 1/N_Test * sum(c) ;
% %mu = 1 - 0.05 ;
% mu = 0.05 ;
% eta = -50;
% 
% if PICP >= mu 
%     gamma = 1 ;
% else 
%     gamma = 0 ;
% end
% %gamma=0;
% 
% MPIW = 1/N_Test * sum(CI_up - CI_low) ;
% % CWC = MPIW * (1 + gamma * exp(-eta * (PICP - mu))) ;
% CWC = MPIW + gamma * exp(-eta * (PICP - mu)) ;














