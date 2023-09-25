%% Ripley & Four class Dataset Plots

rng(1);
M = 10;
R = 20;
maxIte = 10;
%lambda = 1e-7;

%% Processing data

% NPlot = 100;
% X1Plott = linspace(0,1,NPlot);
% [X1Plot,X2Plot] = meshgrid(X1Plott,X1Plott);
% XPlot = [X1Plot(:),X2Plot(:)];

load("X.mat","XTest.mat","Y.mat","YTest.mat","XPlot.mat")

%% K fold cross validation
[mean_error_all_Bay, conf_error_all_Bay, sigma_folds_Bay, noise_folds_Bay] = cross_Bay(X, Y, M, R, maxIte) ;

% Find the minimum element in the matrix
minValue = min(mean_error_all_Bay);

% Calculate the threshold for the difference
threshold = 0.1 * minValue;

% Find the indices of the minimum elements that meet the condition
% index_conf = find(mean_error_all<= minValue + threshold & mean_error_all >= minValue - threshold);
index_conf = find(mean_error_all_Bay > minValue + threshold);

conf_error_all_Bay(index_conf) = 10e20 ;

matrix_conf = reshape(conf_error_all_Bay,[length(sigma_folds_Bay), length(noise_folds_Bay)]) ;

min_value = min(matrix_conf(:)) ;

% Find the indices of the minimum value using find
[index_sigma, index_sigmae] = find(matrix_conf == min_value);

% Extract elements
sigma_Bay = sigma_folds_Bay(index_sigma);
sigmae_Bay = noise_folds_Bay(index_sigmae);

%% Making predictions

% Obtaining the weights
YPlot = ones(size(XPlot,1),1) ;
[W_mean, W_cov, norms] = CP_ALS_Bay_weights(X, Y, M, R, maxIte, sigma_Bay, sigmae_Bay) ;

%[mUT, PUT] = uns(W_mean,W_cov,norms) ;

% Prediction
[prediction_Bay, var_CI, var_PI] = CP_ALS_Bay_predict(XPlot, W_mean, W_cov, norms, M) ;

prediction_Bay = real(prediction_Bay) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

pred_Bay = sign(prediction_Bay);

% Uncertainty intervals
CI_lower_Bay = sign(prediction_Bay - (2*sqrt(var_PI))) ;
CI_upper_Bay = sign(prediction_Bay + (2*sqrt(var_PI))) ;

% variableNames = {'CI_lower_Bay', 'CI_upper_Bay','pred_Bay'};
% 
% for i = 1:length(variableNames)
%     filePath1 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 2 - two dim Freq\Exp 2.2 classification', [variableNames{i}, '.mat']);
%     save(filePath1, variableNames{i});
% end



%% Plotting

    figure(2);
%    fig = gcf;
    hold on
    % s1 = scatter(X(Y==1,1),X(Y==1,2),36,[238, 28, 37]/255,'filled');
    % s2 = scatter(X(Y==-1,1),X(Y==-1,2),36,[1, 90, 162]/255,'filled');
    s1 = scatter(X(Y==1,1),X(Y==1,2),36,[238, 28, 37]/255,'filled');
    s2 = scatter(X(Y==-1,1),X(Y==-1,2),36,[1, 90, 162]/255,'filled');

    s1.MarkerFaceAlpha = 0.25;
    s2.MarkerFaceAlpha = 0.25;
    e1 = contour(X1Plot,X2Plot,reshape(pred_Bay,size(X1Plot)),[0 0],'black','LineWidth',1.5);
    %c1 = contour(X1Plot,X2Plot,reshape(scorePlotCP,size(XPlot)),[0 0],'black','LineWidth',1.5);
    %c2 = contour(X1Plot,X2Plot,reshape(scorePlotHilbert,size(X1Plot)),[0 0],'black','LineWidth',1.5,'LineStyle','--');
    %plot(c1(1,2:end)',c1(2,2:end)')
    e2 = contour(X1Plot,X2Plot,reshape(CI_lower_Bay,size(X1Plot)),[0 0],'black','LineWidth',1.5,'LineStyle','--');
    e3 = contour(X1Plot,X2Plot,reshape(CI_upper_Bay,size(X1Plot)),[0 0],'black','LineWidth',1.5,'LineStyle','--');

    xticks(-1:0.2:1);
    yticks(-1:0.2:1);
    %legend('Class A','Class B','Classifier','Confidence interval','Location','southeast','interpreter','latex')
    axis equal
    axis off
    hold off
    %filename = 'banana'+string(M^2)+'frequencies'+string(R)+'rank'+'.pdf';



%% Evaluating results



% [W_mean, W_cov, norms] = CP_ALS_Bay_weights(X, Y, M, R, maxIte, sigma_Bay, sigmae_Bay);
% [prediction, var_CI, var_PI] = CP_ALS_Bay_predict(XTest, W_mean, W_cov, norms, M, sigmae_Bay);
% 
% %[var_CI, var_PI] = delta_method(C_train, C_test, lambda, sigmae, regu) ;
% 
% prediction = real(prediction) ;
% var_CI = real(var_CI) ;
% var_PI = real(var_PI) ;
% 
% CI_low = real(prediction - (2*sqrt(var_PI))) ;
% CI_up = real(prediction + (2*sqrt(var_PI))) ;
% 
% CI_lower = sign(CI_low) ;
% CI_upper = sign(CI_up) ;
% 
% N_Test = length(YTest) ;
% for n = 1:N_Test
%     if (YTest(n) == CI_lower(n)) || (YTest(n) == CI_upper(n))
%     c(n) = 1 ;
%     else
%     c(n) = 0 ;
%     end
% end
% 
% 
% PICP = 1/N_Test * sum(c) ;
% mu = 1 - 0.05 ;
% %mu = 0.05 ;
% eta = 50 ;
% 
% if PICP >= mu 
%     gamma = 0 ;
% else 
%     gamma = 1 ;
% end
% %gamma=0;
% 
% 
% % MPIW = (1/N_Test * sum(CI_upper - CI_lower)) / abs(CI_up - CI_low) ;
% MPIW = 1/N_Test * sum(CI_up - CI_low) ;
% 
% %CWC = MPIW * (1 + gamma * exp(-eta * (PICP - mu))) ;
% CWC = MPIW + gamma * exp(-eta * (PICP - mu)) ;












