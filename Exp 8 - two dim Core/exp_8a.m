%% Ripley & Four class Dataset Plots

rng(1);
M = 10;
R = 20;
maxIte = 10;
sto_core = 2;
%lambda = 1e-7;

%% Processing data
% close all
% %rng('default');
% X = readmatrix('ripley_training.csv');
% N = size(X,1);
% X = X(randperm(size(X,1)),:);
% Y = (X(:,end)==1)-(X(:,end)==2);
% X = X(:,1:2);
% XMin = min(X);  XMax = max(X);
% X = (X-XMin)./(XMax-XMin);
% 
% NPlot = 100;
% X1Plot = linspace(0,1,NPlot);
% [X1Plot,X2Plot] = meshgrid(X1Plot,X1Plot);
% XPlot = [X1Plot(:),X2Plot(:)];
% 
% XTest = readmatrix('ripley_training.csv');
% NTest = size(XTest,1);
% XTest = XTest(randperm(size(XTest,1)),:);
% YTest = (XTest(:,end)==1)-(XTest(:,end)==2);
% XTest = XTest(:,1:2);
% XTestMin = min(XTest);  XTestMax = max(XTest);
% XTest = (XTest-XTestMin)./(XTestMax-XTestMin);

load("X.mat","XTest.mat","Y.mat","YTest.mat","XPlot.mat","lambda_delta.mat")

%% K fold cross validation
[mean_error_all_Core, conf_error_all_Core, sigma_folds_Core, noise_folds_Core] = cross_Core(X, Y, M, R, maxIte, sto_core, lambda_delta) ;

% Find the minimum element in the matrix
minValue = min(mean_error_all_Core);

% Calculate the threshold for the difference
threshold = 0.1 * minValue;

% Find the indices of the minimum elements that meet the condition
index_conf = find(mean_error_all_Core > minValue + threshold);

conf_error_all_Core(index_conf) = 10e20 ;

matrix_conf = reshape(conf_error_all_Core,[length(sigma_folds_Core), length(noise_folds_Core)]) ;

min_value = min(matrix_conf(:)) ;

% Find the indices of the minimum value using find
[index_sigma, index_sigmae] = find(matrix_conf == min_value);

% Extract elements
sigma_Core = sigma_folds_Core(index_sigma);
sigmae_Core = noise_folds_Core(index_sigmae);

%% Making predictions

% Obtaining the weights
YPlot = ones(size(XPlot,1),1) ;

[W, norms] = CP_ALS_weights(X, Y, M, R, lambda_delta, maxIte);
[prediction, var_CI, var_PI] = CP_ALS_Core_predict(XPlot, W, norms, M, sigma_Core, sto_core, Y, X, sigmae_Core);

prediction = real(prediction) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

pred_Core = sign(prediction);

% Uncertainty intervals
CI_lower_Core = sign(prediction - (2*sqrt(var_PI))) ;
CI_upper_Core = sign(prediction + (2*sqrt(var_PI))) ;

% variableNames = {'CI_lower_Core', 'CI_upper_Core', 'pred_Core'};
% 
% for i = 1:length(variableNames)
%     filePath1 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 2 - two dim Freq\Exp 2.2 classification', [variableNames{i}, '.mat']);
%     save(filePath1, variableNames{i});
% end

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














