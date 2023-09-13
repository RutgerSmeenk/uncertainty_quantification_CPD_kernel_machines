% This study regards a dataset which is learned and where predictions are
% made and the uncertainty in the predictions is quantified 

clear all

rng(19) ;
M = 8;
R = 20;
maxIte = 10;

warning('off','all');


%% Processing data

X = readmatrix('climate.csv'); 

XTest = X ;

perm = randperm(size(X,1));
X = X(perm,:);
X = X(1:floor(0.7*size(X,1)),:);
Y = X(:,end) ;
X = X(:,1:end-1);

YMean = mean(Y);    YStd = std(Y);
XMin = min(X);  XMax = max(X);
Y = (Y-YMean)./YStd ;
X = (X-XMin)./(XMax-XMin);


XTest = XTest(perm,:);
XTest = XTest(floor(0.7*size(XTest,1))+1:end,:);

YTest = XTest(:,end);
XTest = XTest(:,1:end-1);

XTest = (XTest-XMin)./(XMax-XMin);
YTest = (YTest-YMean)./YStd;


variableNames = {'X', 'XTest', 'Y', 'YTest'};

for i = 1:length(variableNames)
    filePath1 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 4 - one dim Bay\exp_4b', [variableNames{i}, '.mat']);
    save(filePath1, variableNames{i});
    filePath2 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 7 - one dim Core\exp_7b', [variableNames{i}, '.mat']);
    save(filePath2, variableNames{i});
end


%% K fold cross validation
[mean_error_all, conf_error_all, lambda_folds] = cross_dat(X, Y, M, R, maxIte) ;

% Find the minimum element in the matrix
minValue = min(mean_error_all);

% Calculate the threshold for the difference
threshold = 0.5 * minValue;

% Find the indices of the minimum elements that meet the condition
index = find(mean_error_all<= minValue + threshold & mean_error_all >= minValue - threshold);

% Find the minimum element of the other matrix considering only the indicesOtherMatrix
min_val_hyper = min(conf_error_all(index));

% Find indices for hyperparameters
index_lambda = find(conf_error_all == min_val_hyper);

% Find hyperparameters
lambda = lambda_folds(index_lambda) ;

%% Making predictions

% Obtaining the weights
[W, norms, reg] = CP_ALS_weights(X, Y, M, R, lambda, maxIte) ;

% Making predictions
[pred, C_test, C_train, regu] = CP_ALS_predict(XTest, W, norms, M, X, reg) ;

% Estimating noise variance
sigmae = var(Y - CP_ALS_predict(X, W, norms, M, X, reg), 1) ;

% Quantifiying the uncertainty
[var_CI, var_PI] = delta_method(C_train, C_test, lambda, sigmae, regu) ;

pred = real(pred) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

CI_lower = pred - (2*sqrt(var_CI)) ;
CI_upper = pred + (2*sqrt(var_CI)) ;

%% Plotting

a = [XTest, YTest, CI_lower, CI_upper, pred] ; 

Plot = sortrows(a);

true_model = [Plot(:,1),Plot(:,3)];
pred_train = Plot(:,2);


figure(1)

l = plot(Plot(:,1),Plot(:,2),'go',Plot(:,1),Plot(:,5),'b',Plot(:,1),Plot(:,3),'r--',Plot(:,1),Plot(:,4),'r--');


l(1).MarkerSize = 2; 
%l(2).MarkerSize = 12;
title('Delta','interpreter','latex')
xlabel('Feature','interpreter','latex')
ylabel('Function','interpreter','latex')
%ylim([-2 2])
%xlim([-100 100])
legend('Test input','Prediction','Prediction Interval','','Location','southeast','interpreter','latex')



%% Plotting ALL

run("C:\Users\rutge\Documents\MATLAB\Thesis\Exp 4 - one dim Bay\exp_4b\exp_4b.m")
run("C:\Users\rutge\Documents\MATLAB\Thesis\Exp 7 - one dim Core\exp_7b\exp_7b.m")


a = [XTest, YTest, CI_lower, CI_upper, CI_lower_Bay, CI_upper_Bay, CI_lower_Core, CI_upper_Core, pred, pred_Bay, pred_Core] ;
Plot = sortrows(a);

ref_model = [Plot(:,1),Plot(:,5)];
m_err_1 = Plot(:,2);
m_err_2 = Plot(:,5);


figure(4);
plot(Plot(:,1),Plot(:,2), 'o', 'Color', 'k','MarkerSize', 2);
hold on;
plot(Plot(:,1),Plot(:,3), '--', 'Color', 'b', 'LineWidth', 1);
plot(Plot(:,1),Plot(:,4), '--', 'Color', 'b', 'LineWidth', 1);
plot(Plot(:,1),Plot(:,5), '--', 'Color', 'g', 'LineWidth', 1);
plot(Plot(:,1),Plot(:,6), '--', 'Color', 'g', 'LineWidth', 1);
plot(Plot(:,1),Plot(:,7), '--', 'Color', 'r', 'LineWidth', 1);
plot(Plot(:,1),Plot(:,8), '--', 'Color', 'r', 'LineWidth', 1);
plot(Plot(:,1),Plot(:,9), 'Color', 'b', 'LineWidth', 1);
plot(Plot(:,1),Plot(:,10), 'Color', 'g', 'LineWidth', 1);
plot(Plot(:,1),Plot(:,11), 'Color', 'r', 'LineWidth', 1);
hold off;


%title('Predictions with confidence intervals','interpreter','latex')
xlabel('Feature','interpreter','latex')
ylabel('Function','interpreter','latex')
ylim([-1.6 1.6])
xlim([min(XTest) max(XTest)])
legend('Test input','Delta $2 \sigma$ prediction interval','','Bayesian $2 \sigma$ prediction interval','','SBC $2 \sigma$ prediction interval','','Prediction delta','Prediction Bayesian', 'Prediction SBC','Location','northwest','interpreter','latex')


