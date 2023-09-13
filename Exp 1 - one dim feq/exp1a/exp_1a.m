% This experiment describes a true function that is within the model class
% the true function is added with noise FITTING CPD DATA

clear all
%% 1. Generating magic tyre formula data 

rng(18) ;
SNR = 20 ;
M = 8;
R = 20;
lambda = 1e-6;
maxIte = 10;


warning('off','all');


% True output model: Magic tyre formula
min_train = -100 ;
max_train = 100 ;
kappa_train = linspace(min_train,max_train,6000) ;
kappa_test = linspace(-100,100,6000) ;
Fz = 1000 ;
Fx_train = [];
Fx_test = [];

for kappa = kappa_train

Fx = tyre(kappa, Fz) ;
Fx_train = [Fx_train Fx];

end

for kappa = kappa_test

Fx = tyre(kappa, Fz) ;
Fx_test = [Fx_test Fx];

end

%% 2. Generating data and making predictions

% Training data for reference model
X = [kappa_train' Fx_train'] ;
perm = randperm(size(X,1));
X = X(perm,:);
X = X(1:floor(0.6*size(X,1)),:);
Y = X(:,end) ;
X = X(:,1:end-1);

% Normalizing data
YMean = mean(Y);    YStd = std(Y);
XMin = min(X);  XMax = max(X);
Y = (Y-YMean)./YStd ;
X = (X-XMin)./(XMax-XMin);


% Test data for reference model
XTest = [kappa_test' Fx_test'] ;
XTest = XTest(perm,:);
XTest = XTest(floor(0.6*size(XTest,1))+1:end,:);
YTest = XTest(:,end);
XTest = XTest(:,1:end-1);
XTest = (XTest-XMin)./(XMax-XMin);
YTest = (YTest-YMean)./YStd;

% Obtaining weights 
%[W, norms] = CP_ALS_weights(X,Y,M,R,lambda,maxIte);
[W, norms, reg] = CP_ALS_weights(X, Y, M, R,lambda, maxIte) ;
    
% Prediction
pred = CP_ALS_predict(XTest,W,norms,M,X,reg) ;
pred = real(pred) ;

%% 3. Using predicitions as new data

% New training data
X = [XTest pred] ;
%M=32;

% Training data
perm = randperm(size(X,1));
X = X(perm,:);
Truth = X ;
Truth_train = Truth(1:floor(0.7*size(Truth,1)),:);


% Adding noise
noise_norm = norm(X(:,2))/(10^(SNR/20));
noise      = randn(size(X(:,2)));
noise      = noise/norm(noise)*noise_norm;
X(:,2)     = X(:,2) + noise ;

sigmae = var(noise,1) ;

XTest = X ; 

X = X(1:floor(0.7*size(X,1)),:);
Y = X(:,end) ;
X = X(:,1:end-1);

% Normalizing data
YMean = mean(Y);    YStd = std(Y);
XMin = min(X);  XMax = max(X);
X = (X-XMin)./(XMax-XMin);

% Test data and Truth data
XTest = XTest(floor(0.7*size(XTest,1))+1:end,:);
Truth_test = Truth(floor(0.7*size(Truth,1))+1:end,:);
Truth_test = Truth_test(:,end) ;
YTest = XTest(:,end) ;
XTest = XTest(:,1:end-1) ;


% Normalizing data
XTest = (XTest-XMin)./(XMax-XMin);

variableNames = {'X', 'XTest', 'Y', 'YTest', 'Truth_train', 'Truth_test', 'sigmae'};

for i = 1:length(variableNames)
    filePath1 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 4 - one dim Bay\exp_4a', [variableNames{i}, '.mat']);
    save(filePath1, variableNames{i});
    filePath2 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 7 - one dim Core\exp_7a', [variableNames{i}, '.mat']);
    save(filePath2, variableNames{i});
end


%% K fold cross validation
[mean_error_all, conf_error_all, lambda_folds] = cross_dat(X, Y, M, R, maxIte) ;

% Find the minimum element in the matrix
minValue = min(mean_error_all);

% Calculate the threshold for the difference
threshold = 0.5 * minValue;

% Find the indices of the minimum elements that meet the condition
index_conf = find(mean_error_all<= minValue + threshold & mean_error_all >= minValue - threshold);

% Find the minimum element of the other matrix considering only the indicesOtherMatrix
min_val_hyper = min(conf_error_all(index_conf));

% Find indices for hyperparameters
index_lambda = find(conf_error_all == min_val_hyper);

% Find hyperparameters
lambda = lambda_folds(index_lambda) ;
%lamda = 1e-16 ;
%% Making predictions

% Obtaining the weights
[W_2, norms, reg] = CP_ALS_weights(X, Y, M, R, lambda, maxIte) ;

% Prediction
[pred_2, C_test, C_train, regu] = CP_ALS_predict(XTest, W_2, norms, M, X, reg) ;


% Quantifiying the uncertainty
[var_CI, var_PI] = delta_method(C_train, C_test, lambda, sigmae, regu) ;

pred_2 = real(pred_2) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

% Uncertainty intervals
CI_lower = pred_2 - (2*sqrt(var_CI)) ;
CI_upper = pred_2 + (2*sqrt(var_CI)) ;

%% 4. Plotting

a = [XTest, YTest, CI_lower, CI_upper, pred_2, Truth_test] ;
Plot = sortrows(a);

ref_model = [Plot(:,1),Plot(:,5)];
m_err_1 = Plot(:,2);
m_err_2 = Plot(:,5);


figure(1)
l = plot(Plot(:,1),Plot(:,2),'go',Plot(:,1),Plot(:,5),'b',Plot(:,1),Plot(:,3),'r--',Plot(:,1),Plot(:,4),'r--',Plot(:,1),Plot(:,6),'c');

l(1).MarkerSize = 2; 

title('Delta','interpreter','latex')
xlabel('Feature','interpreter','latex')
ylabel('Function','interpreter','latex')
ylim([-1.6 1.6])
xlim([min(XTest) max(XTest)])
legend('Test input','Prediction','Delta $2 \sigma$ confidence interval','','True function','Location','southeast','interpreter','latex')

clear c

N_Test = length(YTest) ;
for n = 1:N_Test
    if (Truth_test(n) > CI_lower(n)) && (Truth_test(n) < CI_upper(n))
    c(n) = 1 ;
    else
    c(n) = 0 ;
    end
end


PICP = 1/N_Test * sum(c) ;
mu = 1 - 0.05 ;
eta = 50;

if PICP >= mu 
    gamma = 0 ;
else 
    gamma = 1 ;
end


%% Plotting ALL

run("C:\Users\rutge\Documents\MATLAB\Thesis\Exp 4 - one dim Bay\exp_4a\exp_4a.m")
run("C:\Users\rutge\Documents\MATLAB\Thesis\Exp 7 - one dim Core\exp_7a\exp_7a.m")


a = [XTest, YTest, CI_lower, CI_upper, CI_lower_Bay, CI_upper_Bay, CI_lower_Core, CI_upper_Core, pred_2, pred_Bay, pred_Core, Truth_test] ;
Plot = sortrows(a);

figure(4);
ax = axes;

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
plot(Plot(:,1),Plot(:,12), 'Color', 'c', 'LineWidth', 1);
hold off;


options.axes.Names = {'Position','XLim'};
options.axes.Values = {[.2 .6 .3 .3],[0.7,0.8]};
% Name-Value pairs for the rectangle:
options.rectangle.Names = {};
options.rectangle.Values = {};
% Name-Value pairs for the arrows:
options.arrows.Names = {'HeadLength','HeadWidth'};
options.arrows.Values = {6,6};
% call the function with options:
[zoom_utils] = zoom_plot(ax,options);

%title('Predictions with confidence intervals','interpreter','latex')
xlabel('Feature','interpreter','latex')
ylabel('Function','interpreter','latex')
ylim([-1.6 1.6])
xlim([min(XTest) max(XTest)])
legend('Test input','Delta $2 \sigma$ confidence interval','','Bayesian $2 \sigma$ confidence interval','','SBC $2 \sigma$ confidence interval','','Prediction delta','Prediction Bayesian', 'Prediction SBC','True function','Location','southeast','interpreter','latex')



