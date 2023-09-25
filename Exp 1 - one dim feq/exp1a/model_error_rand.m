% This experiment describes a true function that is within the model class
% the true function is added with noise FITTING CPD DATA

clear all
%% 1. Generating magic tyre formula data 

%rng(10) ;
SNR = 2 ;
M = 4;
R = 20;
lambda = 0;
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
XTest_data = (XTest-XMin)./(XMax-XMin);
YTest = (YTest-YMean)./YStd;

% Obtaining weights 
%[W, norms] = CP_ALS_weights(X,Y,M,R,lambda,maxIte);
[W, norms, reg] = CP_ALS_weights(X, Y, M, R,lambda, maxIte) ;
    
% Prediction
pred = CP_ALS_predict(XTest_data,W,norms,M,X,reg) ;
pred = real(pred) ;


for run = 1:20

%% 3. Using predicitions as new data

% New training data
X = [XTest_data pred] ;

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
Truth = pred ;

% Normalizing data
% YMean = mean(Y);    YStd = std(Y);
XMin = min(X);  XMax = max(X);
%Y = (Y-YMean)./YStd ;
X = (X-XMin)./(XMax-XMin);
%Truth_train = (Truth_train-YMean)./YStd ;

YMean = mean(Y);    YStd = std(Y);
%Y = (Y-YMean)./YStd ;
TruthMean = mean(Truth);    TruthStd = std(Truth);
%Truth = (Truth-TruthMean)./TruthStd ;

% Test data and Truth data
XTest = XTest(floor(0.7*size(XTest,1))+1:end,:);
Truth_test = Truth(floor(0.7*size(Truth,1))+1:end,:);
Truth_test = Truth_test(:,end) ;
YTest = XTest(:,end) ;
XTest = XTest(:,1:end-1) ;


% Normalizing data
XMinTest = min(XTest);  XMaxTest = max(XTest);
XTest = (XTest-XMinTest)./(XMaxTest-XMinTest);
YMeanTest = mean(YTest);    YStdTest = std(YTest);
YTest = (YTest-YMeanTest)./YStdTest;
%Truth_test = (Truth_test-YMean)./YStd;


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
lambda = 0 ;
%% Making predictions

% Obtaining the weights
[W_2, norms, reg] = CP_ALS_weights(X, Y, M, R, lambda, maxIte) ;

% Prediction
[pred_2, C_test, C_train, regu] = CP_ALS_predict(XTest, W_2, norms, M, X, reg) ;


%prediction(:,run) = pred_2 ;

%% 4. Plotting

a = [XTest, pred_2 Truth_test] ;
b = [XTest_data, Truth] ;
Plot{run} = sortrows(a);
Plot2 = sortrows(b);

end

figure(1)
ax = axes ;

plot(Plot2(:,1),Plot2(:,2),'g','LineWidth',2);
hold on

for run = 1:20
plot(Plot{run}(:,1),Plot{run}(:,2),':k');
end


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


%title('Delta','interpreter','latex')
xlabel('Feature','interpreter','latex')
ylabel('Function','interpreter','latex')
ylim([-1.6 1.6])
xlim([min(XTest) max(XTest)])
legend('True function','Prediction realization','Location','southeast','interpreter','latex')


