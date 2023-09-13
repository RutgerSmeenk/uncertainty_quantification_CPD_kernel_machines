%% Ripley & Four class Dataset Plots

rng(1);
M = 10;
R = 20;
maxIte = 10;
%lambda = 1e-7;
warning('off','all');

%% Processing data
close all
%rng('default');
X = readmatrix('banana.csv');
N = size(X,1);
X = X(randperm(size(X,1)),:);
Y = (X(:,end)==1)-(X(:,end)==2);
X = X(:,1:2);
XMin = min(X);  XMax = max(X);
X = (X-XMin)./(XMax-XMin);

NPlot = 100;
X1Plott = linspace(0,1,NPlot);
[X1Plot,X2Plot] = meshgrid(X1Plott,X1Plott);
XPlot = [X1Plot(:),X2Plot(:)];

XTest = readmatrix('banana.csv');
NTest = size(XTest,1);
XTest = XTest(randperm(size(XTest,1)),:);
YTest = (XTest(:,end)==1)-(XTest(:,end)==2);
XTest = XTest(:,1:2);
XTestMin = min(XTest);  XTestMax = max(XTest);
XTest = (XTest-XTestMin)./(XTestMax-XTestMin);

variableNames = {'X', 'XTest', 'Y', 'YTest','XPlot'};

for i = 1:length(variableNames)
    filePath1 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 5 - two dim Bay', [variableNames{i}, '.mat']);
    save(filePath1, variableNames{i});
    filePath2 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 8 - two dim Core', [variableNames{i}, '.mat']);
    save(filePath2, variableNames{i});
end

%% K fold cross validation
[mean_error_all, conf_error_all, lambda_folds] = cross_dat_class(X, Y, M, R, maxIte) ;

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

%% Making predictions


%[W, norms] = CP_ALS_weights(X, Y, M, R,lambda, maxIte) ;
%YPlot = ones(size(XPlot,1),1) ;
[W, norms, reg] = CP_ALS_weights(X, Y, M, R,lambda, maxIte) ;
[prediction, pred_train, C_test, C_train, regu] = CP_ALS_predict(XPlot, W, norms, M, X, reg) ;


% Estimating noise variance
sigmae = var(Y - pred_train, 1) ;

[var_CI, var_PI] = delta_method(C_train, C_test, lambda, sigmae, regu) ;

prediction = real(prediction) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

pred = sign(prediction);
   
% Confidence bound

CI_lower = real(sign(prediction - 2*sqrt(var_PI))) ;
CI_upper = real(sign(prediction + 2*sqrt(var_PI))) ;


%% Plotting

%    figure(plotIdx);
%    fig = gcf;

    figure(1)
    hold on
    s1 = scatter(X(Y==1,1),X(Y==1,2),36,[238, 28, 37]/255,'filled');
    s2 = scatter(X(Y==-1,1),X(Y==-1,2),36,[1, 90, 162]/255,'filled');    
    s1.MarkerFaceAlpha = 0.25;
    s2.MarkerFaceAlpha = 0.25;

    %c1 = contourf(X1Plot,X2Plot,reshape(pred,size(X1Plot)),'black','LineWidth',1.5);
    %colormap winter
    
    %c1 = contour(X1Plot,X2Plot,reshape(scorePlotCP,size(XPlot)),[0 0],'black','LineWidth',1.5);
    %c2 = contour(X1Plot,X2Plot,reshape(scorePlotHilbert,size(X1Plot)),[0 0],'black','LineWidth',1.5,'LineStyle','--');
    %plot(c1(1,2:end)',c1(2,2:end)')
    d1 = contour(X1Plot,X2Plot,reshape(pred,size(X1Plot)),[0 0],'black','LineWidth',1.5);
    d2 = contour(X1Plot,X2Plot,reshape(CI_lower,size(X1Plot)),[0 0],'black','LineWidth',1.5,'LineStyle','--');
    d3 = contour(X1Plot,X2Plot,reshape(CI_upper,size(X1Plot)),[0 0],'black','LineWidth',1.5,'LineStyle','--');
    
    % s1 = scatter(XTest(YTest==1,1),XTest(YTest==1,2), 40, 'k', ".", 'LineWidth', 1.5, 'MarkerFaceColor', 'none');
    % s2 = scatter(XTest(YTest==-1,1),XTest(YTest==-1,2), 10, 'k', "square", 'LineWidth', 1.5, 'MarkerFaceColor', 'none');
 

    xticks(-1:0.2:1);
    yticks(-1:0.2:1);
    %legend('Class A','Class B','Classifier','Confidence interval','Location','southeast','interpreter','latex')
    axis equal
    axis off
    hold off
    %filename = 'banana'+string(M^2)+'frequencies'+string(R)+'rank'+'.pdf';




%% Plotting ALL

%load("CI_lower_Bay.mat","CI_upper_Bay.mat","CI_lower_Core.mat","CI_upper_Core.mat","pred_Bay.mat","pred_Core.mat")

run("C:\Users\rutge\Documents\MATLAB\Thesis\Exp 5 - two dim Bay\exp_5a.m")
run("C:\Users\rutge\Documents\MATLAB\Thesis\Exp 8 - two dim Core\exp_8a.m")
    
    figure(4)

%    figure(plotIdx);
%    fig = gcf;
    hold on
    s1 = scatter(X(Y==1,1),X(Y==1,2),36,[238, 28, 37]/255,'filled');
    s2 = scatter(X(Y==-1,1),X(Y==-1,2),36,[1, 90, 162]/255,'filled');    
    s1.MarkerFaceAlpha = 0.25;
    s2.MarkerFaceAlpha = 0.25;

    %c1 = contourf(X1Plot,X2Plot,reshape(pred,size(X1Plot)),'black','LineWidth',1.5);
    %colormap winter
    
    %c1 = contour(X1Plot,X2Plot,reshape(scorePlotCP,size(XPlot)),[0 0],'black','LineWidth',1.5);
    %c2 = contour(X1Plot,X2Plot,reshape(scorePlotHilbert,size(X1Plot)),[0 0],'black','LineWidth',1.5,'LineStyle','--');
    %plot(c1(1,2:end)',c1(2,2:end)')
    a1 = contour(X1Plot,X2Plot,reshape(pred,size(X1Plot)),[0 0],'blue','LineWidth',1.5);
    a2 = contour(X1Plot,X2Plot,reshape(CI_lower,size(X1Plot)),[0 0],'blue','LineWidth',1.5,'LineStyle','--');
    a3 = contour(X1Plot,X2Plot,reshape(CI_upper,size(X1Plot)),[0 0],'blue','LineWidth',1.5,'LineStyle','--');

    b1 = contour(X1Plot,X2Plot,reshape(pred_Bay,size(X1Plot)),[0 0],'green','LineWidth',1.5);
    b2 = contour(X1Plot,X2Plot,reshape(CI_lower_Bay,size(X1Plot)),[0 0],'green','LineWidth',1.5,'LineStyle','--');
    b3 = contour(X1Plot,X2Plot,reshape(CI_upper_Bay,size(X1Plot)),[0 0],'green','LineWidth',1.5,'LineStyle','--');

    c1 = contour(X1Plot,X2Plot,reshape(pred_Core,size(X1Plot)),[0 0],'red','LineWidth',1.5);
    c2 = contour(X1Plot,X2Plot,reshape(CI_lower_Core,size(X1Plot)),[0 0],'red','LineWidth',1.5,'LineStyle','--');
    c3 = contour(X1Plot,X2Plot,reshape(CI_upper_Core,size(X1Plot)),[0 0],'red','LineWidth',1.5,'LineStyle','--');
    
    % s1 = scatter(XTest(YTest==1,1),XTest(YTest==1,2), 40, 'k', ".", 'LineWidth', 1.5, 'MarkerFaceColor', 'none');
    % s2 = scatter(XTest(YTest==-1,1),XTest(YTest==-1,2), 10, 'k', "square", 'LineWidth', 1.5, 'MarkerFaceColor', 'none');
    hold off

    xticks(-1:0.2:1);
    yticks(-1:0.2:1);
    %legend('Class A','Class B','Classifier','Confidence interval','Location','southeast','interpreter','latex')
    axis equal
    axis off
    hold off
    %filename = 'banana'+string(M^2)+'frequencies'+string(R)+'rank'+'.pdf';

%% Plotting

[W, norms, reg] = CP_ALS_weights(X, Y, M, R,lambda, maxIte) ;
[prediction, pred_train, C_test, C_train, regu] = CP_ALS_predict(XTest, W, norms, M, X, reg) ;


% Estimating noise variance
sigmae = var(Y - pred_train, 1) ;

[var_CI, var_PI] = delta_method(C_train, C_test, lambda, sigmae, regu) ;

prediction = real(prediction) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

CI_low = prediction - (2*sqrt(var_PI)) ;
CI_up = prediction + (2*sqrt(var_PI)) ;

CI_lower = sign(CI_low) ;
CI_upper = sign(CI_up) ;


a = [prediction, CI_low, CI_up] ;
index = 1:length(prediction) ;

Plot = sortrows(a);

true_model = [Plot(:,1),Plot(:,3)] ;
%pred_train = Plot(:,2) ;

loc_lower = knnsearch(Plot(:,3),0) ;
loc_upper = knnsearch(Plot(:,2),0) ;

circle = Plot(:,1) ;
low_limit = Plot(:,2) ;
up_limit = Plot(:,3) ;

figure(5)
%plot(x,noisy_data,'o',x,pred,x,truth,x,CI_min,'--',x,CI_max,'--')
% plot(x,pred,x,truth,x,CI_min,'--',x,CI_max,'--')

%l = plot(true_model(:,1),true_model(:,2),'o',Plot(:,1),Plot(:,2),Plot(:,1),Plot(:,4),'--',Plot(:,1),Plot(:,5),'--');

plot([index(loc_lower) index(loc_upper)],[circle(loc_lower) circle(loc_upper)],'o','MarkerFaceColor', 'k')
hold on
l1 = plot(index(1:loc_lower),circle(1:loc_lower),'g',index(1:loc_lower),low_limit(1:loc_lower),'ko',index(1:loc_lower),up_limit(1:loc_lower),'ko');
hold on
l2 = plot(index(loc_lower:loc_upper),circle(loc_lower:loc_upper),'r',index(loc_lower:loc_upper),low_limit(loc_lower:loc_upper),'ko',index(loc_lower:loc_upper),up_limit(loc_lower:loc_upper),'ko');
hold on
l3 = plot(index(loc_upper:end),circle(loc_upper:end),'g',index(loc_upper:end),low_limit(loc_upper:end),'ko',index(loc_upper:end),up_limit(loc_upper:end),'ko');
hold off

l1(2).MarkerSize = 1; 
l1(3).MarkerSize = 1; 
l2(2).MarkerSize = 1; 
l2(3).MarkerSize = 1;
l3(2).MarkerSize = 1; 
l3(3).MarkerSize = 1; 
%l(2).MarkerSize = 12;
title('Plot','interpreter','latex')
xlabel('Index','interpreter','latex')
ylabel('Sorted latent variables','interpreter','latex')
yline(0,'--b');
ylim([-2 2])
% ylim([-2 2])
% xlim([-100 100])
%legend('YTest','CPPredict','Truth','CI_{min}','CI_{max}')
% legend('CPPredict','Truth','CI_{min}','CI_{max}')
legend('','Prediction','Confidence interval','Location','southeast','interpreter','latex')
    
    
  
%% Evaluating results



[W, norms, reg] = CP_ALS_weights(X, Y, M, R,lambda, maxIte) ;
[prediction, pred_train, C_test, C_train, regu] = CP_ALS_predict(XTest, W, norms, M, X, reg) ;

[var_CI, var_PI] = delta_method(C_train, C_test, lambda, sigmae, regu) ;

prediction = real(prediction) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

CI_low = real(prediction - (2*sqrt(var_PI))) ;
CI_up = real(prediction + (2*sqrt(var_PI))) ;

CI_lower = sign(CI_low) ;
CI_upper = sign(CI_up) ;

clear c
N_Test = length(YTest) ;
for n = 1:N_Test
    if (YTest(n) == CI_lower(n)) || (YTest(n) == CI_upper(n))
    c(n) = 1 ;
    else
    c(n) = 0 ;
    end
end


PICP = 1/N_Test * sum(c) ;
mu = 1 - 0.05 ;
%mu = 0.05 ;
eta = 50 ;

if PICP >= mu 
    gamma = 0 ;
else 
    gamma = 1 ;
end
%gamma=0;


% MPIW = (1/N_Test * sum(CI_upper - CI_lower)) / abs(CI_up - CI_low) ;
MPIW = 1/N_Test * sum(CI_up - CI_low) ;

%CWC = MPIW * (1 + gamma * exp(-eta * (PICP - mu))) ;
CWC = MPIW + gamma * exp(-eta * (PICP - mu)) ;




























%% Plotting


% a = [prediction, CI_low, CI_up] ;
% index = 1:length(prediction) ;
% 
% Plot = sortrows(a);
% 
% true_model = [Plot(:,1),Plot(:,3)] ;
% pred_train = Plot(:,2) ;
% 
% loc_lower = knnsearch(Plot(:,3),0) ;
% loc_upper = knnsearch(Plot(:,2),0) ;
% 
% circle = Plot(:,1) ;
% low_limit = Plot(:,2) ;
% up_limit = Plot(:,3) ;
% 
% figure(2)
% %plot(x,noisy_data,'o',x,pred,x,truth,x,CI_min,'--',x,CI_max,'--')
% % plot(x,pred,x,truth,x,CI_min,'--',x,CI_max,'--')
% 
% %l = plot(true_model(:,1),true_model(:,2),'o',Plot(:,1),Plot(:,2),Plot(:,1),Plot(:,4),'--',Plot(:,1),Plot(:,5),'--');
% 
% plot([index(loc_lower) index(loc_upper)],[circle(loc_lower) circle(loc_upper)],'o','MarkerFaceColor', 'k')
% hold on
% l1 = plot(index(1:loc_lower),circle(1:loc_lower),'g',index(1:loc_lower),low_limit(1:loc_lower),'ko',index(1:loc_lower),up_limit(1:loc_lower),'ko');
% hold on
% l2 = plot(index(loc_lower:loc_upper),circle(loc_lower:loc_upper),'r',index(loc_lower:loc_upper),low_limit(loc_lower:loc_upper),'ko',index(loc_lower:loc_upper),up_limit(loc_lower:loc_upper),'ko');
% hold on
% l3 = plot(index(loc_upper:end),circle(loc_upper:end),'g',index(loc_upper:end),low_limit(loc_upper:end),'ko',index(loc_upper:end),up_limit(loc_upper:end),'ko');
% hold off
% 
% l1(2).MarkerSize = 1; 
% l1(3).MarkerSize = 1; 
% l2(2).MarkerSize = 1; 
% l2(3).MarkerSize = 1;
% l3(2).MarkerSize = 1; 
% l3(3).MarkerSize = 1; 
% %l(2).MarkerSize = 12;
% title('Plot','interpreter','latex')
% xlabel('Index','interpreter','latex')
% ylabel('Sorted latent variables','interpreter','latex')
% yline(0,'--b');
% ylim([-2 2])
% % ylim([-2 2])
% % xlim([-100 100])
% %legend('YTest','CPPredict','Truth','CI_{min}','CI_{max}')
% % legend('CPPredict','Truth','CI_{min}','CI_{max}')
% legend('','Prediction','Confidence interval','Location','southeast','interpreter','latex')





