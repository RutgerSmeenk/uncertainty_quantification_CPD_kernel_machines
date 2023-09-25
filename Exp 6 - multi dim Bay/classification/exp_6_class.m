load("X.mat","XTest.mat","Y.mat","YTest.mat","X_cross.mat","Y_cross.mat","run_num.mat")


rng(run_num);
M = 6;
R = 10;
maxIte = 10;


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
tic ;
[W_mean, W_cov, norms] = CP_ALS_Bay_weights(X, Y, M, R, maxIte, sigma_Bay, sigmae_Bay) ;

% Prediction
[prediction_Bay, var_CI, var_PI] = CP_ALS_Bay_predict(XTest, W_mean, W_cov, norms, M) ;

time_Bay_class = toc ;

prediction_Bay = real(prediction_Bay) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

pred_Bay = sign(prediction_Bay);

% Uncertainty intervals
CI_low_Bay = prediction_Bay - (2*sqrt(var_PI)) ;
CI_up_Bay = prediction_Bay + (2*sqrt(var_PI)) ;

CI_lower_Bay = sign(CI_low_Bay) ;
CI_upper_Bay = sign(CI_up_Bay) ;


% variableNames = {'CI_lower_Bay', 'CI_upper_Bay','CI_low_Bay', 'CI_up_Bay','pred_Bay','prediction_Bay','time_Bay_class', 'hyper_Bay_class'};
% 
% for i = 1:length(variableNames)
%     filePath1 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 3 - multi dim Freq\Exp 5.2 classification', [variableNames{i}, '.mat']);
%     save(filePath1, variableNames{i});
% end


%% Plotting


a = [prediction_Bay, CI_low_Bay, CI_up_Bay] ;
index = 1:length(prediction_Bay) ;

Plot = sortrows(a);

true_model = [Plot(:,1),Plot(:,3)] ;
pred_train = Plot(:,2) ;

loc_lower = knnsearch(Plot(:,3),0) ;
loc_upper = knnsearch(Plot(:,2),0) ;

circle = Plot(:,1) ;
low_limit = Plot(:,2) ;
up_limit = Plot(:,3) ;

figure(2)
%plot(x,noisy_data,'o',x,pred,x,truth,x,CI_min,'--',x,CI_max,'--')
% plot(x,pred,x,truth,x,CI_min,'--',x,CI_max,'--')

%l = plot(true_model(:,1),true_model(:,2),'o',Plot(:,1),Plot(:,2),Plot(:,1),Plot(:,4),'--',Plot(:,1),Plot(:,5),'--');

% plot([index(loc_lower) index(loc_upper)],[circle(loc_lower) circle(loc_upper)],'o','MarkerFaceColor', 'k')
% hold on
% l1 = plot(index(1:loc_lower),circle(1:loc_lower),'g',index(1:loc_lower),low_limit(1:loc_lower),'ko',index(1:loc_lower),up_limit(1:loc_lower),'ko');
% hold on
% l2 = plot(index(loc_lower:loc_upper),circle(loc_lower:loc_upper),'r',index(loc_lower:loc_upper),low_limit(loc_lower:loc_upper),'ko',index(loc_lower:loc_upper),up_limit(loc_lower:loc_upper),'ko');
% hold on
% l3 = plot(index(loc_upper:end),circle(loc_upper:end),'g',index(loc_upper:end),low_limit(loc_upper:end),'ko',index(loc_upper:end),up_limit(loc_upper:end),'ko');
% hold off

plot([index(loc_lower) index(loc_upper)],[circle(loc_lower) circle(loc_upper)],'o','MarkerFaceColor', 'k', 'MarkerSize', 10)
hold on
plot(index(1:loc_lower),circle(1:loc_lower),'g', 'LineWidth', 2);
plot(index(1:loc_lower),low_limit(1:loc_lower),'ko', 'MarkerSize', 1.8);
plot(index(1:loc_lower),up_limit(1:loc_lower),'ko', 'MarkerSize', 1.8);
plot(index(loc_lower:loc_upper),circle(loc_lower:loc_upper),'r', 'LineWidth', 2);
plot(index(loc_lower:loc_upper),low_limit(loc_lower:loc_upper),'ko', 'MarkerSize', 1.8);
plot(index(loc_lower:loc_upper),up_limit(loc_lower:loc_upper),'ko', 'MarkerSize', 1.8);
plot(index(loc_upper:end),circle(loc_upper:end),'g', 'LineWidth', 2);
plot(index(loc_upper:end),low_limit(loc_upper:end),'ko', 'MarkerSize', 1.8);
plot(index(loc_upper:end),up_limit(loc_upper:end),'ko', 'MarkerSize', 1.8);


yline(0,'--b');
% ylim([-3 3])
% xlim([0 130])
%legend('YTest','CPPredict','Truth','CI_{min}','CI_{max}')
% legend('CPPredict','Truth','CI_{min}','CI_{max}')
%legend('','Prediction','Confidence interval','Location','southeast','interpreter','latex')title('')
title('')
xlabel('Index','interpreter','latex','FontSize', 12)
ylabel('Sorted predictions','interpreter','latex','FontSize', 12)
legend('off');


