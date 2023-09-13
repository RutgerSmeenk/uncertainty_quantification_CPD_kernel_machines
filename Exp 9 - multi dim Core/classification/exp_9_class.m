load("X.mat","XTest.mat","Y.mat","YTest.mat","X_cross.mat","Y_cross.mat","run_num.mat")


rng(run_num);
M = 8;
R = 10;
maxIte = 10;
sto_core = 2;


%% K fold cross validation
[mean_error_all_Core, conf_error_all_Core, sigma_folds_Core, lambda_folds_Core] = cross_Core_dat(X, Y, M, R, maxIte, sto_core) ;

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

hyper_Core_class = [lambda_Core, sigma_Core] ;

%% Making predictions

% Obtaining the weights
%YPlot = ones(size(XPlot,1),1) ;

tic ;
[W, norms] = CP_ALS_weights(X, Y, M, R, lambda_Core, maxIte);
[prediction_Core, var_CI, var_PI, sigmae_Core_class] = CP_ALS_Core_predict(XTest, W, norms, M, sigma_Core, sto_core, Y, X);

time_Core_class = toc ;

prediction_Core = real(prediction_Core) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

pred_Core = sign(prediction_Core);

CI_low_Core = real(prediction_Core - (2*sqrt(var_PI))) ;
CI_up_Core = real(prediction_Core + (2*sqrt(var_PI))) ;

CI_lower_Core = sign(CI_low_Core) ;
CI_upper_Core = sign(CI_up_Core) ;

% %% Plotting
% 
% a = [prediction_Core, CI_low_Core, CI_up_Core] ;
% index = 1:length(prediction_Core) ;
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
% p1 = plot(index(1:loc_lower),circle(1:loc_lower),'g',index(1:loc_lower),low_limit(1:loc_lower),'ko',index(1:loc_lower),up_limit(1:loc_lower),'ko');
% hold on
% p2 = plot(index(loc_lower:loc_upper),circle(loc_lower:loc_upper),'r',index(loc_lower:loc_upper),low_limit(loc_lower:loc_upper),'ko',index(loc_lower:loc_upper),up_limit(loc_lower:loc_upper),'ko');
% hold on
% p3 = plot(index(loc_upper:end),circle(loc_upper:end),'g',index(loc_upper:end),low_limit(loc_upper:end),'ko',index(loc_upper:end),up_limit(loc_upper:end),'ko');
% hold off
% 
% p1(2).MarkerSize = 1; 
% p1(3).MarkerSize = 1; 
% p2(2).MarkerSize = 1; 
% p2(3).MarkerSize = 1;
% p3(2).MarkerSize = 1; 
% p3(3).MarkerSize = 1; 
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

%% Evaluating results

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
% MPIW = 1/N_Test * sum(CI_up_Core - CI_low_Core) ;
% % CWC = MPIW * (1 + gamma * exp(-eta * (PICP - mu))) ;
% CWC = MPIW + gamma * exp(-eta * (PICP - mu)) ;

