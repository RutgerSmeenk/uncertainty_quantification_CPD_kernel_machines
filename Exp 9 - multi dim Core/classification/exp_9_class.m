load("X.mat","XTest.mat","Y.mat","YTest.mat","X_cross.mat","Y_cross.mat","run_num.mat","lambda_delta.mat")


rng(run_num);
M = 12;
R = 10;
maxIte = 10;
sto_core = 7;


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
%YPlot = ones(size(XPlot,1),1) ;
rng(run_num)
tic ;
[W, norms] = CP_ALS_weights(X, Y, M, R, lambda_delta, maxIte);
W3=W;
[prediction_Core, var_CI, var_PI] = CP_ALS_Core_predict(XTest, W, norms, M, sigma_Core, sto_core, Y, X, sigmae_Core);
pred3=  prediction_Core ;

time_Core_class = toc ;

prediction_Core = real(prediction_Core) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

pred_Core = sign(prediction_Core);

CI_low_Core = real(prediction_Core - (2*sqrt(var_PI))) ;
CI_up_Core = real(prediction_Core + (2*sqrt(var_PI))) ;

CI_lower_Core = sign(CI_low_Core) ;
CI_upper_Core = sign(CI_up_Core) ;


%% Plotting

a = [prediction_Core, CI_low_Core, CI_up_Core] ;
index = 1:length(prediction_Core) ;

Plot = sortrows(a);

true_model = [Plot(:,1),Plot(:,3)] ;
pred_train = Plot(:,2) ;

loc_lower = knnsearch(Plot(:,3),0) ;
loc_upper = knnsearch(Plot(:,2),0) ;

circle = Plot(:,1) ;
low_limit = Plot(:,2) ;
up_limit = Plot(:,3) ;

figure(3)

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
xlabel('Index','interpreter','latex','FontSize', 12)
ylabel('Sorted predictions','interpreter','latex','FontSize', 12)
legend('off');


