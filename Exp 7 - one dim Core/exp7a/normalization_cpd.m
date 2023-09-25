function [W_new, norms_new] = normalization_cpd(W_mean, norms)

D = size(W_mean,2) ;
R = size(W_mean{1},2) ;
norms_new = norms ;

for m = 1:D
    % m0{m} = reshape(m0{m},2,20) ;
    for r = 1:R
        
        W_r = W_mean{m}(:,r) ;
        normm = norm(W_r) ;

     
        % m0_r = m0{m}(:,r) ;
        % m0{m}(:,r) = m0_r / normm ;

        W_new{m}(:,r) = W_r / normm ;
        norms_new(r) = norms_new(r) * normm ;

    end
    %m0{m} = reshape(m0{m},40,1) ;
end


end









% function [W_new, norms_new] = normalization_cpd(W,norms)
% 
% D = size(W,2) ;
% R = size(W{1},2) ;
% 
% for m = 1:D
%     for r = 1:R
% 
%         W_r = W{m}(:,r) ;
%         normm = norm(W_r) ;
% 
%         W_new{m}(:,r) = W_r / normm ;
%         norms_new(r) = norms(r) * normm ;
% 
%     end
% end
% 
% end