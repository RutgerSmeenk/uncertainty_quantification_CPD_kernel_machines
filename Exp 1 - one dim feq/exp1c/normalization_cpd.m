function [W_new, norms_new] = normalization_cpd(W,norms)

D = size(W,2) ;
R = size(W{1},2) ;
norms_new = norms ;

for d = 1:D
    for r = 1:R
        
        W_r = W{d}(:,r) ;
        normm = norm(W_r) ;

        W_new{d}(:,r) = W_r / normm ;
        norms_new(r) = norms_new(r) * normm ;

    end
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