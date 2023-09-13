function Z = khatri_rao(varargin)
    % KHATRI_RAO takes a list of matrices and returns the (right)
    % Khatri-Rao product.
    % INPUT list of variable number of matrices varargin.
    % OUTPUT outer product outer.

    
% Extract column size of first input matrix
K = size(varargin{1},2) ;
% Extract row size of last input matrix
J = size(varargin{end},1);
% Reshape last input matrix into [J 1 K] tensor
Z = reshape(varargin{end},[J 1 K]);

% Performing khatri-rao product
for n = length(varargin)-1:-1:1
    % Extracting row size of n-th input matrix
    I = size(varargin{n},1);
    % Reshape n-th input matrix into [1 I K] tensor
    A = reshape(varargin{n},[1 I K]);
    % Taking product
    product = A.*Z ;
    % Reshape into [I*J 1 K] tensor
    Z = reshape(product,[I*J 1 K]);
    J = I*J;
end
% Reshaping into desired matrix
Z = reshape(Z,[size(Z,1) K]);
    
end












%     Z = [] ;
%     for m = 1:size(varargin{1},2)
%     Z = [Z, kron(varargin{1}(:,m),varargin{2}(:,m))] ;
%     end
%     
%     
%     for k = 3:length(varargin)
%     for i = 1:size(varargin{k},2)
%     Z = kron(Z(:,i), varargin{k}(:,i)) ;
%     end
%     end
