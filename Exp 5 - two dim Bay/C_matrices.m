function [CC,Cy,C] = C_matrices(A,B,y)
    [N,DA] = size(A);
    [~,DB] = size(B);
    
    C = repmat(A,1,DB).*kron(B, ones(1, DA));
    CC = C'*C;
    Cy = C'*y;
end