function vec_X = vec(X)
    % VEC takes as input a tensor of any order and returns its
    % vectorization, i.e. a column vector.
    % INPUT tensor X.
    % OUTPUT vector vec_X.
    
    vec_X = reshape(X,[numel(X),1]) ;
    
   
    
end