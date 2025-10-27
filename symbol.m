function f = symbol(A,k)
    as = zeros(k*2+1,1);
    
    for i = -k:k
        as(i+k+1) = mean(diag(A,i));  
    end
        
    f = @(z) z.^[-k:k]*as;
end