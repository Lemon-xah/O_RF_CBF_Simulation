function [k_int,N_int,r_final] = finalRounding(k,N,L,p,eta,N_total,zeta)
x = [k;N];
floor_x = floor(x)';
choices = dec2bin(0:2^(2*L)-1, 2*L) - '0';
integer_vectors = repmat(floor_x, 2^(2*L), 1) + choices;
valid_indices = sum(integer_vectors(:, L+1:2*L), 2) == N_total;
valid_integer_vectors = integer_vectors(valid_indices, :);
obj = @(x,y) sum(zeta*(1-p).*x.*( 1-2*(1-2.^(-x)).*qfunc(sqrt(3*(eta+1-1./y).^y)./2.^(x)) ) + p.*x.*qfunc( (2*x-y.*log2(1+eta))./(log2(exp(1))*sqrt(2*y.*eta.*(eta+2)./(eta+1).^2))));  
k_temp = zeros(2^L,L);
N_temp = zeros(2^L,L);
result = zeros(2^L,1);
for i = 1:nnz(valid_indices)
    kN = valid_integer_vectors(i,:);
    k_temp(i,:) = kN(1:L);
    N_temp(i,:) = kN(L+1:2*L);
    result(i) = obj(k_temp(i,:)',N_temp(i,:)');
end
[max_val, max_idx] = max(result);
k_int = k_temp(max_idx,:);
N_int = N_temp(max_idx,:);
r_final = max_val;
end

