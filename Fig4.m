clc
clear
close all

N_total = 256;    
p = 0.1;  
zeta = 0.95;  
SampNum = 500;   

obj0 = @(k, n, SNR, p_b) sum( log(zeta * (1 - p_b) .* k .* (1 - (2.^k - 1) ./ 2.^(k - 1) .* qfunc(sqrt(3 * (SNR + 1 - 1 ./ n) .^ n ./ (4.^k - 1)))) ...
        + p_b .* k .* qfunc((2 * k - n .* log2(1 + SNR)) ./ (log2(exp(1)) * sqrt(2 * n .* SNR .* (SNR + 2) ./ (SNR + 1).^2))) ) );    
fec0 = @(k, n, SNR) sum( log( k .* qfunc((2 * k - n .* log2(1 + SNR)) ./ (log2(exp(1)) * sqrt(2 * n .* SNR .* (SNR + 2) ./ (SNR + 1).^2))) ) );

L = 4;  
  
for i = 1:length(L)
    for j = 1:SampNum
        try
            p_b = p*ones(L(i),1);
            C = 2;
            sigma2 = C/2;
            h = sqrt(sigma2).*(randn(L(i),SampNum) + 1i*randn(L(i),SampNum));  
            eta = abs(h).^2; 
            [K, N, r(i,j), K_fec, N_fec, r_fec(i,j)] = fun_opt_dk2(L(i), p_b, eta(:,j), N_total, zeta);
            [k_int,N_int,r_int(i,j)] = finalRounding(K,N,L(i),p_b,eta(:,j),N_total,zeta);  
            [k_fec_int,N_fec_int,r_fec_int(i,j)] = finalfecRounding(K_fec,N_fec,L(i),p_b,eta(:,j),N_total,zeta);
            r_final(i,j)= obj0(k_int',N_int',eta(:,j),p_b);
            r_fec_final(i,j)= fec0(k_fec_int',N_fec_int',eta(:,j));
        catch ME
            fprintf('Error at i=%d, j=%d: %s\n', i, j, ME.message);
            continue;
        end
    end
end

negative_cols = any(r_final < 0 | isnan(r_final), 1); 
filtered_r = r_final(:, ~negative_cols); 
mean_r = mean(filtered_r, 2); 
negative_cols_fec = any(r_fec_final < 0 | isnan(r_fec_final), 1); 
filtered_r_fec = r_fec_final(:, ~negative_cols_fec); 
mean_r_fec = mean(filtered_r_fec, 2); 

