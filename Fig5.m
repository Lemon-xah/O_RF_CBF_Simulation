clc
clear
close all

L = 4;  
N_total = 256;     
p_b = 0.1*ones(L,1);  
zeta_1 = 0:0.05:1;  
zeta = 1-zeta_1;
C = 2;  
SampNum = 100;    
sigma2 = C/2;
h = sqrt(sigma2).*(randn(L,SampNum) + 1i*randn(L,SampNum));  
eta = abs(h).^2;  
obj0 = @(k, n, z, SNR) sum( log( z * (1 - p_b) .* k .* (1 - (2.^k - 1) ./ 2.^(k - 1) .* qfunc(sqrt(3 * (SNR + 1 - 1 ./ n) .^ n ./ (4.^k - 1)))) ...
        + p_b .* k .* qfunc((2 * k - n .* log2(1 + SNR)) ./ (log2(exp(1)) * sqrt(2 * n .* SNR .* (SNR + 2) ./ (SNR + 1).^2))) ) );    
fec0 = @(k, n, SNR) sum( log( k .* qfunc((2 * k - n .* log2(1 + SNR)) ./ (log2(exp(1)) * sqrt(2 * n .* SNR .* (SNR + 2) ./ (SNR + 1).^2))) ) );

for i = 1:length(zeta)
    for j = 1:SampNum
        try
        [K, N, r(i,j), K_fec, N_fec, r_fec(i,j)] = fun_opt_dk0(L, p_b, eta(:,j), N_total, zeta(i)); 
        [k_int,N_int,r_int(i,j)] = finalRounding(K,N,L,p_b,eta(:,j),N_total,zeta(i));  
        [k_fec_int,N_fec_int,r_fec_int(i,j)] = finalfecRounding(K_fec,N_fec,L,p_b, eta(:,j),N_total,zeta(i));
        r_final(i,j)= obj0(k_int',N_int',zeta(i),eta(:,j));
        r_fec_final(i,j)= fec0(k_fec_int',N_fec_int',eta(:,j));
        catch ME
            fprintf('Error at i=%d, j=%d: %s\n', i, j, ME.message);
            continue;
        end
    end
end

negative_cols = any(r_final <= 0 | isnan(r_final), 1); 
filtered_r = r_final(:, ~negative_cols);
mean_r = mean(filtered_r, 2); 
negative_cols_fec = any(r_fec_final <= 0 | isnan(r_fec_final), 1); 
filtered_r_fec = r_fec_final(:, ~negative_cols_fec); 
mean_r_fec = mean(filtered_r_fec, 2); 

figure
plot(zeta_1(1:9),mean_r(1:9),'-^','Color','r','LineWidth',6,'MarkerSize',22);
hold on;
plot(zeta_1(1:9),mean_r_fec(1:9),'-^','Color','b','LineWidth',6,'MarkerSize',22); 
xlabel('\zeta','FontSize',64,'FontName','Times New Roman');
ylabel('Uplink sum-log throughput','FontSize',64,'FontName','Times New Roman');
legend('O-RF-CBF','O-RF','FontSize',42,'FontName','Times New Roman','Location','southwest');
set(gca,'FontName','Times New Roman','FontSize',42)
set(gca,'Linewidth',1.5)

grid on
