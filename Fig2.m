clc
clear
close all

N_total = 256;   
L = 2;
zeta = 0.95;  
SampNum = 500;  
C = 2;
sigma2 = C/2;
h = sqrt(sigma2).*(randn(L,SampNum) + 1i*randn(L,SampNum));  
eta = abs(h).^2;  
obj0 = @(k, n, SNR, p_b) sum( log(zeta * (1 - p_b) .* k .* (1 - (2.^k - 1) ./ 2.^(k - 1) .* qfunc(sqrt(3 * (SNR + 1 - 1 ./ n) .^ n ./ (4.^k - 1)))) ...
        + p_b .* k .* qfunc((2 * k - n .* log2(1 + SNR)) ./ (log2(exp(1)) * sqrt(2 * n .* SNR .* (SNR + 2) ./ (SNR + 1).^2))) ) );    
p1 = 0:0.04:1;  
p2 = 0:0.04:1;

for m = 1:size(eta,2) 
    for i = 1:length(p1) 
        for j = 1:length(p2) 
            p = [p1(i);p2(j)];
            [k_temp,N_temp,r_temp,k_fec_temp,N_fec_temp,r_fec_temp] = fun_opt_dk2(L,p,eta(:,m),N_total,zeta);
            [k_int_temp,N_int_temp,r_final_temp] = finalRounding(k_temp,N_temp,L,p,eta(:,m),N_total,zeta); 
            r_final(m,j,i) = obj0(k_int_temp',N_int_temp',p,eta(:,m));
        end
    end
end
r100 = mean(r_final,1);

negative_cols = any(r100 < 0 | isnan(r100), 1); 
filtered_r = r100(:, ~negative_cols); 
mean_r = mean(filtered_r, 2); 

r_fec = mean_r(end,end);
[P1, P2] = meshgrid(0:0.1:1,0:0.1:1); 
figure
surf(P1,P2,mean_r,'FaceColor', 'y', 'FaceAlpha',0.5); hold on
surf(P1,P2,r_fec.*ones(size(P1)),'FaceColor', 'c', 'FaceAlpha',0.5);
xlabel('p_1','FontSize',46,'FontName','Times New Roman');
ylabel('p_2','FontSize',46,'FontName','Times New Roman');
zlabel('Sum throughput','FontSize',46,'FontName','Times New Roman');
legend('O-RF-CBF','O-RF','FontSize',30,'FontName','Times New Roman','Location','southeast');
set(gca,'FontName','Times New Roman','FontSize',28)
set(gca,'Linewidth',1)
grid on

plot3(1, 1, r_final(end,end), 'go', 'MarkerSize', 20, 'MarkerFaceColor', 'g','HandleVisibility', 'off')
