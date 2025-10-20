clc
clear
close all

L = 2;  
N_total = 256;     
zeta = 0.95;

C = 2;  
SampNum = 500;  
sigma2 = C/2;
h0 = sqrt(sigma2).*(randn(1,SampNum) + 1i*randn(1,SampNum));  
h = [h0;h0];
eta = abs(h).^2;  
p1 = 0:0.04:1;  

for i = 1:length(p1)
    for j = 1:SampNum
        try
            p_b = [p1(i);0.4];
            [K, N, r(i,j), K_fec, N_fec, r_fec(i,j)] = fun_opt_dk2(L, p_b, eta(:,j), N_total, zeta);  
            k1_temp(i,j) = K(1);
            k2_temp(i,j) = K(2);
            N1_temp(i,j) = N(1);
            N2_temp(i,j) = N(2);
        catch ME
            fprintf('Error at i=%d, j=%d: %s\n', i, j, ME.message);
            continue;
        end
    end
end

k1 = mean(k1_temp,2);
k2 = mean(k2_temp,2);
N1 = mean(N1_temp,2);
N2 = mean(N2_temp,2);

figure(2)
yyaxis left
plot(p1,k1,'-ko','MarkerFaceColor', 'k','LineWidth',6,'MarkerSize',22);
hold on;
plot(p1,k2,'-mo','LineWidth',6,'MarkerSize',22);
ylabel('Allocated transmitted bit number','FontSize',64,'FontName','Times New Roman');
yyaxis right
plot(p1,N1,'-ks','MarkerFaceColor', 'k','LineWidth',6,'MarkerSize',22);
hold on;
plot(p1,N2,'-ms','LineWidth',6,'MarkerSize',22);
xlabel('p_1(p_2=0.4)','FontSize',64,'FontName','Times New Roman');
ylabel('Allocated channel symbol number','FontSize',64,'FontName','Times New Roman');
legend('K_1 of O-RF-CBF','K_2 of O-RF-CBF','N_1 of O-RF-CBF','N_2 of O-RF-CBF','FontSize',42,'FontName','Times New Roman','Location','northeast');
set(gca,'FontName','Times New Roman','FontSize',42)
set(gca,'Linewidth',1.5)
grid on