function [K_end, N_end, r_end, K_fec_end, N_fec_end, r_fec_end] = fun_opt_dk2(L, p_b, SNR, N_total, zeta)  
    iter = 100;   
    dlt = 1e-6;    
    K = 120 * ones(iter + 1, L);     
    N = floor(N_total/L) * ones(iter + 1, L);    
    K_fec = 120 * ones(iter + 1, L);  
    N_fec = floor(N_total/L) * ones(iter + 1, L); 
    obj0 = @(k, n) sum(zeta * (1 - p_b) .* k .* (1 - (2.^k - 1) ./ 2.^(k - 1) .* qfunc(sqrt(3 * (SNR + 1 - 1 ./ n) .^ n ./ (4.^k - 1)))) ...
        + p_b .* k .* qfunc((2 * k - n .* log2(1 + SNR)) ./ (log2(exp(1)) * sqrt(2 * n .* SNR .* (SNR + 2) ./ (SNR + 1).^2))));  
    fec0 = @(k, n) sum(k .* qfunc((2 * k - n .* log2(1 + SNR)) ./ (log2(exp(1)) * sqrt(2 * n .* SNR .* (SNR + 2) ./ (SNR + 1).^2))));
    r = zeros(iter, 1);
    r_fec = zeros(iter, 1);

    for j = 1:iter
        K(j+1, :) = P2(K(j, :), N(j, :), L, p_b, SNR, zeta, N_total, dlt);
        N(j+1, :) = P3(K(j+1, :), N(j, :), L, p_b, SNR, zeta, N_total, dlt);  
        K_fec(j+1, :) = P2_fec(K_fec(j, :), N_fec(j, :), L, SNR, N_total, dlt);
        N_fec(j+1, :) = P3_fec(K_fec(j+1, :), N_fec(j, :), L, SNR, N_total, dlt);
        
        r(j) = obj0(K(j+1, :)', N(j+1, :)');
        r_fec(j) = fec0(K_fec(j+1, :)', N_fec(j+1, :)');
    end

    K_end = K(end, :)';
    N_end = N(end, :)';
    K_fec_end = K_fec(end, :)';
    N_fec_end = N_fec(end, :)';
    r_end = r(end);
    r_fec_end = r_fec(end);
end

function K_new = P2(K_now, N_now, L, p_b, SNR, zeta, N_total, dlt)
    r_mean = @(k, n, p, eta) p * k * qfunc((k - (n / 2) * log2(1 + eta)) / (log2(exp(1)) * sqrt((n * eta * (eta + 2)) / (2 * (eta + 1)^2)))) ...
        + (1 - p) * zeta * (k * (1 - (2.^k - 1) / 2.^(k - 1) * qfunc(sqrt(3 / (4.^k - 1) * (eta + (n - 1) / n)^n))));
    
    K_temp = 1:4*N_total;
    K_new = zeros(1, L);
    
    for j = 1:L
        d1K = zeros(1, length(K_temp));
        for i = 1:length(K_temp)
            d1K(i) = (r_mean(K_temp(i) + dlt, N_now(j), p_b(j), SNR(j)) - r_mean(K_temp(i) - dlt, N_now(j), p_b(j), SNR(j))) / (2 * dlt); 
        end
        interp_func = @(x) interp1(K_temp, d1K, x, 'pchip');
        sign_changes = find(diff(sign(d1K)) ~= 0);
        first_change = sign_changes(1);
        K_new(j) = fzero(interp_func, [K_temp(first_change), K_temp(first_change + 1)]);
    end  
end

function N_new = P3(K_now, N_now, L, p_b, SNR, zeta, N_total, dlt)
    obj_approx1 = @(k, n, p, eta) zeta * (1 - p) .* k .* (1 - (1 - 0.5.^k) .* exp(-(3 * (eta + 1 - 1 ./ n).^n / 2.^(2 * k + 1)))) ...
        + p .* k .* 0.5 .* exp(-(2 * k - n .* log2(1 + eta)).^2 .* (eta + 1).^2 ./ (4 * (log2(exp(1)))^2 * n .* eta .* (eta + 2)));

    obj_approx2 = @(k, n, p, eta) zeta * (1 - p) .* k .* (1 - (1 - 0.5.^k) .* exp(-(3 * (eta + 1 - 1 ./ n).^n / 2.^(2 * k + 1)))) ...
        + p .* k .* (1 - 0.5 .* exp(-(2 * k - n .* log2(1 + eta)).^2 .* (eta + 1).^2 ./ (4 * (log2(exp(1)))^2 * n .* eta .* (eta + 2))));

    d1u = (obj_approx1(K_now', N_now' + dlt, p_b, SNR) - obj_approx1(K_now', N_now' - dlt, p_b, SNR)) / (2 * dlt);  
    d2u = (obj_approx1(K_now', N_now' + dlt, p_b, SNR) + obj_approx1(K_now', N_now' - dlt, p_b, SNR) - 2 * obj_approx1(K_now', N_now', p_b, SNR)) / dlt^2;
    d1v = (obj_approx2(K_now', N_now' + dlt, p_b, SNR) - obj_approx2(K_now', N_now' - dlt, p_b, SNR)) / (2 * dlt);  
    d2v = (obj_approx2(K_now', N_now' + dlt, p_b, SNR) + obj_approx2(K_now', N_now' - dlt, p_b, SNR) - 2 * obj_approx2(K_now', N_now', p_b, SNR)) / dlt^2;
    th = 2 * K_now ./ log2(1 + SNR)';
    use_first_approx = (N_now <= th);
    a = zeros(L, 1);
    b = zeros(L, 1);
    c = zeros(L, 1);
    for i = 1:L
        if use_first_approx(i)
            a(i) = min(d2u(i), 0);
            b(i) = d1u(i);
            c(i) = obj_approx1(K_now(i), N_now(i), p_b(i), SNR(i));
        else
            a(i) = min(d2v(i), 0);
            b(i) = d1v(i);
            c(i) = obj_approx2(K_now(i), N_now(i), p_b(i), SNR(i));
        end
    end

    lambda = 0; 
    N_old = N_now';
    max_iter_gd = 20;
    alpha_n = 0.2;
    alpha_lambda = 0.01;

    for iter_gd = 1:max_iter_gd
        g_N = zeros(L, 1);
        for i = 1:L
            delta_Ni = N_old(i) - N_now(i);
            denom = a(i)*delta_Ni^2 + b(i)*delta_Ni + c(i);
            if denom > 0
                g_N(i) = (2*a(i)*delta_Ni + b(i)) / denom;
            else
                g_N(i) = 0; 
            end
        end
        g_constraint = sum(N_old) - N_total;
        N_new_val = N_old + alpha_n * (g_N - lambda);
        lambda_new = lambda + alpha_lambda * g_constraint;
        N_new_val = max(N_new_val, 1);
        delta = (sum(N_new_val) - N_total) / L;
        N_new_val = N_new_val - delta;
        N_new_val = max(N_new_val, 1); 
        while abs(sum(N_new_val) - N_total) > 1e-6
            adjust = (N_total - sum(N_new_val)) / sum(N_new_val > 1);
            N_new_val(N_new_val > 1) = N_new_val(N_new_val > 1) + adjust;
            N_new_val = max(N_new_val, 1);
        end
        N_old = N_new_val;
        lambda = lambda_new;
    end
    N_new = N_old';
end

function K_fec_new = P2_fec(K_fec_now, N_fec_now, L, SNR, N_total, dlt)
    r_fec_mean = @(k, n, eta) k * qfunc((k - (n / 2) * log2(1 + eta)) / (log2(exp(1)) * sqrt((n * eta * (eta + 2)) / (2 * (eta + 1)^2))));
    
    K_fec_temp = 1:4*N_total;
    K_fec_new = zeros(1, L);
    
    for j = 1:L
        d1K = zeros(1, length(K_fec_temp));
        for i = 1:length(K_fec_temp)
            d1K(i) = (r_fec_mean(K_fec_temp(i) + dlt, N_fec_now(j), SNR(j)) - r_fec_mean(K_fec_temp(i) - dlt, N_fec_now(j), SNR(j))) / (2 * dlt); 
        end
        interp_func = @(x) interp1(K_fec_temp, d1K, x, 'pchip');
        sign_changes = find(diff(sign(d1K)) ~= 0);
        first_change = sign_changes(1);
        K_fec_new(j) = fzero(interp_func, [K_fec_temp(first_change), K_fec_temp(first_change + 1)]);
    end 
end

function N_fec_new = P3_fec(K_fec_now, N_fec_now, L, SNR, N_total, dlt)
    fec_approx1 =  @(k,n,eta) k.*0.5.*exp( -(2*k-n.*log2(1+eta)).^2.*(eta+1).^2./(4*(log2(exp(1)))^2*n.*eta.*(eta+2)) ) ;
    fec_approx2 =  @(k,n,eta) k.*(1-0.5.*exp( -(2*k-n.*log2(1+eta)).^2.*(eta+1).^2./(4*(log2(exp(1)))^2*n.*eta.*(eta+2)) ));
    d1u = (fec_approx1(K_fec_now,N_fec_now+dlt,SNR)-fec_approx1(K_fec_now,N_fec_now-dlt,SNR))/(2*dlt);  
    d2u = (fec_approx1(K_fec_now,N_fec_now+dlt,SNR)+fec_approx1(K_fec_now,N_fec_now-dlt,SNR)-2*fec_approx1(K_fec_now,N_fec_now,SNR))/dlt^2;
    d1v = (fec_approx2(K_fec_now,N_fec_now+dlt,SNR)-fec_approx2(K_fec_now,N_fec_now-dlt,SNR))/(2*dlt);  
    d2v = (fec_approx2(K_fec_now,N_fec_now+dlt,SNR)+fec_approx2(K_fec_now,N_fec_now-dlt,SNR)-2*fec_approx2(K_fec_now,N_fec_now,SNR))/dlt^2;
    th = 2 * K_fec_now ./ log2(1 + SNR)';
    use_first_approx = (N_fec_now <= th);
    a = zeros(L, 1);
    b = zeros(L, 1);
    c = zeros(L, 1);
    for i = 1:L
        if use_first_approx(i)
            a(i) = min(d2u(i), 0);
            b(i) = d1u(i);
            c(i) = fec_approx1(K_fec_now(i), N_fec_now(i), SNR(i));
        else
            a(i) = min(d2v(i), 0);
            b(i) = d1v(i);
            c(i) = fec_approx2(K_fec_now(i), N_fec_now(i), SNR(i));
        end
    end
    lambda = 0; 
    N_fec_old = N_fec_now';
    max_iter_gd = 20;
    alpha_n = 0.2; 
    alpha_lambda = 0.01; 

    for iter_gd = 1:max_iter_gd
        g_N = zeros(L, 1);
        for i = 1:L
            delta_Ni = N_fec_old(i) - N_fec_now(i);
            denom = a(i)*delta_Ni^2 + b(i)*delta_Ni + c(i);
            if denom > 0
                g_N(i) = (2*a(i)*delta_Ni + b(i)) / denom;
            else
                g_N(i) = 0; 
            end
        end
        g_constraint = sum(N_fec_old) - N_total;
        N_new_val = N_fec_old + alpha_n * (g_N - lambda);
        lambda_new = lambda + alpha_lambda * g_constraint;
        N_new_val = max(N_new_val, 1);
        delta = (sum(N_new_val) - N_total) / L;
        N_new_val = N_new_val - delta;
        N_new_val = max(N_new_val, 1); 
        while abs(sum(N_new_val) - N_total) > 1e-6
            adjust = (N_total - sum(N_new_val)) / sum(N_new_val > 1);
            N_new_val(N_new_val > 1) = N_new_val(N_new_val > 1) + adjust;
            N_new_val = max(N_new_val, 1);
        end
        N_fec_old = N_new_val;
        lambda = lambda_new;
    end
    N_fec_new = N_fec_old';
end