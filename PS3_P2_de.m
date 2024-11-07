clear;
close all;

%===============================================================================================
%------------------------------PARAMETRIZATION AND INITIALIZATION-------------------------------
%===============================================================================================

p.theta = 0.4;
p.beta = 0.99;
p.sigma = 2;
p.delta = 0.1;
[k_ss,c_ss] = steady(p);

epsilon = 10^(-6);
N = 50;
policy_init = 10;

k = linspace(3/4*k_ss,5/4*k_ss,N);
v_new = 30*log(k);

%===============================================================================================
%---------------------------------------COMPUTATION---------------------------------------------
%===============================================================================================

vv = -inf*ones(N,N);
k_prim = zeros(1,N);
c = zeros(1,N);
c_imp = zeros(1,N);
c_new = zeros(1,N);
b = k.^p.theta+(1-p.delta)*k;
for s = 1:2000
    v = v_new;
    for i = 1:N
        if b(i) < k(N)
            i_max = find((k-b(i))>0,1)-1;
        else
            i_max = N;
        end
        vv(1:i_max,i) = F(b(i)-k(1:i_max),p)+p.beta*v(1:i_max);
        [v_new(i),j] = max(vv(1:i_max,i));
        k_prim(i) = k(j);
        c(i) = b(i) - k(j);
    end
    diff = abs(v_new-v);
    if max(diff) < epsilon
        break
    end

    if s > policy_init
        for ss = 1:2000
            for i = 1:N
                [~,j] = min(abs(k-(b(i)-c(i))));
                v_new(i) = F(c(i),p)+p.beta*v(j);
            end
            diff2 = abs(v_new-v);
            v = v_new;
            if max(diff2) < epsilon
                break
            end
        end
    end
end

c_imp = c_implied(k_prim,c,p);
Euler_error = (c_imp-c)./c;
%===============================================================================================
%----------------------------------------FUNCTIONS----------------------------------------------
%===============================================================================================

function [k_ss,c_ss] = steady(p)
    k_ss = (p.theta/(1/p.beta-1+p.delta))^(1/(1-p.theta));
    c_ss = k_ss^p.theta - p.delta*k_ss;
end

function u = F(c,p)
    u = (c.^(1-p.sigma)-1)/(1-p.sigma);
end

function c_imp = c_implied(k_prim,c,p)
    c_imp = (p.beta*(p.theta*k_prim.^(p.theta-1)+1-p.delta).*c.^(-p.sigma)).^(-1/p.sigma);
end