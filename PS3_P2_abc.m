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
N = 200;
k = linspace(3/4*k_ss,5/4*k_ss,N);
v_new = log(k);

%===============================================================================================
%---------------------------------------COMPUTATION---------------------------------------------
%===============================================================================================

vv = -inf*ones(N,N);
k_prim = zeros(1,N);
for s = 1:2000
    v = v_new;
    for i = 1:N
        b = k(i)^p.theta+(1-p.delta)*k(i);
        if b < k(N)
            i_max = find((k-b)>0,1)-1;
        else
            i_max = N;
        end
        vv(1:i_max,i) = F(b-k(1:i_max),p)+p.beta*v(1:i_max);
        [v_new(i),j] = max(vv(1:i_max,i));
        k_prim(i) = k(j);
    end
    diff = abs(v_new-v);
    if max(diff) < epsilon
        break
    end
end


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