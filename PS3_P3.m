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
N = 100;
k = linspace(3/4*k_ss,5/4*k_ss,N);
v_new = 30*log(k);
%%
%===============================================================================================
%---------------------------------------COMPUTATION---------------------------------------------
%===============================================================================================

k_prim = zeros(1,N);
b = k.^p.theta+(1-p.delta)*k;
c = zeros(1,N);
c_imp = zeros(1,N);
for s = 1:2000
    v = v_new;
    for i = 1:N
        if b(i) < k(N)
            i_max = find((k-b(i))>0,1)-1;
        else
            i_max = N;
        end
        if i == 1
            k_min = k(1);
        else
            k_min = k_prim(i-1);
        end
        bc = @(x) value(x,b(i),k,v,p);
        k_prim(i) = gsearch(k_min,min(b(i),k(N)),bc);
        v_new(i) = F(b(i)-k_prim(i),p)+p.beta*interp1(k,v,k_prim(i));
    end
    diff = abs(v_new-v);
    if max(diff) < epsilon
        break
    end
end

c = b - k_prim;
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

function v = value(x,b,grid,v,p)
    v = F(b-x,p)+p.beta*interp1(grid,v,x);
end

function opt = gsearch(a,b,fun)
    t = (-1+sqrt(5))/2;
    epsilon = 10^(-6);
    x1 = (1-t)*a+ t*b;
    x2 = t*a+ (1-t)*b;
    for i = 1:1000
        if fun(x1) < fun(x2)
            b = x1;
            x1 = x2;
            x2 = t*a + (1-t)*b;
        else
            a = x2;
            x2 = x1;
            x1 = (1-t)*a + t*b;
        end
        if abs(x1-x2) < epsilon
            break
        end
    end

    if i == 1000
        fprintf('Warning! gsearch not converged \n')
    end
    
    opt = x1;
end