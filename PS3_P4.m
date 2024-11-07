clear;
close all;

%===============================================================================================
%------------------------------PARAMETRIZATION AND INITIALIZATION-------------------------------
%===============================================================================================

p.alpha = 0.4;
p.beta = 0.99;
p.sigma = 2;
p.delta = 0.1;
p.theta = 0.5;
p.mu = 1;
[k_ss,c_ss,n_ss] = steady(p);

epsilon = 10^(-6);
N = 51;
k = linspace(8,15,N);
v_new = 10*log(k);
%%
%===============================================================================================
%---------------------------------------COMPUTATION---------------------------------------------
%===============================================================================================

%Defining handles for policy function
c = zeros(1,N);
k_prim = zeros(1,N);
n = zeros(1,N);
%Computing maximal available budget under given k
b = k.^p.theta+(1-p.delta)*k;

for s = 1:2000
    v = v_new;
    for i = 1:N
        if b(i) < k(N) %Finding index of maximal considered k_prim
            i_max = find((k-b(i))>0,1)-1;
        else
            i_max = N;
        end
        if i == 1 %Restricting consumption choice since c increases with k
            c_min = 0.2;
        else
            c_min = c(i-1);
        end
        bc = @(x) value(x,k(i),k,v,p); %Defining function handle fixing k and value function
        c(i) = gsearch(c_min,b(i),bc); %Finding optimal consumption
        n(i) = labor(c(i),k(i),p); %Finding corresponding labor supply
        k_prim(i) = k(i)^p.alpha*n(i)^(1-p.alpha)+(1-p.delta)*k(i)-c(i); %Finding corresponding k_prim
        v_new(i) = F(c(i),n(i),p)+p.beta*interp1(k,v,k_prim(i),'linear','extrap'); %Computing updated value function
    end
    diff = abs(v_new-v);
    if max(diff) < epsilon %Checking the convergence criterion
        break
    end
end

c_imp = c_implied(k_prim,c,p);
Euler_error = (c_imp-c)./c;

%===============================================================================================
%----------------------------------------FUNCTIONS----------------------------------------------
%===============================================================================================

function [k,c,n] = steady(p) %computes the steady state variable values
    Q = ((1-p.beta+p.beta*p.delta)/(p.alpha*p.beta))^(-1/(1-p.alpha));

    n = ((1-p.alpha)/p.theta*Q^(-p.alpha)*(Q^p.alpha-p.delta*Q)^(-p.sigma))^(1/(p.mu+p.sigma));

    k = p.alpha^(1/(1-p.alpha))*(1/p.beta-1+p.delta)^(-1/(1-p.alpha))*n;

    c = (1/(p.alpha*p.beta)-(1-p.delta)/p.alpha-p.delta)*k;
end

function u = F(c,n,p) %Utility function
    u = (c.^(1-p.sigma)-1)/(1-p.sigma)-p.theta*n.^(1+p.mu)/(1+p.mu);
end

function c_imp = c_implied(k_prim,c,p) %c implied by Euler Equation
    c_imp = (p.beta*(p.theta*k_prim.^(p.theta-1)+1-p.delta).*c.^(-p.sigma)).^(-1/p.sigma);
end

function n = labor(c,k,p) %FOC for n given c
    n = min(((1-p.alpha)/p.theta*k^p.alpha*c^(-p.sigma))^(1/(p.alpha+p.mu)),1);
end

function val = value(x,k,grid,v,p) %Continuation utility under given choice of c
    n = labor(x,k,p);
    k_prim = min(k^p.alpha*n^(1-p.alpha)+(1-p.delta)*k-x,max(grid));
    val = F(x,n,p)+p.beta*interp1(grid,v,k_prim,'linear','extrap');
end

function opt = gsearch(a,b,fun) %General golden search algorithm
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