clear;
close all;

%===============================================================================================
%------------------------------PARAMETRIZATION AND INITIALIZATION-------------------------------
%===============================================================================================

p.alpha = 0.4;
p.beta = 0.99;
p.gamma = 2;
p.delta = 0.1;
p.theta = 8;
p.mu = 1;
p.rho = 0.95;
p.sigma = 0.007;
[k_ss,c_ss,n_ss] = steady(p);

epsilon = 10^(-6);
N = 100;
M = 5;
policy_init = 10;
k = linspace(0.9*k_ss,1.1*k_ss,N);
%%
%===============================================================================================
%---------------------------------------COMPUTATION---------------------------------------------
%===============================================================================================

%Computing state space for productivity
sigma_A = sqrt(p.sigma^2/(1-p.rho^2));
[A,p.pi] = rouwenhorst(p.rho,sigma_A,M);
A = exp(A);

%Defining handles for falue and policy functions
v_new = 10*ones(M,1)*log(k)-40;
c = zeros(M,N);
k_prim = zeros(M,N);
n = zeros(M,N);
%Computing maximal available budget under given k
b = zeros(M,N);
for i = 1:M
    b(i,:) = A(i)*k.^p.alpha+(1-p.delta)*k;
end

for s = 1:2000
    v = v_new;
    for i = 1:N
        for j = 1:M
            if i == 1 %Restricting consumption choice since c increases with k
                c_min = 0.2;
            else
                c_min = c(j,i-1);
            end
            bc = @(x) value(x,k(i),A(j),j,k,v,p); %Defining function handle fixing k and value function
            c(j,i) = gsearch(c_min,b(j,i),bc); %Finding optimal consumption
            n(j,i) = labor(c(j,i),k(i),A(j),p); %Finding corresponding labor supply
            k_prim(j,i) = A(j)*k(i)^p.alpha*n(j,i)^(1-p.alpha)+(1-p.delta)*k(i)-c(j,i); %Finding corresponding k_prim
            v_new(j,i) = value(c(j,i),k(i),A(j),j,k,v,p); %Computing updated value function
        end
    end
    diff = abs(v_new-v);
    if max(diff) < epsilon %Checking the convergence criterion
        break
    end

    if s > policy_init
        for ss = 1:2000
            for i = 1:N
                for j= 1:M
                    v_new(j,i) = value(c(j,i),k(i),A(j),j,k,v,p);
                end
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

function [k,c,n] = steady(p) %computes the steady state variable values
    Q = ((1-p.beta+(p.beta*p.delta))/(p.alpha*p.beta))^(1/(p.alpha-1)); 
%analytical solution of n_bar, note that when p.g is too low, nb can
%"exceed" one and we restrict it to the corner solution.
    n = min(1,((((Q^p.alpha)-(p.delta*Q))^(-p.gamma))*(Q^p.alpha)*((1-p.alpha)/p.theta))...
    ^(1/(p.mu+p.gamma))); 

    k = Q*n;%accoring to analytical solution, k_bar = Q*n_bar

    c = (1/(p.alpha*p.beta)-(1-p.delta)/p.alpha-p.delta)*k;
end

function u = F(c,n,p) %Utility function
    u = (c.^(1-p.gamma)-1)/(1-p.gamma)-p.theta*n.^(1+p.mu)/(1+p.mu);
end

function c_imp = c_implied(k_prim,c,p) %c implied by Euler Equation
    c_imp = (p.beta*(p.alpha*k_prim.^(p.alpha-1)+1-p.delta).*c.^(-p.gamma)).^(-1/p.gamma);
end

function n = labor(c,k,A,p) %FOC for n given c
    n = min(((1-p.alpha)/p.theta*A*k^p.alpha*c^(-p.gamma))^(1/(p.alpha+p.mu)),1);
end

function val = value(x,k,A,j,grid,v,p) %Continuation utility under given choice of c
    n = labor(x,k,A,p);
    k_prim = min(A*k^p.alpha*n^(1-p.alpha)+(1-p.delta)*k-x,max(grid));
    cont = zeros(size(v,1),1);
    for i = 1:size(v,1)
        cont(i) = interp1(grid,v(i,:),k_prim,'linear','extrap');
    end
    val = F(x,n,p)+p.beta*p.pi(j,:)*cont;
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

function [z,pi_new] = rouwenhorst(rho,sigma,N)
    z=linspace(-sqrt(N-1)*sigma,sqrt(N-1)*sigma,N);
    p = (1+rho)/2;
    pi=[p,1-p;1-p,p];
    for i=3:N
        pi_new=zeros(i,i);
        pi_new(1:i-1,1:i-1) = pi_new(1:i-1,1:i-1)+p*pi;
        pi_new(2:i,2:i) = pi_new(2:i,2:i)+p*pi;
        pi_new(1:i-1,2:i) = pi_new(1:i-1,2:i)+(1-p)*pi;
        pi_new(2:i,1:i-1) = pi_new(2:i,1:i-1)+(1-p)*pi;
        
        pi_new(2:i-1,:)=pi_new(2:i-1,:)/2;
        pi=pi_new;
    end
end
