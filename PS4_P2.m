clear;
close all;

%===============================================================================================
%------------------------------PARAMETRIZATION AND INITIALIZATION-------------------------------
%===============================================================================================

p.alpha = 0.4;
p.beta = 0.99;
p.gamma = 3;
p.delta = 0.1;
p.theta = 8;
p.mu = 1;
p.rho = 0.95;
p.sigma = 0.007;
[k_ss,c_ss,n_ss] = steady(p); 

epsilon = 10^(-5);
N = 50;
M = 5;
k = linspace(0.9*k_ss,1.1*k_ss,N);
%%
%===============================================================================================
%---------------------------------------COMPUTATION---------------------------------------------
%===============================================================================================

%Computing state space for productivity
sigma_A = sqrt(p.sigma^2/(1-p.rho^2));
[A,p.pi] = rouwenhorst(p.rho,sigma_A,M);
A = exp(A);

%Initializing vectors for falue and policy functions
c = 0.2*ones(M,1)*k;
for i = 1:M
    n(i,:) = labor(c(i,:),k,A(i),p);
    k_prim(i,:) = A(i)*k.^p.alpha.*n(i,:).^(1-p.alpha)+(1-p.delta)*k-c(i,:);
end
c_hat = zeros(M,N);
x = zeros(M,N);
k_hat = ones(M,1)*k;
n_hat = zeros(M,N);
k_prim_new = zeros(M,N);

%Iteration
for s = 1:2000
    for i=1:N
        for j=1:M
            Eu = p.pi(j,:)*(F_c(c(:,i),p).*R(k(i),n(:,i),A,p)); %Expected value of R*u_c calculation
            c_hat(j,i) = F_c_inv(p.beta*Eu,p); %Consumption function for today
            x(j,i) = c(j,i)+k(i); %Tommorow's budget
            handle = @(xx) prod(xx,c_hat(j,i),A(j),x(j,i),p);
            k_hat(j,i) = lsqnonlin(handle, k_hat(j,i));
            n_hat(j,i) = labor(c_hat(j,i),k_hat(j,i),A(j),p);
        end
    end
    for j=1:M %Interpolation
        k_prim_new(j,:) = interp1(k_hat(j,:),k,k,'spline','extrap');
        c(j,:) = interp1(k_hat(j,:),c_hat(j,:),k,'spline','extrap');
        n(j,:) = interp1(k_hat(j,:),n_hat(j,:),k,'spline','extrap');
    end
    diffs=abs(k_prim-k_prim_new);
    if max(diffs) < epsilon %Checking the convergence criterion
        break
    end
    k_prim=k_prim_new;
end

%===============================================================================================
%----------------------------------------FUNCTIONS----------------------------------------------
%===============================================================================================

function [k,c,n] = steady(p) %computes the steady state variable values
    Q = ((1-p.beta+(p.beta*p.delta))/(p.alpha*p.beta))^(1/(p.alpha-1)); 
    %analytical solution of n_bar, note that when p.g is too low, nb can
    %"exceed" one and we restrict it to the corner solution.
    n = min(1,((((Q^p.alpha)-(p.delta*Q))^(-p.gamma))*(Q^p.alpha)*((1-p.alpha)/p.theta))...
    ^(1/(p.mu+p.gamma))); 

    k = Q*n; %accoring to analytical solution, k_bar = Q*n_bar

    c = (1/(p.alpha*p.beta)-(1-p.delta)/p.alpha-p.delta)*k;
end

function u = F(c,n,p) %Utility function
    u = (c.^(1-p.gamma)-1)/(1-p.gamma)-p.theta*n.^(1+p.mu)/(1+p.mu);
end

function u_c = F_c(c,p) %Utility function derivative over c
    u_c = c.^(-p.gamma);
end

function u_c_inv = F_c_inv(x,p) %Inverse of utility function derivative over c
    u_c_inv = x^(-1/p.gamma);
end

function c_imp = c_implied(k_prim,c,p) %c implied by Euler Equation
    c_imp = (p.beta*(p.alpha*k_prim.^(p.alpha-1)+1-p.delta).*c.^(-p.gamma)).^(-1/p.gamma);
end

function n = labor(c,k,A,p) %FOC for n given c
    n = min(((1-p.alpha)/p.theta*A*k.^p.alpha.*c.^(-p.gamma)).^(1/(p.alpha+p.mu)),1);
end

function a = R(k,n,A,p) %Gross return on capital
    a = p.alpha*A'*k^(p.alpha-1).*n.^(1-p.alpha)+1-p.delta;
end

function diff = prod(k,c,A,x,p) %difference between capital and product
    n = labor(c,k,A,p);
    y = A*k^p.alpha*n^(1-p.alpha) + (1-p.delta)*k;
    diff = y-x;
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
