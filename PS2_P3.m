clear;
close all;

%===============================================================================================
%------------------------------PARAMETRIZATION AND INITIALIZATION-------------------------------
%===============================================================================================

params.alpha = 0.33;
params.beta = 0.99;
params.gamma = 1.0001;
params.delta = 0.023;
params.rho = 0.97;
params.mu = 1;
params.theta = 0.76;
l = 2;
k = 1;
T = 300;
[k_ss,c_ss,l_ss]  = steady(params);

%===============================================================================================
%---------------------------------------COMPUTATION---------------------------------------------
%===============================================================================================
% c)
A = initialize_A(params);
B = initialize_B(params);
[a1,a2] = solab(A,B,l);

% e)
x_hat = zeros(4,T+1); %vector of consumption, capital and productivity
x_hat(1,1) = 0;
x_hat(2,1) = 0.01;
for t = 1:T
    [x_hat(1:2,t+1)] = a2*[x_hat(1:2,t)]; %iterating state variables through law of motion
end
x_hat(3:4,:) = a1*x_hat(1:2,:);
x_ss = [k_ss;1;c_ss;l_ss]*ones(1,T+1);
x_t = exp(x_hat + log(x_ss));

plot(x_t(1,:)', 'LineWidth',3);        
legend('k_t');
xlabel('Time');
ylabel('Capital');
title('Plot of k_t');
%%
plot(x_t(3,:)', 'LineWidth',3); 
legend('c_t');
xlabel('Time');
ylabel('Consumption');
title('Plot of c_t');

%===============================================================================================
%----------------------------------------FUNCTIONS----------------------------------------------
%===============================================================================================

function A = initialize_A(params)
    A = zeros(4,4);
    A(1,1) = 1;
    A(2,2) = 1;
    A(3,1) = (1-params.alpha)*(1-params.beta+params.beta*params.delta)/params.gamma;
    A(3,2) = -(1-params.beta+params.beta*params.delta)/params.gamma;
    A(3,3) = 1;
    A(3,4) = -(1-params.alpha)*(1-params.beta+params.beta*params.delta)/params.gamma;
end

function B = initialize_B(params)
    B = zeros(4,4);
    B(1,1) = 1/params.beta;
    B(1,2) = (1/params.beta-1+params.delta)/params.alpha;
    B(1,3) = params.delta - (1/params.beta-1+params.delta)/params.alpha;
    B(1,4) = (1-params.alpha)*(1/params.beta-1+params.delta)/params.alpha;
    B(2,2) = params.rho;
    B(3,3) = 1;
    B(4,1) = 1-params.alpha;
    B(4,2) = -1;
    B(4,3) = params.gamma;
    B(4,4) = params.mu + params.alpha;
end

function [k,c,l] = steady(params) %computes the steady state variable values
    l = params.theta^(-1/(1+params.mu+params.gamma))*(params.alpha/(1-params.alpha))...
        ^(-1/(1+params.mu+params.gamma))*(params.alpha^(params.alpha/(1-params.alpha))*...
        (1/params.beta-1+params.delta)^(-(params.alpha/(1-params.alpha))-1/params.gamma)-params.delta*...
        params.alpha^(1/(1-params.alpha))*(1/params.beta-1+params.delta)^(-1/(1-params.alpha)...
        -1/params.gamma))^(-params.gamma/(1+params.mu+params.gamma));

    k = params.alpha^(1/(1-params.alpha))*(1/params.beta-1+params.delta)^(-1/(1-params.alpha))*l;

    c = (1/(params.alpha*params.beta)-(1-params.delta)/params.alpha-params.delta)*k;
end

% Function solab by Paul Klein -> https://paulklein.ca/newsite/codes/codes.php
function [f,p] = solab(a,b,nk);

[s,t,q,z] = qz(a,b);
[s,t,q,z] = ordqz(s,t,q,z,'udo');
z21 = z(nk+1:end,1:nk);
z11 = z(1:nk,1:nk);

if rank(z11)<nk;
	error('Invertibility condition violated')
end

z11i = z11\eye(nk);
s11 = s(1:nk,1:nk);
t11 = t(1:nk,1:nk);

if abs(t(nk,nk))>abs(s(nk,nk)) | abs(t(nk+1,nk+1))<abs(s(nk+1,nk+1));
   warning('Wrong number of stable eigenvalues.');
end

dyn = s11\t11;

f = real(z21*z11i);
p = real(z11*dyn*z11i);
end