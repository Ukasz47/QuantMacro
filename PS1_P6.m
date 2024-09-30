clear;
close all;

%===============================================================================================
%------------------------------PARAMETRIZATION AND INITIALIZATION-------------------------------
%===============================================================================================

params.alpha = 0.36;
params.beta = 0.99;
params.sigma = 1;
params.delta = 0.025;
params.T = 300;
params.k_bar = ((1-params.beta*(1-params.delta))/(params.alpha*params.beta))^(1/(params.alpha-1));
params.k_0 = params.k_bar;
epsilon = 10^(-6);
h = 10^(-6); %numerical derivative step size
derv_switch = 0; % 0 - only forward difference used, 1 - average of forward and backward difference
i = 1;
r_t = 0.01*ones(params.T+1,1);
r_t_new = zeros(params.T+1,1);
r_t(1) = params.alpha*params.k_0^(params.alpha-1)-params.delta;
    r_t_new(1) = params.alpha*params.k_0^(params.alpha-1)-params.delta;
J = zeros(params.T,params.T);

%===============================================================================================
%----------------------------------------ITERATION----------------------------------------------
%===============================================================================================

dk = iter(r_t,params);
for j = 1:params.T
    r_t_forward = r_t;
    r_t_forward(j+1) = r_t_forward(j+1) + h;
    dk_forward = iter(r_t_forward,params);
    if derv_switch == 0
        J(:,j) = (dk_forward-dk)/h;
    else
        r_t_backward = r_t;
        r_t_backward(j+1) = r_t_backward(j+1) - h;
        dk_backward = iter(r_t_backward,params);
        J(:,j) = (dk_forward-dk_backward)/(2*h);
    end
end
dk = iter(r_t,params);

while max(abs(dk))/params.k_0 > epsilon
    i = i+1;
    r_t_new(2:params.T+1) = r_t(2:params.T+1) - J\dk;
    r_t_new = max(r_t_new,10^(-4)*ones(params.T+1,1));
    dk = iter(r_t_new,params);
    s = r_t_new(2:params.T+1) - r_t(2:params.T+1);
    J = J + dk*s'/(s'*s);
    r_t = r_t_new;
end

k_t = ((r_t+params.delta)/params.alpha).^(1/(params.alpha-1));
plot(k_t);

%===============================================================================================
%----------------------------------------FUNCTIONS----------------------------------------------
%===============================================================================================

function x = feasibility(k,k_1,c,params)
    x = k^params.alpha + (1-params.delta)*k - k_1 - c;
end

function dk = iter(r_t,params)
    k_t = ((r_t+params.delta)/params.alpha).^(1/(params.alpha-1));
    c_bar = params.k_bar^params.alpha-params.delta*params.k_bar;
    c_t(params.T+1) = c_bar;
    k_s_t = zeros(params.T+1,1);
    k_s_t(params.T+1) = k_t(params.T+1);

    options = optimoptions('lsqnonlin', 'Display', 'off');
    lb = 0;  
    ub = [];

    for t = params.T:-1:1
        c_t(t) = c_t(t+1)/(params.beta*(params.alpha*k_t(t+1)^(params.alpha-1)+1-params.delta));
        bc = @(k) feasibility(k,k_s_t(t+1),c_t(t),params); %auxiiary function to transmit consumption and future capital as parameters
        k_s_t(t) = lsqnonlin(bc, k_s_t(t+1), lb, ub, options);
    end
    dk = k_s_t(1:params.T)-k_t(1:params.T);
end
