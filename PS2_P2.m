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
l = 2;
k = 1;
T = 300;
k_0_ratio = 0.9;

%===============================================================================================
%---------------------------------------COMPUTATION---------------------------------------------
%===============================================================================================
% c)
D = initialize_D(params);
[Q,Lambda] = eig(D); %obtaining eigendecomposition
[~,order] = sort(diag(Lambda,0)); 

Q_prim = inv(Q); 
Q_prim = Q_prim(order,:); %sorting the system by increasing eigenvalues
a1 = -inv(Q_prim(l+1:l+k,l+1:l+k))*Q_prim(l+1:l+k,1:l); %computing policy function
a2 = D(1:l,1:l) + D(1:l,l+1:l+k)*a1; %computing law of motion

% d)
k_ss = (params.alpha/(1/params.beta-1+params.delta))^(1/(1-params.alpha)); %computing steady state values
c_ss = k_ss^params.alpha - params.delta*k_ss;
k_hat = zeros(T+1,1);
k_hat(1) = log(k_0_ratio);
for t = 1:T
    k_hat(t+1) = a2(1,1)*k_hat(t); %iterating capital through law of motion
end
c_hat = a1(1,1)*k_hat; %applying policy rule to compute consumption
k_t = exp(k_hat + log(k_ss)); %computing absolute values from log deviations
c_t = exp(c_hat + log(c_ss));

plot(k_t, 'LineWidth',3);        
legend('k_t');
xlabel('Time');
ylabel('Capital');
title('Plot of k_t');
%%
plot(c_t, 'LineWidth',3); 
legend('c_t');
xlabel('Time');
ylabel('Consumption');
title('Plot of c_t');

%%
% e)
x_hat = zeros(3,T+1); %vector of consumption, capital and productivity
x_hat(1,1) = 0;
x_hat(2,1) = 0.01;
for t = 1:T
    [x_hat(1:2,t+1)] = a2*[x_hat(1:2,t)]; %iterating state variables through law of motion
end
x_hat(3,:) = a1*x_hat(1:2,:);
x_ss = [k_ss;1;c_ss]*ones(1,T+1);
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

function D = initialize_D(params)
    D = zeros(3,3);
    D(1,1) = 1/params.beta;
    D(1,2) = (1/params.beta-1+params.delta)/params.alpha;
    D(1,3) = params.delta - (1/params.beta-1+params.delta)/params.alpha;
    D(2,2) = params.rho;
    D(3,1) = -(1-params.alpha)*(1/params.beta-1+params.delta)/params.gamma;
    D(3,2) = -(1-params.alpha)*(1/params.beta-1+params.delta)^2*params.beta/(params.alpha*params.gamma) ...
        +params.rho*(1-params.beta+params.beta*params.delta)/params.gamma;
    D(3,3) = 1 - (1-params.alpha)*(1-params.beta+params.beta*params.delta)/params.gamma ...
        *(params.delta-(1/params.beta-1+params.delta)/params.alpha);
end