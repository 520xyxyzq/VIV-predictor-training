% Generalize to multi-mode
% import index.mat before this script 

%% Simulate mode 4 5 coupling case

% Firstly run a marginal mode 5 case
U = 2.47; % marginal mode 5 
run Non_iterative.m;
Cl = Fr;a = Ar;m = inn;w = wni(inn);u = U; % CL A mode w u

% Add 33% of Cl from marginal marginal mode 4 case
% Simulate influence from mode 4
U = 2.43; % marginal mode 4
run Non_iterative.m;
Cl = Fr/3 + Cl;

%% Generate inputs and outputs

% Mean and stds of training sets in the 4th fold 
% because U = 2.5 is unseen in training
% Obtained from running the 2nd section of split.m with kk = 4:4

u_m = 1.5130; u_std = 0.9275;
w_m = 61.4185; w_std = 36.9943;
A_m = 0.0247; A_std = 0.0041;

% Standardizing inputs
u = (u - u_m)/u_std; w = (w - w_m)/w_std; a = (a - A_m)/A_std;
x_multi = [u,w,a,m]'; % set up input vector (4 by 1)

%% Train GPR predictor

% run GPR to train the predictor from fold 4
% remeber to set kk = 4:4
run GPR.m

%% Predict

% Initialize variables to store coefficients and upper-lower bounds 
% from sine functions of order NN = 1:N
% N defined in GPR.m
C_multi = zeros(N,1);
upper  = zeros(N,1);
lower  = zeros(N,1);

% predict coefficients and upper-lower bounds
for NN = 1:N
    [c_multi,~,ul_multi] = predict(gpr_model{NN}, x_multi');
    C_multi(NN,1) = c_multi;
    upper(NN,1) = ul_multi(:,1);
    lower(NN,1) = ul_multi(:,2);
end

% map cj to CL
% sinbase defined in GPR.m
CL_multi = (C_multi' * sinbase)';
up_multi = (upper' * sinbase)';
lw_multi = (lower' * sinbase)';
loss_multi = mean((CL_multi - Cl).^2);

%% Results Visualization

% x (axis) defined in GPR.m 
scatter(x, Cl')
hold on;plot(x,CL_multi');

% 95% confidence interval
% color grey
patch([x, fliplr(x)], [up_multi', fliplr(lw_multi')], 'm', 'FaceColor',[0.75 0.75 0.75],'EdgeColor','none');
% set transparency
alpha(0.4)
disp(['prediction loss is ' num2str(loss_multi)])