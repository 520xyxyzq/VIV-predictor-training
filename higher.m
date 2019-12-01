% Generalizing to higher mode (mode 6)
% run GPR with kk =1:1 before this script 
% run command already included in 2nd section 
% Only setting kk is enough 

%% Generate inputs and output 

% Gnereate a mode 6 case
U = 3.5; % flow speed 
run Non_iterative.m;
Cl = Fr;a = Ar;m = inn;w = wni(inn);u = U; % CL A mode w u

% Mean and stds of training sets in the first fold 
% 63 sets defined by index.mat
% Obtained from running the 2nd section of split.m with kk = 1:1
u_m = 1.5092; u_std = 0.9180;
w_m = 61.5094; w_std = 37.2016;
A_m = 0.0245; A_std = 0.0039;

% Standardizing inputs
u = (u - u_m)/u_std; w = (w - w_m)/w_std; a = (a - A_m)/A_std;
x_high = [u,w,a,m]'; % set up input vector (4 by 1)

%% Train GPR predictor

% run GPR to train the predictor from fold 1
% remeber to set kk = 1:1
run GPR.m
%% Predict
% Initialize variables to store coefficients and upper-lower bounds 
% from sine functions of order NN = 1:N
% N defined in GPR.m
C_high = zeros(N,1);
upper  = zeros(N,1);
lower  = zeros(N,1);

% predict coefficients and upper-lower bounds
for NN = 1:N
    [c_high,~,ul_high] = predict(gpr_model{NN}, x_high');
    C_high(NN,1) = c_high;
    upper(NN,1) = ul_high(:,1);
    lower(NN,1) = ul_high(:,2);
end

% map cj to CL
% sinbase defined in GPR.m
CL_high = (C_high' * sinbase)';
up_high = (upper' * sinbase)';
lw_high = (lower' * sinbase)';
loss_high = mean((CL_high - Cl).^2);
%% Results Visualization

% x (axis) defined in GPR.m 
scatter(x, Cl')
hold on;plot(x,CL_high');

% 95% confidence interval
% color grey
patch([x, fliplr(x)], [up_high', fliplr(lw_high')], 'm', 'FaceColor',[0.75 0.75 0.75],'EdgeColor','none');
% set transparency
alpha(0.4)
disp(['prediction loss is ' num2str(loss_high)])