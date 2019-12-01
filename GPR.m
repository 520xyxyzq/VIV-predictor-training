% GPR
% import index.mat, CL.mat before running
%% Load coefficients

N = 20; % number of sine functions, divided by 2 gives N in report 
k = 10; % k-fold validation
Nd = 70; % total number of datasets 
Ntrain = Nd-Nd/k; % number of training datasets
Ntest = Nd/k; % number of testing datasets

L = 7.9; n = 98 + 1; % riser spatial sampling
x = linspace(0, L, n); % x axis along riser
sinbase = sin((1:1:N)'*pi*x/L); % set up sinbase to calculate CL from coefficients

% import ground truth, inputs
CL_test = csvread('Ytest2.csv'); % CL for testing
CL_train = csvread('Ytrain2.csv'); % CL for training
Xtest2 = csvread('Xtest2.csv');
Xtrain2 = csvread('Xtrain2.csv');
%% Approximate CL
% Constructing string type expression and coefficient cell 
expr = '0';coef_str = {};
for NN = 1:N
    expr = strcat(expr,'+a',num2str(NN),'*sin(',num2str(NN),'*pi*x/',num2str(L),')');
    coef_str{NN} = strcat('a',num2str(NN));
end

% Assigning expression and coefficients
fittp = fittype(expr, 'coefficients', coef_str);

Coef = [];
for cc = 1:size(CL, 2)
    fitted = fit(x', CL(:,cc),fittp);
    Coef = [Coef coeffvalues(fitted)'];
end

% Initialize Ytrain, Ytest
Ytrain = zeros(N,Ntrain, k);
Ytest = zeros(N,Ntest);

% load coefficients from Coef as ground truth
for kk = 1:k
    CC = Coef;
    Ytest(:,:,kk) = Coef(:,index(kk,:)); % fetch test sets
    CC(:,index(kk,:)) = []; % delete test sets
    Ytrain(:,:,kk) = CC(:,1:Ntrain);
end

%% k-fold Validation
% GPR k-fold validation
Loss = zeros(1,k);
Ltrain = zeros(1,k);
for kk = 1:1 % Set end value of kk as 1 to output results for report
    % Initialize parameters for testing 
    C_pred = zeros(N, Ntest);
    C_train = zeros(N, Ntrain);
    upper = zeros(N,Ntest);
    lower = zeros(N,Ntest);
    gpr_model = {}; 
    for NN = 1:N
        gprmdl = fitrgp(Xtrain2(:,kk*Ntrain-Ntrain+1:kk*Ntrain)', Ytrain(NN,:,kk)'); % fit GPR model
        gpr_model{NN} = gprmdl; % store fitted gpr model in a cell
        [c_pred,~,ul] = predict(gprmdl, Xtest2(:,kk*Ntest-Ntest+1:kk*Ntest)'); % predict one of coefficients NN(1~N) in test sets
        assert (size(c_pred,1) == Nd/k && size(c_pred,2) == 1, 'c_pred size incorrect');
        c_train = predict(gprmdl, Xtrain2(:, kk*Ntrain-Ntrain+1:kk*Ntrain)'); % predict training sets
        assert (size(c_train,1) == Ntrain && size(c_train,2) == 1, 'c_train size incorrect');
        C_train(NN,:) = c_train';
        C_pred(NN,:) = c_pred';
        upper(NN,:) = ul(:,1)';
        lower(NN,:) = ul(:,2)';
    end
    CL_train_pred = (C_train'*sinbase)'; % mapping
    CL_pred = (C_pred'*sinbase)'; % map predicted coefficients to predicted CL
    up_pred = (upper'*sinbase)';
    lw_pred = (lower'*sinbase)';
    Ltrain(1,kk) = mean(mean((CL_train(:,kk*Ntrain-Ntrain+1:kk*Ntrain) - CL_train_pred).^2,1)); % Calulate MSE  
    Loss(1,kk) = mean(mean((CL_test(:,kk*Ntest-Ntest+1:kk*Ntest) - CL_pred).^2,1));  
end
%%
% Visualize results (from the last cycle)
n_disp = 1; % the number of test performance to plot
% plot the predicted
% divide the curves by 2.5/1.5 to reduce amplitude (only for report)
% CL in practice should stay largely below 0.5
hold on;plot(x,CL_pred(:,n_disp)'/2.5);
%%
% plot the confidence intervals
% color grey
patch([x, fliplr(x)], [up_pred(:,n_disp)'/2.5, fliplr(lw_pred(:,n_disp)'/2.5)], 'm', 'FaceColor',[0.75 0.75 0.75],'EdgeColor','none');
% set transparency
alpha(0.4)
disp(['The average training loss is ' num2str(mean(Ltrain)/kk*k)]) % mean of training loss
disp(['The average testing loss is ' num2str(mean(Loss)/kk*k)]) % mean of loss over kk/10 to make up for kk=1 case