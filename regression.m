% Linear regression
%%
Loss_train = zeros(1,10); Loss_test = zeros(1,10);
P_train =zeros(5,63,10); P_test = zeros(5,7,10);
for ii = 1:10
    W=[Xtrain(:,:,ii); ones(1,size(Xtrain,2))]'; T = Ytrain(:,:,ii)';
    Theta = inv(W'*W)*W'*T;
    % training loss
    L_train = sum(mean((Theta'*W' - T').^2,2));
    % testing loss
    W_test=[Xtest(:,:,1); ones(1,size(Xtest,2))]';T_test = Ytest(:,:,1)';
    L_test = sum(mean((Theta'*W_test' - T_test').^2,2));
    Loss_train(ii) = L_train; Loss_test(ii) = L_test;
    P_train(:,:,ii) = Theta'*W';P_test(:,:,ii) = Theta'*W_test';
end
%%
L = 7.9; n = 98 + 1;x = linspace(0, L, n)';
sinbase = [sin(1*pi*x/L),  sin(3*pi*x/L),  sin(5*pi*x/L), sin(7*pi*x/L), sin(9*pi*x/L)]';
%plot(x, sinbase'*Ytrain(:,end,1), x, sinbase'*P_train(:,end,1))
%hold on; 
plot(x, sinbase'*Ytest(:,1,1), x, sinbase'*P_test(:,1,1))