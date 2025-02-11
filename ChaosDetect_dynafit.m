% Chaos detection in Logistic Map 
% Datasets from https://doi.org/10.17632/k4x675k5dm.2
% Training set : LM_TRAIN_Data_Paper.txt
% Test set : LM_TEST_Data_Paper.txt

clear
close all


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TRAIN DATASET : X
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Train = load('LM_TRAIN_Data_Paper.txt');

% chaotic if LE > 0.1, regular else 
ind_chaotic = find(Train(:,3) > 0.1); 
ind_regular = find(Train(:,3) <= 0.1); 
fprintf('%d regular training samples\n', length(ind_regular))
fprintf('%d chaotic training samples\n', length(ind_chaotic))
X_reg = Train(ind_regular,4:end);
X_chaos = Train(ind_chaotic,4:end);
X_reg = X_reg';
X_chaos = X_chaos';


tic
% TRAINING
% Build kernel Gramm matrix 
K_reg = LM_kernel_matrix(X_reg, X_reg);   
H_reg = train(K_reg); % train on regular data
K_chaos = LM_kernel_matrix(X_chaos, X_chaos);   
H_chaos = train(K_chaos); % train on chaotic data
fprintf('Learning time (sec) %f \n',toc)

% Training accuracy
K_reg_chaos = LM_kernel_matrix(X_reg, X_chaos);   
K_chaos_reg = K_reg_chaos';
% regular data 
reg_d = diag(K_reg-K_reg*H_reg*K_reg); 
chaos_d = diag(K_reg-K_reg_chaos*H_chaos*K_reg_chaos');
d = [reg_d chaos_d]; 
[~,class]=find(d==min(d,[],2)); % class 1 = regular, class 2 = chaotic
TR = length(find(class == 1)); % True regular
FC = length(find(class == 2)); % false chaotic
% chaotic data 
reg_d =diag(K_chaos-K_chaos_reg*H_reg*K_chaos_reg');
chaos_d = diag(K_chaos-K_chaos*H_chaos*K_chaos); 
d = [reg_d chaos_d]; 
[~,class]=find(d==min(d,[],2)); % class 1 = regular, class 2 = chaotic
TC = length(find(class == 2)); % True chaotic
FR = length(find(class == 1)); % false regular
fprintf('Training accuracy = %f\n', 100*(TR+TC)/(TR+TC+FR+FC));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TEST DATASET : Y
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Test = load('LM_TEST_Data_Paper.txt');
ind_chaotic = find(Test(:,3) > 0.1); 
ind_regular = find(Test(:,3) <= 0.1); 
fprintf('%d regular test samples\n', length(ind_regular))
fprintf('%d chaotic test samples\n', length(ind_chaotic))
Y_reg = Test(ind_regular,4:end);
Y_chaos = Test(ind_chaotic,4:end);
Y_reg = Y_reg';
Y_chaos = Y_chaos';

% Test accuracy
% regular data 
K1 = LM_kernel_matrix(Y_reg, Y_reg);
K2 = LM_kernel_matrix(Y_reg, X_reg);
reg_d = diag(K1-K2*H_reg*K2');
K2 = LM_kernel_matrix(Y_reg, X_chaos);
chaos_d = diag(K1-K2*H_chaos*K2');
d = [reg_d chaos_d]; 
[~,class]=find(d==min(d,[],2)); % class 1 = regular, class 2 = chaotic
TR = length(find(class == 1));
FC = length(find(class == 2));
% chaotic data 
K1 = LM_kernel_matrix(Y_chaos, Y_chaos);
K2 = LM_kernel_matrix(Y_chaos, X_reg);
reg_d = diag(K1-K2*H_reg*K2');
K2 = LM_kernel_matrix(Y_chaos, X_chaos);
chaos_d = diag(K1-K2*H_chaos*K2');
d = [reg_d chaos_d]; 
[~,class]=find(d==min(d,[],2)); % class 1 = regular, class 2 = chaotic
TC = length(find(class == 2));
FR = length(find(class == 1));
fprintf('Test accuracy = %f\n', 100*(TR+TC)/(TR+TC+FR+FC));


function H = train(K)
[eigenvecs, eigenvals] = eig(K);
D=diag(eigenvals);
ind=find(D>1e-8);
V = eigenvecs(:,ind);
invD = [];
for i=1:length(ind)
    invD(i,i) = 1/sqrt(D(ind(i)));
end
H = V*invD*invD*V';
end

function K = LM_kernel_matrix(X, Y)
% computes Gram matrix with Logistic Map kernel
% input: 
% X: lxp sample matrix with p sample vectors of l components   
% Y: lxq sample matrix with q sample vectors of l components 
% Output:
% K: pxq kernel Gram matrix
[N,px] = size(X);
[N,py] = size(Y);
K = [];
for i=1:px % number of X patterns
    for j=1:py % number of Y patterns
        K(i,j) = 0;
        for k=1:N 
            xy = X(k,i)*Y(k,j);
            K(i,j) = K(i,j) + xy/(1-xy);
        end
    end
end
end
