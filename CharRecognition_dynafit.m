% Char recognition using dynafit
clear
close all 

n_in = 100 % classification based on the n_in first samples in the trajectory 
kernel_type = 1 % (1:polynomial, 2:RBF) 

if kernel_type == 1 
   fprintf('Polynomial kernel\n')
   kernel_param = 2
elseif kernel_type == 2
   fprintf('RBF kernel\n')
    if n_in == 10
        kernel_param = 2.9
    elseif n_in == 20
        kernel_param = 4.2
    elseif n_in == 50
        kernel_param = 7.5
    elseif n_in == 100
        kernel_param = 15.0
    end
else
    fprintf('Kernel not implemented\n')
    return
end

nb_runs = 100;

% Dataset from https://archive.ics.uci.edu/dataset/175/character+trajectories
load mixoutALL_shifted.mat 

n_samples = length(mixout); 

n_classes = 20;
for class=1:n_classes 
    ind = find(consts.charlabels == class);
    X{class} = [];
    for i=1:length(ind)
        data_tmp = mixout{ind(i)}; 
        j = find(data_tmp(1,:)~=0 | data_tmp(2,:)~=0 | data_tmp(3,:)~=0); % on enleve les zeros du debut
        if j(1)+n_in-1 <= length(data_tmp)
            data = data_tmp(:,j(1):j(1)+n_in-1);
        else
            data = data_tmp(:,j(1):length(data_tmp));
            data = [data zeros(3, j(1)+n_in-length(data_tmp)-1)]; % on complete avec des zeros si trajectoire trop courte 
        end
        data_vec = data(:);
        X{class} = [X{class} data_vec];
    end
    % Affiche example de char
    figure(1)
    x(1)=0; y(1)=0;
    for k=1:n_in
        x(k+1)=x(k)+data(1,k);
        y(k+1)=y(k)+data(2,k);
    end
    subplot(5,4,class)
    hold on
    plot(x,y, 'k','LineWidth',2)
    axis off
    axis square
    title(consts.key(class))
    drawnow

end


for run=1:nb_runs 

    % Training and Test sets
    for class=1:n_classes
        % data permutation des data : 10% training 90% testing
        r = 0.1; 
        n = size(X{class},2);
        tr = randperm(n);
        train_trial{class} = tr(1:round(r*n));
        test_trial{class} = tr(round(r*n)+1:end);         
        X_train{class} = X{class}(:,train_trial{class});
        X_test{class} = X{class}(:,test_trial{class});        
        if run==1 
            fprintf('class %d\t n_samples %d (Train:%d, Test:%d)\n', class, n, length(train_trial{class}), length(test_trial{class}))
        end
    end

    % Training 
    tic
    for class=1:n_classes
        % Build Kernel Gramm matrix for each class
        K = kernel_matrix(kernel_type, X_train{class}, X_train{class}, kernel_param);
        % solve eigenproblem
        [eigenvecs, eigenvals] = eig(K);
        D=diag(eigenvals);
        ind=find(D>1e-3); 
        V = eigenvecs(:,ind);
        inv_D = [];
        for i=1:length(ind)
            inv_D(i,i) = 1/sqrt(D(ind(i))); 
        end
        H{class} = V*inv_D*inv_D*V';        
    end
    TrainingTime(run) = 1000*toc;
    
    %% Test 
    err=0; ok=0;
    for test_class=1:n_classes
            K1 = kernel_matrix(kernel_type, X_test{test_class}, X_test{test_class}, kernel_param);
            d = [];
            for train_class=1:n_classes
                K2 = kernel_matrix(kernel_type, X_test{test_class}, X_train{train_class}, kernel_param);
                d = [d diag(K1-K2*H{train_class}*K2')]; % diag donne les distances pour chaque test sample
            end
            [~,class] = find(d==min(d,[],2)); 
            ok = ok + length(find(class == test_class)); 
            err = err + length(find(class ~= test_class)); 
    end
    accuracy(run) = 100*ok/(ok+err);
    fprintf('run %d accuracy  = %f\n', run, accuracy(run));
end
fprintf('Test accuracy (in percent) = %f +/- %f \n', mean(accuracy),std(accuracy));
fprintf('Training Time (ms) = %f +/- %f \n',mean(TrainingTime),std(TrainingTime))


function K = kernel_matrix(kernel_type, X, Y, kernel_param)
% computes Gram matrix with linear, polynomial and RBF kernels
% Inputs:
% Kernel_type (1:polynomial, 2:RBF) 
% X: lxp sample matrix with p sample vectors of l components   
% Y: lxq sample matrix with q sample vectors of l components 
% kernel_param: Kernel parameter (degree for polynomial, kernel width for RBF)
% Output:
% K: pxq kernel Gram matrix
    switch kernel_type
        case 1;
            %Polynomial Kernel
            K = (X'*Y + 1).^kernel_param;
        case 2;
            % Gaussian/RBF Kernel
            K = exp(-(dist(X',Y)/kernel_param).^2);
        otherwise
            K = 0;
    end
end

