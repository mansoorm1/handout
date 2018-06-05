% ROC curve demo 
% -----------------------------------
% Author: Muharram Mansoorizadeh, mansoorm@basu.ac.ir
% -----------------------------------
function demo_roc()
    % Initialize workspace
    close all hidden
    clear 
    clc
    % Generate a tiny two-class classification problem
    %class 0, sampled from a two-dimensional normal distribution 
    % centered at (0.5, 0.5).
    %class 1, same as class 1 but centered at (-0.5, -0.5). 

    x1 = randn(100,2) + 0.5 ; y1 = zeros(100,1) ; %class 0
    x2 = randn(100,2) - 0.5 ; y2 = ones(100,1) ;  %class 1
    x =[x1;x2] ; y=[y1;y2] ; 

    train_x = x(1:2:end,:) ; train_y = y(1:2:end,:) ; 
    test_x = x(2:2:end,:) ;  test_y= y(2:2:end,:) ; 

    [~ , ~, posterior] = classify(test_x, train_x, train_y ,'linear') ; 
    p_yhat = posterior(:,2) ;% probaliblity for class = 1
    % ROC analysis
    % iterate over various levels of threshold and get the 
    % true-positive and false-positive rates
    TP =[] ;
    FP = [] ;
    p = sum(test_y == 1) ; % number of positive samples
    n = sum(test_y == 0); % number of negative samples
    for th =0:0.01:1 % threshold levels
        yhat = p_yhat >= th ; 
        t = sum(yhat == 1 & test_y == 1)/ p ; % true positive rate
        f = sum(yhat == 1 & test_y == 0)/ n ; % false positive rate 
        TP =[TP;t] ; FP=[FP;f] ; 
    end
    %area under curve is summed up by trapezoidal integration
    auc = 0 ;
    for k=2:numel(FP)
         auc = auc + abs((TP(k) +TP(k-1)) * (FP(k)-FP(k-1))/2)  ; 
    end

    figure, plot (FP,TP ) ; % , '.') ;
    title(['AUC = ' num2str(auc)]);
    xlabel('false positive rate(fpr)') ;
    ylabel('true positive rate(tpr)') ;
end
