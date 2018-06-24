%A very basic implementation of lstm networks for
% 1D time series prediction
% -----------------------------------
% Author: Muharram Mansoorizadeh, mansoorm@basu.ac.ir
% -----------------------------------

function demo_lstm()
    close all hidden
    clear
    clc
    global net data 
    
    %Generate a simple time series
    t = linspace(-2*pi,2*pi,1000) ; 
    data = sin(t) ; 
    % prepare data:
    w=1 ; % number of previous samples used for prediction
    lstm_train(data , w) ;
    [e, x2] = lstm_eval(net , data ) ; 
    
    figure, plot(data)
    hold on, plot( x2) 
    title(['RMSE = ' num2str(e)]);
end % end of main function

function lstm_train(x,w)
global data data_x data_y net

%input 
% x is a 1D time series
% w is the number of previous samples used for prediction
    
    %prepare data:
    n  = length (x) ; 
    
    x=[zeros(1,w) x] ; % zero-padd x 
    data_x = zeros(n,w) ; 
    data_y = zeros(n,1) ; 
    for k=1:n-w-1
        data_x(k,:) = x(k:k+w-1) ; 
        data_y(k) = x(k+w);
    end
    
    %setup network:
    m = size(data_x,2) ; 
    net.lag = w ; 
    net.wf  = 0.01 * randn (1+ m, 1) ; 
    net.wi  = 0.01 * randn (1+ m, 1) ; 
    net.wc  = 0.01 * randn (1+ m, 1) ; 
    net.wo  = 0.01 * randn (1+ m, 1) ; 
    net.bf  = 0  ; 
    net.bi  = 0  ;  
    net.bo  = 0  ; 
    net.bc  = 0  ; 
    
    x0 = [ ...
        net.wf ;...
        net.wi ;...
        net.wc ;...
        net.wo ;...
        net.bf ;...
        net.bi ;...
        net.bo ;...
        net.bc ;...
    ];
    options = optimset() ;
    options.MaxFunEvals  = 10000 ; 
    weights = fminsearch(@lstm_test, x0 , options ) ; 
end


function e = lstm_test(x)
global net data

        %Extract network parameters from x:
        idx = 1; 
        net.wf = x(idx:idx+length(net.wf)-1) ; idx = idx + length(net.wf) ; 
        net.wi = x(idx:idx+length(net.wi)-1) ; idx = idx + length(net.wi) ; 
        net.wc = x(idx:idx+length(net.wc)-1) ; idx = idx + length(net.wc) ; 
        net.wo = x(idx:idx+length(net.wo)-1) ; idx = idx + length(net.wo) ; 
        net.bf = x(idx:idx+length(net.bf)-1) ; idx = idx + length(net.bf) ; 
        net.bi = x(idx:idx+length(net.bi)-1) ; idx = idx + length(net.bi) ; 
        net.bo = x(idx:idx+length(net.bc)-1) ; idx = idx + length(net.bc) ; 
        net.bc = x(idx:idx+length(net.bo)-1) ; idx = idx + length(net.bo) ; 
        
        %Evaluate the proposed weights:
        e = lstm_eval(net , data ) ;       
end

function [e,yhat] = lstm_eval(net , x  ) 
    h = 0 ; % h from previous step, h_t-1
    C = 0 ; % C from previous step, C_t=1
    yhat = zeros(size(x));
    y    = x ; 
    x = [zeros(1,net.lag) x] ; 
    for k =1 :length(x) - net.lag
        x_k = [h x(k:k+net.lag-1)] ; 
        f = S(x_k * net.wf + net.bf);
        i = S(x_k * net.wi + net.bi);
        o = S(x_k * net.wo + net.bo);
        Ch = tanh(x_k * net.wc + net.bc) ;
        C = f * C + i * Ch ;
        yhat(k) = o * tanh(C) ; 
        
    end
    e = y - yhat ; 
    e = sum(sum(e .* e)) ; 
    e = sqrt(e / length(x)) ;
     disp(e) ;
end

function y=S(x)
    y = 1 ./ (1 + exp(x)) ; 
end
