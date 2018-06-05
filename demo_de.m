% A very basic implementation of 
% Differential Evolution (DE) algorithm
% Reference:
% Andries, P. Engelbrecht. 
% "Computational intelligence: an introduction." (2007).
% -----------------------------------
% Author: Muharram Mansoorizadeh, mansoorm@basu.ac.ir
% -----------------------------------
function demo_de()
    %Initialize environment:
    close all hidden
    clear 
    clc 
    
    % Initialize parameters:
    pop_size = 50 ;         % population size
    max_generation = 10 ;   % Max generation
    crossover_rate = 0.1 ;  % Cross over rate
    
    %Global Information:
    BestX = zeros(max_generation , 2) ; 
    BestF = zeros(max_generation , 1) ; 
  
    alpha  = 1 ; % motion step control
    %prepare figures:
    % figure 1 shows generations 
    nrows1 = fix(sqrt(max_generation)) ; 
    ncols1 = ceil(max_generation/nrows1) ;
    
    %figure 2 shows path of each bird:
    nrows2 = fix(sqrt(pop_size)) ; 
    ncols2 = ceil(pop_size/nrows2) ;
    
    %Generate initial population:
    particles =[]  ; 
    
    for k=1:pop_size 
        particles(k).x(1) = fix(rand() * 10) - 5; 
        particles(k).x(2) = fix(rand() * 10) - 5;
        particles(k).fitness = fitness_de(particles(k).x) ; 
        particles(k).v = [0 0];
    end
    figure(1) , title('Generations ') ; 
    figure(2) , title('Particles') ; 
    
    fprintf (1 , 'Best Particles\n');
    fprintf (1 , 'Gen\tx1\tx2\tfit\n');
    
    %Evolution through generations:
    for t=1:max_generation
        %Update population:
        oparticles = particles ; 
        gbest = 1; 
        for k=1:pop_size
                %Mutation step
            n= [0 0] ; 
            while n(1) == n(2)
              n = ceil(rand(1,2)*pop_size)  ; 
            end
            dx = oparticles(n(2)).x - oparticles(n(1)).x ; 
            trial = oparticles(k).x + rand() * dx ;  
                %Cross over step
            mask = false(size(trial)) ; 
            first = ceil(rand()*numel(trial)) ; 
            mask(first) = true ; 
            next = 1 ; 
            while rand() < crossover_rate && ...
                  sum(mask) < numel(mask)
                mask(next) = 1 ; 
                next = next + 1 ; 
            end
                %Survival decision: keep parnet or child ?
            child = oparticles(k) ; 
            child.x(mask) = trial(mask);
            child.fitness = fitness_de(child.x) ; 
            if child.fitness < oparticles(k).fitness
                particles(k)   = child ; 
                particles(k).v = particles(k).x - oparticles(k).x ;
            end
                % Find best particle of the generation:
            if particles(k).fitness < particles(gbest).fitness
                gbest = k ; 
            end
        end
        
        %Update global information
        BestX(t, :) =particles(gbest).x  ; 
        BestF(t, :) =particles(gbest).fitness  ; 
        
         

        %Display generations
        fprintf (1 , '%d\t%2.2f\t%2.2f\t%2.2f\n', t, particles(gbest).x(1), ... 
                    particles(gbest).x(2), particles(gbest).fitness );
        %plot population:
        for k=1:pop_size
            figure(1),subplot(nrows1, ncols1,t)  ;
            hold on , plot (oparticles(k).x(1) , oparticles(k).x(2) , 'Ok') ;
            quiver(oparticles(k).x(1),oparticles(k).x(2), ...
                   particles(k).v(1),particles(k).v(2),'k');
            hold on , plot (particles(gbest).x(1) , particles(gbest).x(2) , '*k') ;
        end
        
        %plot motion paths:
        for k=1:pop_size
            figure(2),subplot(nrows2, ncols2,k)  ;
            hold on , plot(oparticles(k).x(1),oparticles(k).x(2),'.k')
        end
    end
    
    V = [0 0;diff(BestX)] ; 
    figure(3) , clf
    plot(BestX(:,1) ,BestX(:,2) , 'Ok' ) ; 
    hold on, plot(BestX(:,1) ,BestX(:,2) , '-b' ) ; 
    hold on, quiver(BestX(:,1) ,BestX(:,2),V(:,1) ,V(:,2) , 'Ok' ) ; 
    title('Best particles') ;
    
    figure(4) , clf 
    plot(1:max_generation , BestF ) ; 
    title('Best fitness') ;
    
    
end

function f = fitness_de(x )
% Ackley fitness function
%     f= -a*exp(-b*sqrt((1/d) *(x1.*x1 + x2.*x2))) - ...
%            exp ((1/d) *(cos(c*x1)+ cos(c*x2))) + a + exp(1) ;
%        
    a = 20;     b = 0.2;  c = 2*pi ;   d = 2 ; 

    sum1 = sum(x .* x);
    sum2 = sum(cos( c .* x)) ; 
    term1 = -a * exp(-b*sqrt(sum1./d));
    term2 = -exp(sum2./d);
    f = term1 + term2 + a + exp(1);
end
