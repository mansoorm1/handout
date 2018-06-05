% A very basic implementation of 
% Particle swarm optimization (PSO) algorithm
% -----------------------------------
% Author: Muharram Mansoorizadeh, mansoorm@basu.ac.ir
% -----------------------------------
function demo_pso()
    % Initialization
    pop_size = 5 ;         % population size
    max_generation = 10 ;   % Max generation
    c1 = 0.1 ; 
    c2 = 0.1 ;
    colors ='ymcrgbk';
    alpha  = 1 ; % motion step control
    %prepare figures:
    % figure 1 shows generations 
    nrows1 = fix(sqrt(max_generation)) ; 
    ncols1 = ceil(max_generation/nrows1) ;
    
    %figure 2 shows path of each bird:
    nrows2 = fix(sqrt(pop_size)) ; 
    ncols2 = ceil(pop_size/nrows2) ;

    %Generate initial population:
    for k=1:pop_size 
        birds(k).x(1) = fix(rand() * 10) - 5; 
        birds(k).x(2) = fix(rand() * 10) - 5;
        birds(k).v = [0 0];
        birds(k).lbest.x=[] ; % cognitive best place 
        birds(k).lbest.fitness = inf ; 
    end
    
    fprintf (1 , 'Best Birds\n');
    fprintf (1 , 'Gen\tx1\tx2\tfit\n');

    %Evolution through generations:
    for t=1:max_generation
        gbest = 1 ; 
        alpha = alpha * 0.9 ; 

        for k=1:pop_size
            birds(k).fitness = fitness(birds(k).x(1) , birds(k).x(2)) ;
            %fprintf (1 , '%2.2f\t%2.2f\t%2.2f\n' ,birds(k).x(1), birds(k).x(2) , birds(k).fitness);
            if birds(k).fitness < birds(gbest).fitness % Find global best
                gbest = k  ; 
            end

            if birds(k).fitness < birds(k).lbest.fitness % Find local best
                 birds(k).lbest.fitness = birds(k).fitness ;
                 birds(k).lbest.x = birds(k).x ; 
            end
        end

        %Update population:
        obirds = birds ; 
        for k=1:pop_size
            r1 = rand(1,2) ; 
            r2 = rand(1,2) ; 
            birds(k).v = birds(k).v + ...                       % Inertial
            c1 * (r1 .* (birds(k).lbest.x - birds(k).x)) + ...   % Cognitive 
            c2 * (r2 .* (birds(gbest).x - birds(k).x)) ;        % Social
            birds(k).x = birds(k).x + alpha * birds(k).v ; 
        end

        %Display generations
        fprintf (1 , '%d\t%2.2f\t%2.2f\t%2.2f\n', t, obirds(gbest).x(1), ... 
                    obirds(gbest).x(2), obirds(gbest).fitness );
        %plot population:
        for k=1:pop_size
            figure(1),subplot(nrows1, ncols1,t)  ;
            hold on , plot (obirds(k).x(1) , obirds(k).x(2) , 'Ok') ;
            quiver(obirds(k).x(1),obirds(k).x(2), ...
                   birds(k).v(1),birds(k).v(2),'k');
            hold on , plot (obirds(gbest).x(1) , obirds(gbest).x(2) , '*k') ;
        end
        
        %plot motion paths:
        for k=1:pop_size
            figure(2),subplot(nrows2, ncols2,k)  ;
            hold on , plot(obirds(k).x(1),obirds(k).x(2),'.k')
            hold on , quiver(obirds(k).x(1),obirds(k).x(2), ...
                   birds(k).v(1),birds(k).v(2),'k');
        end
    end   
end

function f = fitness(x1 , x2 )
% Ackley fitness function
%     f= -a*exp(-b*sqrt((1/d) *(x1.*x1 + x2.*x2))) - ...
%            exp ((1/d) *(cos(c*x1)+ cos(c*x2))) + a + exp(1) ;
%        
    a = 20;     b = 0.2;  c = 2*pi ;   d = 2 ; 

    sum1 = x1.^2 + x2.^2;
    sum2 = cos(c*x1) + cos(c*x2);
    term1 = -a * exp(-b*sqrt(sum1./d));
    term2 = -exp(sum2./d);
    f = term1 + term2 + a + exp(1);

end
