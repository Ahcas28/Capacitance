
proportions = [];
for gamma = 0:0.5:4
    N = 100;
    cx = [0:1:N-1]';
    cy = zeros(N,1);
    cz = zeros(N,1);
    c = [cx cy cz];
    
    R = 0.3*ones(N,1)';
    %%% Plot the geometry
    figure, hold on
    t = linspace(0,2*pi);
    for n = 1:N
        plot(c(n,1)+R(n)*cos(t), c(n,2)+R(n)*sin(t),'k')  
        text(c(n,1),c(n,2),num2str(n))
    end
    daspect([1 1 1])
    hold off
    close 
    
    vol = 4*pi*R.^3/3;
    k0 = 0.0000001;
    delta = 10^(-5);
    v2 = ones(1,N);
    N_multi = 2;
    %%% Compute the static capacitance matrix
    % Maximum order for multipole expansion (n = 0, 1, ..., N_multi)
    % If we use higher order, then accuracy improves. Usually 0 is sufficiently large.
    gamma_skin = gamma;
    
    matC_static = MakeC_mn(R,c,k0,N_multi);
    GCM_static = diag(delta.*v2./vol)*matC_static;
    
    %%% Compute eigenmodes
    [evec_static,eval_static] = eig(GCM_static);
    [resonances_static,I] = sort(sqrt(diag(eval_static)),'ComparisonMethod','real');
    modes_static = evec_static(:,I);
    
    
    % compute the normalization factor
    fun = @(theta,phi,r) exp(gamma_skin*r*sin(theta)*cos(phi))*r^2*sin(theta);
    int_A = int_trapez_3(fun,200,0,pi,0,2*pi,0,R(1));
    A_norm = exp(gamma_skin*cx)*int_A;
    
    %%% Compute the skin capacitance matrix using Multipole
    
    matC_skin = MakeCmn_skin(gamma_skin,R,c,k0,N_multi);
    
    GCM_skin = diag(delta.*v2./A_norm')*matC_skin;

    [evec_skin,eval_skin,eigen_left] = eig(GCM_skin);
    counter = 0;
    for j = 1:N
        if norm(evec_skin(1:floor(0.2*N),j))>0.8
            counter = counter + 1;
        end
    end
    proportions =  [proportions counter/N];
    N
end
plot([0:0.5:4],proportions,'k','LineWidth',2)
box off
xlim([0,4])
xlabel('Strength of the gauge potential \gamma','FontSize',18)
ylabel('Proportion','FontSize',18)
set(gca,'XTick',0:0.5:4)
set(gca,'TickLabelInterpreter','latex','FontSize',18)
