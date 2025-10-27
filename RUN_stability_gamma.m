N = 50;
d =1;
cx = d*[0:1:N-1]';
cy = zeros(N,1);
cz = zeros(N,1);
c = [cx cy cz];

N_stab = 100;
epsilon = 0.1;


R = 0.3*ones(N,1)';

vol = 4*pi*R.^3/3;
k0 = 0.0000001;
delta = 10^(-5);
v2 = ones(1,N);
N_multi = 2;

%%% Compute the static capacitance matrix
% Maximum order for multipole expansion (n = 0, 1, ..., N_multi)
% If we use higher order, then accuracy improves. Usually 0 is sufficiently large.
gamma_skin = 1;

% compute the normalization factor


mean_loc = zeros(N,N_stab+1);



% compute without perturb

int_A_temp = zeros(N,1);
    for j = 1:N
        fun = @(theta,phi,r) exp(gamma_skin*r*sin(theta)*cos(phi))*r^2*sin(theta);
        int_A_temp(j) = int_trapez_3(fun,200,0,pi,0,2*pi,0,R(1));
    end

    A_norm = diag(exp(gamma_skin*ones(N,1).*(cx)).*int_A_temp);
    matC_skin =  MakeCmn_skin_gammas(gamma_skin*ones(N,1),R,c,k0,N_multi);
    GCM_skin = diag(delta.*v2)*matC_skin\A_norm;
    %%% Compute eigenmodes
    [evec_skin,eval_skin] = eig(GCM_skin);
    mean_loc(:,end) = produce_condensation_loc(cx,evec_skin,N);

for i = 1:N_stab
    gammas = gamma_skin*ones(N,1).*(1+(-epsilon+ 2*epsilon*rand(N,1)));

    int_A_temp = zeros(N,1);
    for j = 1:N
        fun = @(theta,phi,r) exp(gammas(j)*r*sin(theta)*cos(phi))*r^2*sin(theta);
        int_A_temp(j) = int_trapez_3(fun,200,0,pi,0,2*pi,0,R(1));
    end

    A_norm = diag(exp(gammas.*(cx)).*int_A_temp);
    matC_skin =  MakeCmn_skin_gammas(gammas,R,c,k0,N_multi);
    GCM_skin = diag(delta.*v2)*matC_skin\A_norm;
    %%% Compute eigenmodes
    [evec_skin,eval_skin] = eig(GCM_skin);
    mean_loc(:,i) = produce_condensation_loc(cx,evec_skin,N);
    i
end


figure
hold on
plot(cx,mean_loc(:,end),'--r','linewidth', 3)
for i = 1:N_stab
    plot(cx,mean_loc(:,i),'color', [.5 .5 .5], 'linewidth', 1.5)
end
legend('Without perturbation','With random perturbations','Location','southeast','FontSize',18)
title('Degree of condensation')
xlim([0,cx(end)])
xlabel('Position of the resonators','FontSize',18)
ylabel('Degree of condensation','FontSize',18)
set(gca,'XTick',0:0.5:4)
set(gca,'TickLabelInterpreter','latex','FontSize',18)




function matC = MakeCmn_skin_gammas(gammas,R,c,k0,N_multi)

N = size(c,1);
cx = c(:,1);
N_block = lin_ind(N_multi,N_multi); 
M = N*N_block;
fun2 = @(theta,phi) 1;
f2_out = compute_harmonics(fun2,N_multi);


matS = MakeS_mn(R,c,k0,N_multi);
[L,U,P] = lu(matS); % Solve the linear systems by LU-factorization
matC = zeros(N,N);
psis = zeros(M,N);
phis = zeros(M,N);
for j = 1:N
    phi_j = zeros(M,1);
    fun = @(theta,phi) exp(gammas(j)*R(1)*sin(theta)*cos(phi));
    f1_out = compute_harmonics(fun,N_multi);
    phi_j(N_block*(j-1)+1:N_block*j) =  R(j)^2*exp(gammas(j)*cx(j))*f1_out;
    psi_j_temp = zeros(M,1); 
    psi_j_temp(N_block*(j-1)+1:N_block*j) = f2_out;
    y = L\(P*psi_j_temp);
    psis(:,j) = U\y;
    phis(:,j) = phi_j;
end
for j = 1:N
    for i = 1:N
        matC(i,j) = -phis(:,i)'*psis(:,j);
    end
end

end