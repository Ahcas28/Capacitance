N = 30;
d = 1;
cx0 = d*[0:1:N-1]';
cy = zeros(N,1);
cz = zeros(N,1);

N_stab = 100;
cx_delta = zeros(N,N_stab);
epsilon = 0.1;
for i = 1:N_stab
    cx_delta(:,i) = epsilon*(-d+2*d.*rand(size(cx0))); 
end

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
fun = @(theta,phi,r) exp(gamma_skin*r*sin(theta)*cos(phi))*r^2*sin(theta);
int_A = int_trapez_3(fun,200,0,pi,0,2*pi,0,R(1));
mean_loc = zeros(N,N_stab+1);

fun = @(theta,phi) exp(gamma_skin*R(1)*sin(theta)*cos(phi));
f1_out = compute_harmonics(fun,N_multi);

fun2 = @(theta,phi) exp(-0*R(1)*sin(theta)*cos(phi));
f2_out = compute_harmonics(fun2,N_multi);

for i = 1:N_stab
    cx = cx0+cx_delta(:,i);
    A_norm = exp(gamma_skin*(cx))*int_A;
    c = [cx cy cz];
    matC_skin = MakeCmn_skin_new(gamma_skin,R,c,k0,N_multi,f1_out,f2_out);
    GCM_skin = diag(delta.*v2./A_norm')*matC_skin;
    %%% Compute eigenmodes
    [evec_skin,eval_skin] = eig(GCM_skin);
    mean_loc(:,i) = produce_condensation_loc(cx,evec_skin,N);
    i
end

cx = cx0;
A_norm = exp(gamma_skin*(cx))*int_A;
c = [cx cy cz];
matC_skin = MakeCmn_skin_position(gamma_skin,R,c,k0,N_multi,f1_out,f2_out);
GCM_skin = diag(delta.*v2./A_norm')*matC_skin;
%%% Compute eigenmodes
[evec_skin,eval_skin] = eig(GCM_skin);
mean_loc(:,end) = produce_condensation_loc(cx,evec_skin,N);

figure
hold on
plot([1:N],mean_loc(:,end),'--r','linewidth', 3)
for i = 1:N_stab
    plot([1:N],mean_loc(:,i),'color', [.5 .5 .5], 'linewidth', 1.5)
end
ylabel('Degree of condensation','FontSize',16,'interpreter','latex')
xlabel('Position of the resonators','FontSize',16,'interpreter','latex')
% legend('Without perturbation','With random perturbations','Location','southeast','interpreter','latex')
% xlim([1,30])
% set(gca,'XTick',[1,10,20,30])
set(gca,'TickLabelInterpreter','latex','FontSize',16)

set(gca,)
ylim([0,1])
function matC = MakeCmn_skin_position(gamma,R,c,k0,N_multi,f1_out,f2_out)

N = size(c,1);
cx = c(:,1);
N_block = lin_ind(N_multi,N_multi); 
M = N*N_block;



%  Assume all Ris are equal
% fun = @(theta,phi) exp(gamma*R(1)*sin(theta)*cos(phi));
% f1_out = compute_harmonics(fun,N_multi);
% 
% fun2 = @(theta,phi) exp(-0*R(1)*sin(theta)*cos(phi));
% f2_out = compute_harmonics(fun2,N_multi);


matS = MakeS_mn(R,c,k0,N_multi);
[L,U,P] = lu(matS); % Solve the linear systems by LU-factorization
matC = zeros(N,N);
psis = zeros(M,N);
phis = zeros(M,N);
for j = 1:N
    phi_j = zeros(M,1);
    phi_j(N_block*(j-1)+1:N_block*j) =  R(j)^2*exp(gamma*cx(j))*f1_out;
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
