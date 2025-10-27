close all

%%%==========Define parameters==========%%%
%%% Number of resonators
N = 51;

%%% Radius of resonators
rad = 0.3;

%%% Radius of circle
R = 20;

%%% Perturbation
%epsilon = 0.9/N;
eps = 0.26
i_defect = 26

%%% Lists of positions:
%dtheta = 2*pi*(1 - (1/N + epsilon)) / (N-1);
dtheta = 2*pi/(N);
theta = pi + (0:N-1)*dtheta;     % N-1 regular angles


cx = R*cos(theta)';
cy = R*sin(theta)';
cz = zeros(N,1);
c = [cx cy cz];
rd = rad*ones(N,1);
rd(i_defect) = rad + eps;

figure, hold on
t = linspace(0,2*pi);
for n = 1:N
    plot(c(n,1)+rd(n)*cos(t), c(n,2)+rd(n)*sin(t),'k')
    text(c(n,1),c(n,2),num2str(n))
end
daspect([1 1 1])
hold off
%close

vol = 4*pi*R.^3/3;
k0 = 0.0000001;
delta = 10^(-5);
v2 = ones(1,N);
N_multi = 2;

%%% Compute the static capacitance matrix
% Maximum order for multipole expansion (n = 0, 1, ..., N_multi)
% If we use higher order, then accuracy improves. Usually 0 is sufficiently large.
gamma_skin = 0.5;  % Peut être un scalaire ou un vecteur de taille N

%%% Vecteurs tangents (sens trigo)
tx = -sin(theta)';
ty =  cos(theta)';

%%% Vecteurs gamma * tangents
gx = gamma_skin * tx;
gy = gamma_skin * ty;
gz = zeros(N,1);
gammas = [gx gy gz];
%%% (Optionnel) Visualisation
%quiver(cx, cy, gx, gy, 0, 'r', 'LineWidth', 1.5)  % Vecteurs en rouge


% compute the normalization factor
disp('Computing the normalization factor');
%fun = @(theta,phi,r) exp(gamma_skin*r*sin(theta)*cos(phi))*r^2*sin(theta);
%int_A = int_trapez_3(fun,200,0,pi,0,2*pi,0,R(1));
%A_norm = exp(gamma_skin*cx)*int_A;

A_norm = zeros(N,1);
for n = 1:N
    cn = c(n,:);                    % Centre
    gamma_vec = [gx(n), gy(n), gz(n)];  % Vecteur gamma local
    norm_gamma = norm(gamma_vec);
    %fun = @(theta,phi,r) exp(gamma_vec(1)*r*sin(theta)*cos(phi) + gamma_vec(2)*r*sin(theta)*sin(phi) + gamma_vec(3)*r*cos(theta) )*r^2*sin(theta);
    int_A_analytic = -(4*pi / norm_gamma^3) * (sinh(norm_gamma * rad) - norm_gamma * rad * cosh(norm_gamma * rad))
    %int_A_n = int_trapez_3(fun, 100, 0, pi, 0, 2*pi, 0, r)
    A_norm(n) = exp(dot(gamma_vec, cn)) * int_A_analytic;
end

disp('Computing defect normalization factor');
gamma_vec_idefect = [gx(i_defect), gy(i_defect), gz(i_defect)];
cn_idefect = c(i_defect,:);
int_A_defect_analytic = -(4*pi / norm_gamma^3) * (sinh(norm_gamma * (rad + eps)) - norm_gamma * (rad + eps) * cosh(norm_gamma * (rad + eps)))
A_norm(i_defect) = exp(dot(gamma_vec_idefect, cn_idefect))*int_A_defect_analytic;


%%% Compute the skin capacitance matrix using Multipole
disp('Computing skin capacitance matrix using multipole');
matC_skin = MakeCmn_skin_circular(gammas,rd,c,k0,N_multi,i_defect);

GCM_circular = diag(delta.*v2./A_norm')*matC_skin;
save('GCM_circular_N51_R20_r03_eps026.mat', 'GCM_circular', '-hdf5');


%%% Compute eigenmodes
[evec_skin,eval_skin,eigen_left] = eig(GCM_circular);
[resonances_skin,I] = sort(sqrt(diag(eval_skin)),'ascend');
modes_skin = evec_skin(:,I);

figure
title("modes_skin")
hold on
for j = 1:N
    plot(cx,abs(real(modes_skin(:,j))))
end

