close all

%%%==========Define parameters==========%%%
%%% Number of resonators
% Horizontally
N_X = 51;

% Vertically
N_Y = 1;

% Total
N = N_X*N_Y;

% Unit cell
d = 1;

% Defect parameters
eps = 0.045
x_defect = 25;
y_defect = 1;
i_defect = x_defect * N_Y + y_defect;
% Radius
radius = 0.25;
Radius = radius*ones(N,1);

% X-Positions
cx_0 = d *[0:1:N_X-1]';
cx =[];
for i = 1:N_Y
    cx = [cx; cx_0];
end

% Y-Positions
cy = [];
for i = 1:N_Y
    cy = [cy; (i-1)*d*ones(N_X,1)];
end

% Z-Positions
cz = zeros(N,1);

% XYZ-Positions
[cx,ind] = sort(cx);
cy = cy(ind);
cz = cz(ind);
c = [cx cy cz];

% Induce defect
Radius(i_defect) = Radius(i_defect) + eps;

figure, hold on
t = linspace(0,2*pi);
for n = 1:N
    plot(c(n,1)+Radius(n)*cos(t), c(n,2)+Radius(n)*sin(t),'k')
%     text(c(n,1),c(n,2),num2str(n))
end
daspect([1 1 1])
xlim([min(cx)-1,max(cx)+1])
%hold off
% close

%vol = 4*pi*R.^3/3;
% small wave-number,
% contrast parameter,
% speeds,
% order of multipole expansion
k0 = 0.0000001;
delta = 10^(-5);
v2 = ones(1,N);
N_multi = 2;

disp('Computing skin capacitance matrix');
gamma_skin = 1.0;

% compute the normalization factor
fun = @(theta,phi,r) exp(gamma_skin*r*sin(theta)*cos(phi))*r^2*sin(theta);
disp('Computing damping exponential R');
%int_A = int_trapez_3(fun,200,0,pi,0,2*pi,0,Radius(1));
int_A_analytic = -(4*pi / gamma_skin^3) * (sinh(gamma_skin * radius) - gamma_skin * radius * cosh(gamma_skin * radius))

disp('Computing damping exponential Rd');
%int_A_defect = int_trapez_3(fun,200,0,pi,0,2*pi,0,Radius(i_defect));
int_A__defect_analytic = -(4*pi / gamma_skin^3) * (sinh(gamma_skin * (radius+eps)) - gamma_skin * (radius+eps) * cosh(gamma_skin * (radius+eps)))

A_norm = exp(gamma_skin*cx)*int_A_analytic;
A_norm(i_defect) = exp(gamma_skin*cx(i_defect))*int_A__defect_analytic;

%%% Compute the skin capacitance matrix using Multipole
disp('Computing skin capacitance matrix using multipole');
matC_skin = MakeCmn_skin_defect(gamma_skin,Radius,c,k0,N_multi,i_defect);

GCM_skin = diag(delta.*v2./A_norm')*matC_skin;

disp('Computing modes and frequencies');
[evec_skin,eval_skin,eigen_left] = eig(GCM_skin);
values_skin = sqrt(diag(eval_skin));
values_skin = real(values_skin);  % ou abs(values) selon le cas

[resonances_skin,I] = sort(values_skin,'ascend');
modes_skin = evec_skin(:,I);

%save('modes_skin_N_101_R02_eps_m0003.mat', 'modes_skin', '-hdf5');
%save('eval_skin_N_101_R02_eps_m0003.mat', 'eval_skin', '-hdf5');
save('GCM_skin_N_51_R025_eps_0045.mat', 'GCM_skin', '-hdf5');





%produce_condensation(cx,modes_skin,N)

figure
title("modes_skin")
hold on
for j = 1:N
    plot(cx,abs(real(modes_skin(:,j))))
end
%colorbar




























