%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
%% Define parameters

N_L = 25;
L = 1;
N = N_L*L;
d = 1;
R = 0.3*ones(N,1);
cx_0 = d *[0:1:N_L-1]';
cx =[];
for i = 1:L
    cx = [cx; cx_0];
end
cy = [];
for i = 1:L
    cy = [cy; (i-1)*d*ones(N_L,1)];
end

cz = zeros(N,1);
[cx,ind] = sort(cx);
cy = cy(ind);
cz = cz(ind);
c = [cx cy cz];

figure, hold on
t = linspace(0,2*pi);
for n = 1:N
    plot(c(n,1)+R(n)*cos(t), c(n,2)+R(n)*sin(t),'k')
%     text(c(n,1),c(n,2),num2str(n))
end
daspect([1 1 1])
xlim([min(cx)-1,max(cx)+1])
%hold off
% close

vol = 4*pi*R.^3/3;
k0 = 0.0000001;
delta = 10^(-5);
v2 = ones(1,N);
N_multi = 2;

%%% Compute the static capacitance matrix
% Maximum order for multipole expansion (n = 0, 1, ..., N_multi)
% If we use higher order, then accuracy improves. Usually 0 is sufficiently large.
disp('Computing skin capacitance matrix');
gamma_skin = 1;

% compute the normalization factor
fun = @(theta,phi,r) exp(gamma_skin*r*sin(theta)*cos(phi))*r^2*sin(theta);
int_A = int_trapez_3(fun,200,0,pi,0,2*pi,0,R(1));
A_norm = exp(gamma_skin*cx)*int_A;

%%% Compute the skin capacitance matrix using Multipole
matC_skin = MakeCmn_skin(gamma_skin,R,c,k0,N_multi);

GCM_skin = diag(delta.*v2./A_norm')*matC_skin;

%%% Compute eigenmodes
[evec_skin,eval_skin,eigen_left] = eig(GCM_skin);
values_skin = sqrt(diag(eval_skin));
values_skin = real(values_skin);  % ou abs(values) selon le cas

[resonances_skin,I] = sort(values_skin,'ascend');
modes_skin = evec_skin(:,I);

%produce_condensation(cx,modes_skin,N)

figure
title("modes_skin")
hold on
for j = 1:N
    plot(1:N,abs(real(modes_skin(:,j))))
end
colorbar
