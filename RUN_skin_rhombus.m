H = 6;
N_0 = 2*27+1; % choose odd so that one resonator is in the origin
% this function gives the number of resonator on the i th line using linear
% interpolation
a = -N_0/(H-1);
compute_N_l = @(l) floor(a*(abs(l)-1)+N_0)+1;
d = 1;
h = 1;
r= 0.3;
cx = d*[0:1:N_0-1]';
cx = cx - (cx(1)+cx(end))/2;
cy = zeros(N_0,1);
%lets build a rhombus
for l = 2:H
    N_l = (compute_N_l(l))
    cx_l = d*[0:1:N_l-1]';
    cx_l = cx_l - (cx_l(1)+cx_l(end))/2;
    cx = [cx; cx_l];
    cy = [cy; h*(l-1)*ones(N_l,1)];
    % negative part
    cx = [cx; cx_l];
    cy = [cy; -h*(l-1)*ones(N_l,1)];
end
N = size(cx,1)
[cx,ind] = sort(cx);
cy = cy(ind);
cz = zeros(N,1);
c = [cx cy cz];
R = r*ones(N,1);
figure, hold on
t = linspace(0,2*pi);
for n = 1:N
    plot(c(n,1)+R(n)*cos(t), c(n,2)+R(n)*sin(t),'k')  
%     text(c(n,1),c(n,2),num2str(n))
end
daspect([1 1 1])
hold off
% close 

vol = 4*pi*R.^3/3;
k0 = 0.0000001;
delta = 10^(-5);
v2 = ones(1,N);
N_multi = 2;

%%% Compute the static capacitance matrix
% Maximum order for multipole expansion (n = 0, 1, ..., N_multi)
% If we use higher order, then accuracy improves. Usually 0 is sufficiently large.
gamma_skin = 2;

% compute the normalization factor
fun = @(theta,phi,r) exp(gamma_skin*r*sin(theta)*cos(phi))*r^2*sin(theta);
int_A = int_trapez_3(fun,200,0,pi,0,2*pi,0,R(1));
A_norm = exp(gamma_skin*cx)*int_A;

%%% Compute the skin capacitance matrix using Multipole

matC_skin = MakeCmn_skin(gamma_skin,R,c,k0,N_multi);

GCM_skin = diag(delta.*v2./A_norm')*matC_skin;

%%% Compute eigenmodes
[evec_skin,eval_skin,eigen_left] = eig(GCM_skin);
[resonances_skin,I] = sort(sqrt(diag(eval_skin)),'ComparisonMethod','real');
modes_skin = evec_skin(:,I);

produce_condensation(cx,modes_skin,N)

figure
title("skin modes")
hold on
NN = 15;
for j = 1:NN
    subplot(NN,1,j); 
%     hold on
% %     scatter(cx,cy,20,abs(real(modes_skin(:,j))),'filled')
%     colorbar
    plot(1:N,(real(modes_skin(:,j))))
end