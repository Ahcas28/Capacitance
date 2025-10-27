function matC = MakeCmn_skin_new(gamma,R,c,k0,N_multi,f1_out,f2_out)

N = size(c,1);
cx = c(:,1);
N_block = lin_ind(N_multi,N_multi); 
M = N*N_block;



%  Assume all Ris are equal
% fun = @(theta,phi) exp(gamma*R(1)*sin(theta)*cos(phi));
% f1_out = compute_harmonics(fun,N_multi);
% 
% fun2 = @(theta,phi) 1;
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