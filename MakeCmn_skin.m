function matC = MakeCmn_skin(gamma,R,c,k0,N_multi)

N = size(c,1);
cx = c(:,1);
N_block = lin_ind(N_multi,N_multi);
M = N*N_block;



%  Assume all Ris are equal
disp('f1 harmonics');
fun = @(theta,phi) exp(gamma*R(1)*sin(theta)*cos(phi));
f1_out = compute_harmonics(fun,N_multi);

disp('f2 harmonics');
fun2 = @(theta,phi) 1;
f2_out = compute_harmonics(fun2,N_multi);


matS = MakeS_mn(R,c,k0,N_multi);
disp('LU Factorization');
[L,U,P] = lu(matS); % Solve the linear systems by LU-factorization
matC = zeros(N,N);
psis = zeros(M,N);
phis = zeros(M,N);
for j = 1:N
    disp(sprintf('Compute harmonics %d', j));
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
