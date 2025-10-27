function matC = MakeC_mn(R,c,k0,N_multi)

N = size(c,1);
cz = c(:,3);
N_block = lin_ind(N_multi,N_multi); 
M = N*N_block;

matS = MakeS_mn(R,c,k0,N_multi);
[L,U,P] = lu(matS); % Solve the linear systems by LU-factorization
matC = zeros(N,N);
for j = 1:N
    phi_j = zeros(M,1); 
    phi_j(N_block*(j-1)+1) = 1;
    y = L\(P*phi_j);
    psi_j = U\y;
    for i = 1:N
        if length(R) > 1
            Ri = R(i);
        else
            Ri = R;
        end
        matC(i,j) = -4*pi*Ri^2*psi_j(N_block*(i-1)+1); % psi_j(N_block*(i-1)+1) = phi_i.'*matS\phi_j, psi_j = matS\phi_j
    end
end

end