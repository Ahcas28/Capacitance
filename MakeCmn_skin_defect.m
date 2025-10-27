## Copyright (C) 2025 Sacha DUPUY
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {} {@var{retval} =} MakeCmn_skin_defect (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Sacha DUPUY <sacha@MacBook-Pro-de-Sacha.local>
## Created: 2025-05-18

function matC = MakeCmn_skin_defect(gamma,R,c,k0,N_multi,defect_index)

N = size(c,1);
cx = c(:,1);
N_block = lin_ind(N_multi,N_multi);
M = N*N_block;



%  Assume all Ris are equal
disp('Computing f1 harmonics');

fun = @(theta,phi) exp(gamma*R(1)*sin(theta)*cos(phi));
f1_out = compute_harmonics(fun,N_multi);

disp('Computing f1 defect harmonics');

fun_defect = @(theta,phi) exp(gamma*R(defect_index)*sin(theta)*cos(phi));
f1_defect_out = compute_harmonics(fun_defect,N_multi);

disp('Computing f2 harmonics');

fun2 = @(theta,phi) 1;
f2_out = compute_harmonics(fun2,N_multi);

disp('LU factorization');

matS = MakeS_mn(R,c,k0,N_multi);
[L,U,P] = lu(matS); % Solve the linear systems by LU-factorization
matC = zeros(N,N);
psis = zeros(M,N);
phis = zeros(M,N);
for j = 1:N
    phi_j = zeros(M,1);
    if (j==defect_index)
       phi_j(N_block*(j-1)+1:N_block*j) =  R(j)^2*exp(gamma*cx(j))*f1_defect_out;
    else
       phi_j(N_block*(j-1)+1:N_block*j) =  R(j)^2*exp(gamma*cx(j))*f1_out;
    end
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
