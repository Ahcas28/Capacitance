%%% Converts double indices n, m for spherical harmonics into linear
%%% indices i_nm ordered as follows:
%%% n = 0,  m = 0,  i_nm = 1
%%% n = 1,  m = -1, i_nm = 2
%%%         m = 0,  i_nm = 3
%%%         m = 1,  i_nm = 4
%%% n = 2,  m = -2, i_nm = 5
%%%         m = -1, i_nm = 6
%%%         m = 0,  i_nm = 7
%%%         m = 1,  i_nm = 8
%%%         m = 2,  i_nm = 9, ...

function out = lin_ind(n,m)
if abs(m) > n
    disp('Error: \t |m| > n'); 
    out = 0;
else
    out = n^2 + m + n + 1;
end
