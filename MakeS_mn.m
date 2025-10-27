%%% Computes the mutlipole discretisation of the single layer potential for
%%% a collection of N spheres at centres given by Nx3 matrix c, with radii
%%% given by R. R can be either a scalar or a vector of length N.

function S = MakeS_mn(R,c,k0,N_multi)

N = size(c,1);
Nr = length(R);
areDifferent = (Nr > 1);

Jdata_k0 = zeros(N_multi+1,Nr);
Hdata_k0 = zeros(N_multi+1,Nr);

for i = 1:Nr
    Jdata_k0(:,i) = makeBesselJdata(N_multi, k0*R(i)  );
    Hdata_k0(:,i) = makeHankel1data(N_multi, k0*R(i)  );
end

SpecialFuncDataM = zeros(N_multi+1,2*Nr);
SpecialFuncDataL = Jdata_k0;
for i = 1:Nr
    SpecialFuncDataM(:,2*i-1) = Jdata_k0(:,i);
    SpecialFuncDataM(:,2*i) = Hdata_k0(:,i);
end

%%% Precompute Wigner symbols
c_coeff = zeros(N_multi+1,2*N_multi+1,N_multi+1,2*N_multi+1,N_multi+1,2*N_multi+1);
for l1 = 0:N_multi
    for m1 = -l1:l1
        for l2 = 0:N_multi
            for m2 = -l2:l2
                for l3 = 0:N_multi
                    for m3 = -l3:l3
                        c_coeff(l1+1,m1+N_multi+1,l2+1,m2+N_multi+1,l3+1,m3+N_multi+1) = C_coeff(l1,m1,l2,m2,l3,m3);
                    end
                end
            end
        end
    end
end

N_block = lin_ind(N_multi,N_multi); 
S = zeros(N_block*N);

for i=1:N
    for j=1:N
        if i==j
            if areDifferent == 1
                dataM = SpecialFuncDataM(:,2*(i-1)+1:2*i);
                Ri = R(i);
            else
                dataM = SpecialFuncDataM;
                Ri = R;
            end
            S((i-1)*N_block+1:i*N_block,(i-1)*N_block+1:i*N_block)= makeS_diag(Ri,k0,N_multi,dataM);
        else            
            if areDifferent == 1
                dataLi = SpecialFuncDataL(:,i);
                dataLj = SpecialFuncDataL(:,j);
                Rj = R(j);
            else
                dataLi = SpecialFuncDataL;               
                dataLj = SpecialFuncDataL;
                Rj = R;
            end
            S((i-1)*N_block+1:i*N_block,(j-1)*N_block+1:j*N_block)= makeS_offdiag(c(i,:),c(j,:),Rj,k0,N_multi,dataLi,dataLj,c_coeff);
        end
    end
end

end

function M = makeS_diag(R,k0,N_multi,SpecialFuncDataM)
% make the diagonal blocks of S

Jdata_k0R=SpecialFuncDataM(:,1);
Hdata_k0R=SpecialFuncDataM(:,2);

const = -1i*k0*R^2;

N_block = lin_ind(N_multi,N_multi); 
Sk0=zeros(N_block);

for n=0:N_multi
    In = n+1;
    Jk0 = Jdata_k0R(In);
    Hk0 = Hdata_k0R(In);
    val = const*Jk0*Hk0;
    for m = -n:n
        Inm = lin_ind(n,m);
        Sk0(Inm,Inm) = val;
    end
end

M=Sk0;
end

function M = makeS_offdiag(ci,cj,Rj,k0,N_multi,SpecialFuncDataLi,SpecialFuncDataLj,c_coeff)
% make the off-diagonal blocks of S

Jdata_k0Ri=SpecialFuncDataLi;
Jdata_k0Rj=SpecialFuncDataLj;

N_block = lin_ind(N_multi,N_multi); 
Sk0=zeros(N_block);

const=-1i*k0*Rj^2;

Hdata = makeHankel1data(N_multi,k0*norm(cj-ci));
for l=0:N_multi
    Il = l+1;
    Jlk0= Jdata_k0Ri(Il);
    for m = -l:l
        Ilm = lin_ind(l,m);
        for lp=0:N_multi
            Ilp = lp+1;
            Jlpk0= Jdata_k0Rj(Ilp);
            for mp = -lp:lp
                Ilpmp = lin_ind(lp,mp);
                Sk0(Ilpmp,Ilm)=const*Jlpk0*A_coeff_mn(lp,mp,l,m,N_multi,cj-ci,c_coeff,Hdata)*Jlk0;
            end
        end
    end
end

M=Sk0;

end