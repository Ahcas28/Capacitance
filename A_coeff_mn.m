function A = A_coeff_mn(lp,mp,l,m,N_multi,z,c_coeff,Hdata)

    th = acos(z(3)/norm(z));
    sg = 2*(z(2) >= 0) -1;
    phi = sg*acos(z(1)/sqrt(z(1)^2 + z(2)^2));
    
    A = 0;
    for n = 0:N_multi
        for mu = -n:n
            if abs(th) < 1e-6                
                Ynmu = sqrt((2*n+1)/4/pi) * (abs(mu) < 1e-3);
            elseif abs(th-pi) < 1e-6
                Ynmu = (-1)^n*sqrt((2*n+1)/4/pi) * (abs(mu) < 1e-3);
            else               
                Ynmu = harmonicY(n,mu,th,phi);
            end
            A = A + Ynmu*c_coeff(lp+1,mp+N_multi+1,l+1,m+N_multi+1,n+1,mu+N_multi+1)*Hdata(n+1);
        end
    end
    
end
