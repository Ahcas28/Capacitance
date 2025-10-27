function f_out = compute_harmonics(fun,N_multi)
    f_out = zeros(lin_ind(N_multi,N_multi),1);
    ind = 1;
    for n = 0:N_multi
        for m = -n:n
            integrand = @(theta, phi) fun(theta,phi)*sin(theta)*conj(harmonicY(n,m,theta,phi));
            f_out(ind) = int_trapez(integrand,200,0,pi,0,2*pi);
            ind = ind+1;
        end
    end
end
