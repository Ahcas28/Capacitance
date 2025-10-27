function Y = harmonicY(n,m,th,phi)
%HARMONICY  Spherical harmonic function.
%
%   Y = HARMONICY(N,M,TH,PHI) computes the surface spherical harmonic of
%   degree N and order M, evaluated at each element of inclination TH and
%   azimuth PHI. N and M must be scalar integers where M <= abs(N). TH and
%   PHI must be arrays of equal size.
%

% check if m is odd and negative
isoddm = mod(m,2) == 1;
isnegm = m < 0;

% if m is negative, compute the symmetric function where m > 0
if isnegm
    m = abs(m);
end

% normalization factor
a = (2*n+1)/(4*pi);
b = exp(gammaln(n-m+1) - gammaln(n+m+1));
C = sqrt(a*b);

% associated Legendre function
P = legendre(n,cos(th));
P = P(m+1,:)';

E = exp(1i*m*phi);
        
% surface spherical harmonics
Y = C * P .* E;

% if m was negative
if isnegm
    Y = conj(Y);
    if isoddm
        Y = -Y;
    end
end

end