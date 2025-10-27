function out = makeM(t,delta,kappar,rhor,vol,C,rhot, sqrtkappat, w3)
%MAKE_M 
%   Output: returns function handle: coefficient of degree one equation associated to (11.4.10) in Erik's thesis
%   Input: W1, W2, W3: function handle as on p. 268 in Erik's thesis
%       delta,kappa_r,rho_r scalars
%       C capaciatance matrix (same dimensions as W1,...,W3)

%MAKE_M 
%   Output: returns function handle: coefficient of degree one equation associated to (11.4.10) in Erik's thesis (p. 268)
%   Input:  t       scalar  time variable
%           delta   scalar  High contrast parameter \delta, delta = rho_b/rho_0
%           kappar  scalar  constant bulk modulus in resonator material
%           rhor    scala
% r  constant density in resonator material
%           vol     vector  volume of resonators
%           C       matrix  capacitance matrix, UNCERTAIN: renormalized or not? 
%           rhot    function handle     (time-dependent) density inside the resonators, vector-valued (NOT matrix-valued)
%           sqrtkappat      function handle     square root of the (time-dependent) bulk modulus inside the resonators, vector-valued (NOT matrix-valued)
%           w3      funciton handle     (time-dependent) vector-valued function (NOT matrix-valued) given by (w3(t))_i = sqrt(kappa_i(t))/2 * d/dt kappa_i'(t)/kappa_i(t)^(3/2), see page 268 in Eriks thesis
%
%   Output: M(t)    matrix  M(t) = delta*kappar/deltar * W1(t) * C * W2(t) + W3(t)
%

D = diag(vol);
R = diag(rhot(t));
Rinv = diag(1./rhot(t));
K = diag(sqrtkappat(t));
W3 = diag(w3(t));
% out = delta*kappar/(vol*rhor)*K*R*C*K*Rinv + W3;
out = delta*kappar/rhor*inv(D)*K*R*C*K*Rinv + W3;

end

