function [Fx] = tyre(kappa, Fz)
% Fz - vertical load in [N]
% kappa - longitudinal slip in [%]
 
% Fxparameters - .mat file that contains parameters e.g. 'MF87FxParameters.mat'
 
%initialise parameters a1 to a11
%load(Fxparameters);
b1 = -21.3;
b2 = 1144;
b3 = 49.6;
b4 = 226;
b5 = 0.069;
b6 = -0.006;
b7 = 0.056;
b8 = 0.486;
 
%unit conversion of inputs
Fz_kN = Fz./1000;
 
%calculation of coefficients
Dx = b1 * Fz_kN^2 + b2 * Fz_kN;
Cx = 1.65;
BCDx = (b3 * Fz_kN^2 + b4 * Fz_kN)*exp(b5*Fz_kN*(-1));
Bx = BCDx / (Cx*Dx);
Ex = b6 * Fz_kN^2 + b7 * Fz_kN + b8;
 
BxPhix = Bx*(1-Ex)*kappa + Ex*atan(Bx*kappa);
 
Fx = Dx*sin(Cx*atan(BxPhix));


% B = 14;
% C = 1.6;
% D = 0.6;
% E = -0.2;
% 
% 
% Fx = D*sin(C*atan((1-E)*kappa + E/B * atan(B*kappa)));

 
end