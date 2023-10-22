function [g1Pred,g2Pred,c1,c2,c12] = discretePredictg1g2(x1,x2,x0,P01,P02,sigma,t)
%Predictg1g2 Summary of this function goes here
%   This program is used to determine g1 and g2 for a set of 3 detectors
%   given a location for two emitters
%   x0 = [x01,y01;x02,y02;x03,y03] coordinates of each APD
%   x1 = [x1,y1] coordinate of particle 1 (optimised over)
%   x2 = [x2,y2] coordinate of particle 2 (optimised over)
%   g1 = [g11;g12;g13] measured g1 values
%   g1 = [g21;g22;g23] measured g2 values
%   P01 nominal maximum power of particle 1
%   P02 nominal maximum power of particle 2
%   c1 - number of counts from particle one
%   c2 number of counts from particle two (NB not distinguished)
%   c12 - number of coincidences at zero time delay
%   t = length of time
% This function modified to perform the calculation in terms of the number
% of counts, also returns the number of counts obtained
% 23/4/2019


for ct = 1:3 %indexed over all detectors
    %Determine radii
    r1(ct) = sqrt((x0(ct,1) - x1(1))^2 + (x0(ct,2) - x1(2))^2);
    r2(ct) = sqrt((x0(ct,1) - x2(1))^2 + (x0(ct,2) - x2(2))^2);
    
    %Received power (ideal limit)
    P1(ct) = P01 * exp(-(r1(ct).^2/2)/(2*sigma^2)); % Calculating the power for emitter 1
    P2(ct) = P02 * exp(-(r2(ct).^2/2)/(2*sigma^2)); % Calculating the power for emitter 2
    
%     Recieved counts (assuming below saturation)
    c1(ct) = poissrnd(P1(ct)*t);
    c11(ct) = poissrnd(P1(ct)^2 * t);
    c2(ct) = poissrnd(P2(ct)*t);
    c22(ct) = poissrnd(P2(ct)^2 * t);
    c12(ct) = poissrnd(P1(ct)*P2(ct)*t);
    
%     alpha(ct) = c1(ct)/c2(ct);

    g1Pred(ct) = (c1(ct) + c2(ct))/t;
    g2Pred(ct) = (2*c12(ct))./(c11(ct)+2*c12(ct)+c22(ct));
    
    % g1Pred(ct) = (P1(ct) + P2(ct))./(P01 + P02);
    % g2Pred(ct) = (2*alpha(ct))./(1+alpha(ct)).^2;
end




end
