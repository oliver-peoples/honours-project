function [g1Pred,g2Pred] = Predictg1g2_onedetector(x1,x2,x0,P01,P02,sigma)
%Predictg1g2 Summary of this function goes here
%   This program is used to determine g1 and g2 for a single detector
%   given a location for two emitters
%   x0 = [x01,y01] coordinates of  APD
%   x1 = [x1,y1] coordinate of particle 1 (optimised over)
%   x2 = [x2,y2] coordinate of particle 2 (optimised over)
%   g1 = [g11;g12;g13] measured g1 values
%   g1 = [g21;g22;g23] measured g2 values
%   P01 nominal maximum power of particle 1
%   P02 nominal maximum power of particle 2



    %Determine radii
    r1 = sqrt((x0(1) - x1(1))^2 + (x0(2) - x1(2))^2);
    r2 = sqrt((x0(1) - x2(1))^2 + (x0(2) - x2(2))^2);
    
    %Received power
    P1 = P01 * exp(-(r1.^2/2)/(2*sigma^2)); % Calculating the power for emitter 1
    P2 = P02 * exp(-(r2.^2/2)/(2*sigma^2)); % Calculating the power for emitter 2
    
    alpha = P1/P2;
    
    g1Pred = (P1 + P2)./(P01 + P02);
    g2Pred = (2*alpha)./(1+alpha).^2;




end

