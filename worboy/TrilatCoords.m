function [c2] = TrilatCoords(xx,g1,g2,x0,P01,sigma)
%TrilatCoords Summary of this function goes here
%   This program is used in fminsearch to find the location of two emitters
%   specified by x1 and x2, given a set of three detectors. g1 and g2 are
%   measured values of g1 and g2 and the output is the difference according
%   to the chi2 metric.
%   x0 = [x01,y01;x02,y02;x03,y03] coordinates of each APD
%   x1 = [x1,y1] coordinate of particle 1 (optimised over)
%   x2 = [x2,y2] coordinate of particle 2 (optimised over)
%   g1 = [g11;g12;g13] measured g1 values
%   g1 = [g21;g22;g23] measured g2 values
%   P01 nominal maximum power of particle 1
%   P02 nominal maximum power of particle 2

x1 = [xx(1),xx(2)];
x2 = [xx(3),xx(4)];

%P01 = 1;
P02 = xx(5);
for ct = 1:3 %indexed over all detectors
    %Determine radii
    r1(ct) = sqrt((x0(ct,1) - x1(1))^2 + (x0(ct,2) - x1(2))^2);
    r2(ct) = sqrt((x0(ct,1) - x2(1))^2 + (x0(ct,2) - x2(2))^2);
    
    %Received power
    P1(ct) = P01 * exp(-(r1(ct).^2/2)/(2*sigma^2)); % Calculating the power for emitter 1
    P2(ct) = P02 * exp(-(r2(ct).^2/2)/(2*sigma^2)); % Calculating the power for emitter 2
    
    alpha(ct) = P1(ct)/P2(ct);
    
    g1Pred(ct) = (P1(ct) + P2(ct))./(P01 + P02);
    g2Pred(ct) = (2*alpha(ct))./(1+alpha(ct)).^2;
end

% c2 = sum((g1 - g1Pred).^2) + sum((g2 - g2Pred).^2);
c2 = sum(((g1 - g1Pred).^2)./g1) + sum(((g2 - g2Pred).^2)./g2);

end

