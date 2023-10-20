%Callg1g2_ForPRA.m
%
%This script calls the program to calculate g1 and g2 and then predict the
%starting positions

%Modified for variable confidence interval 11/2/19
%This version modified to make the Fig. 3 for the PRA version
clear

%Repeat multiple times to get a feel for the effect of random fluctuations
NumSamples = 501;

%Fraction of population used to define the confidence range - here
%population corresponding to one standard deviation
ConfFrac = 1 - 1/sqrt(exp(1));

%Position of emitters
x1 = [2*rand-1,2*rand-1];
x2 = [2*rand-1, 2*rand-1];

%This is a strange set
%x1 = [-0.5150,    0.8734];
%x2 = [0.7204,   -0.2055];

%Strange set 2
%x1 = [-0.0470,   -0.6462];
%x2 = [0.0058    0.4619];

%Position of detectors
x0 = 1.0*[0,1;
    -sin(60*pi/180),-cos(60*pi/180);
    sin(60*pi/180),-cos(60*pi/180)];

sigma = 1; %sigma = 0.21 * \ambda/NA - standard deviation of PSF
%Powers
P01 = 1;
%P02 = 0.5;
%P02 = 0.8;

t = 10000;

%Strange set
%P02 = 0.4794;
%Strange set 2
%P02 = 0.8819;

%Data for figure
P02 = 0.3617;
x1 = [-0.6300,   -0.1276];
x2 = [0.5146,   -0.5573];
%variab = 0.01;
% P01 = 0.3617;
% P02 = 0;
% x2 = [-0.6300,   -0.1276];
% x1 = [0.5146,   -0.5573];

%Data for close emitters
% P02 = 0.3617
% x1 = [-0,   0.1276];
% x2 = [0.,   0.13];


[g1m,g2m] = discretePredictg1g2(x1,x2,x0,P01,P02,sigma,t);
%Step over x1 and x2 values, for simplicty assume x1 in the correct
%position, but step over x2

%Now calculate confocal and g2 scan
NumXpoint = 101;
NumYpoint = 99;
xax = linspace(-1.5,1.5,NumXpoint);
yax = linspace(-1.5,1.5,NumYpoint);
for ctx = 1:NumXpoint
    for cty = 1:NumYpoint
        [g1Map(ctx,cty),g2Map(ctx,cty)] = Predictg1g2_onedetector(x1,x2,[xax(ctx),yax(cty)],P01,P02,sigma);
    end
end

%We expect noise in the measured values of g1 and g2, let us assume at the
%derived from a gaussian noise source
variab = 0.0;

%Set options for fminsearch
options = optimset('TolFun',1e-8);


%Step over multiple experiments
for cts = 1:NumSamples
    
    randn('seed',cts);
    
    g1n = g1m.*(1 + variab*randn(1,3));
    g2n = g2m.*(1 + variab*randn(1,3));

%numy = 101;
%y2vals = linspace(-2,2,numy);
    %fun = @(xx)TrilatCoords(xx,g1n,g2n,x0,P01,P02,sigma);
    %Code modified so that P02 unknown
    fun = @(xx)discreteTrilatCoords(xx,g1n,g2n,x0,P01,sigma,t);
%for ct = 1:numy
%    x2test = [0.5,y2vals(ct)];
%    xx = [x1(1), x1(2); x2(1),y2vals(ct)];
    %[c2(ct)] = TrilatCoords(xx,g1m,g2m,x0,P01,P02);
%end
    %xx0 = [x1(1), x1(2); x2(1),1];
    %xx0 = [0,0,0,0];%0.5*rand(1,4);%[0,0.5,0,0.5];
    xx0 = [rand(1,4),0.5]; %Assume that the power ratio is 0.5 as first guess
    %xx0 = [0, 0.5, 0, 0.5];
    [xx,chi2(cts)] = fminsearch(fun,xx0,options);
    
    %Record values
%    x1s(cts,1) = xx(1,1);
%    x1s(cts,2) = xx(1,2);
%    x2s(cts,1) = xx(2,1);
%    x2s(cts,2) = xx(2,2);
    if xx(5) < P01
        x1s(cts,1) = xx(1);
        x1s(cts,2) = xx(2);
        x2s(cts,1) = xx(3);
        x2s(cts,2) = xx(4);
        P02s(cts) = xx(5);
    else
        x1s(cts,1) = xx(3);
        x1s(cts,2) = xx(4);
        x2s(cts,1) = xx(1);
        x2s(cts,2) = xx(2);
        P02s(cts) = 1/xx(5);
    end
%    P02s(cts) = xx(5);
end





%Make output
figure(1)
clf

subplot(2,2,1)
hold on
for cts = 1:NumSamples
    plot(x1s(cts,1),x1s(cts,2),'g.')
    plot(x2s(cts,1),x2s(cts,2),'r.')
end
for ct = 1:3
    plot(x0(ct,1),x0(ct,2),'ko')
end
plot(x1(1),x1(2),'k+')
plot(x2(1),x2(2),'k+')
%for ct = 1:2
%    plot(xx(ct,1),xx(ct,2),'r+')
%end
axis([-1.5 1.5 -1.5 1.5])

subplot(2,2,2)
hold on
for ct = 1:3
    plot(x0(ct,1),x0(ct,2),'ko')
end

%Now calculate COnfFrac confidence interval
rr1 = sqrt((x1s(:,1) - mean(x1s(:,1))).^2 + (x1s(:,2) - mean(x1s(:,2))).^2);
RTab1 = [x1s,rr1];
SortTab1 = sortrows(RTab1,3);
FractionBoundary = ceil(ConfFrac*NumSamples);
xx1 = SortTab1(1:FractionBoundary,1);
yy1 = SortTab1(1:FractionBoundary,2);
k1 = boundary(SortTab1(1:FractionBoundary,1),SortTab1(1:FractionBoundary,2));
%Now determine area
Area1 = polyarea(xx1(k1),yy1(k1));
Weff1 = 2*sqrt(Area1/pi);
plot(mean(x1s(:,1)),mean(x1s(:,2)),'k+')
plot(xx1(k1),yy1(k1),'g-')

rr2 = sqrt((x2s(:,1) - mean(x2s(:,1))).^2 + (x2s(:,2) - mean(x2s(:,2))).^2);
RTab2 = [x2s,rr2];
SortTab2 = sortrows(RTab2,3);
FractionBoundary = ceil(ConfFrac*NumSamples);
xx2 = SortTab2(1:FractionBoundary,1);
yy2 = SortTab2(1:FractionBoundary,2);
k2 = boundary(SortTab2(1:FractionBoundary,1),SortTab2(1:FractionBoundary,2));
%Now determine area
Area2 = polyarea(xx2(k2),yy2(k2));
Weff2 = 2*sqrt(Area2/pi);
plot(mean(x2s(:,1)),mean(x2s(:,2)),'k+')
plot(xx2(k2),yy2(k2),'r-')
axis([-1.5 1.5 -1.5 1.5])


subplot(2,2,3)
pcolor(xax,yax,g1Map'); shading interp; colorbar
hold on
plot(x1(1),x1(2),'k+')
plot(x2(1),x2(2),'k+')
contour(xax,yax,g2Map',[0:0.05:0.5],'ShowText','On')

subplot(2,2,4)
% pcolor(xax,yax,g2Map'); shading interp; colorbar
% hold on
% plot(x1(1),x1(2),'k+')
% plot(x2(1),x2(2),'k+')


noisemodelCF = 1 + variab*randn(size(g1Map));
g1MapNoise = g1Map.*noisemodelCF;
noisemodelCF = 1 + variab*randn(size(g2Map));
g2MapNoise = g2Map.*noisemodelCF;

pcolor(xax,yax,g1MapNoise'); shading interp; colorbar
hold on
plot(x1(1),x1(2),'k+')
plot(x2(1),x2(2),'k+')
contour(xax,yax,g2MapNoise',[0:0.05:0.5],'ShowText','On')

figure(2)
clf
hold on
pcolor(xax,yax,g1Map'); shading interp; colorbar
for cts = 1:NumSamples
    plot(x1s(cts,1),x1s(cts,2),'g.')
    plot(x2s(cts,1),x2s(cts,2),'r.')
end
for ct = 1:3
    plot(x0(ct,1),x0(ct,2),'ko')
end
plot(x1(1),x1(2),'k+')
plot(x2(1),x2(2),'k+')

plot(xx1(k1),yy1(k1),'k-')
plot(xx2(k2),yy2(k2),'k-')

%contour(xax,yax,g2Map',[0:0.05:0.5],'ShowText','On')

%contour(xax,yax,g1Map');

%for ct = 1:2
%    plot(xx(ct,1),xx(ct,2),'r+')
%end
axis([-1.5 1.5 -1.5 1.5])
% plot((mean(x1s),1),(mean(x1s),2),'g.')
% plot(mean(x2s,1),mean(x2s,2),'r.')
% 
% plot(x1(1),x1(2),'k+')
% plot(x2(1),x2(2),'k+')
% axis([-1.5 1.5 -1.5 1.5])

%Code for Fig 3a
pcolor(xax,yax,(P01+P02)*g1Map'); shading interp; colorbar
axis([-1.5 1.5 -1.5 1.5])
hold on
plot(x1(1),x1(2),'k+')
plot(x2(1),x2(2),'k+')

%Code for Fig 3c
clf
hold on
for ct = 1:3
    plot(x0(ct,1),x0(ct,2),'ko','LineWidth',2)
end

for cts = 1:NumSamples
    plot(x1s(cts,1),x1s(cts,2),'.','Color',[135,206,250]/255)
    plot(x2s(cts,1),x2s(cts,2),'.','Color',[255,140,0]/255)
end
plot(xx1(k1),yy1(k1),'b-','LineWidth',2)
plot(xx1(k1),yy1(k1),'k:','LineWidth',2)
plot(xx2(k2),yy2(k2),'-','Color',[255,140,0]/255,'LineWidth',2)
plot(xx2(k2),yy2(k2),'k:','LineWidth',2)
plot(x1(1),x1(2),'k+','LineWidth',2)
plot(x2(1),x2(2),'k+','LineWidth',2)

%Output parameters on screen
fprintf('x_1 = [ %1.4f , %1.4f ] \n',x1(1),x1(2))
fprintf('x_2 = [ %1.4f , %1.4f ] \n',x2(1),x2(2))
fprintf('alpha = [ %1.4f ] \n',P02)
fprintf('inferred values \n')
fprintf('x_1s =  [ %1.4f , %1.4f ] \n',mean(x1s(1)),mean(x1s(2)))
fprintf('x_2s =  [ %1.4f , %1.4f ] \n',mean(x2s(1)),mean(x2s(2)))
fprintf('alpha = [ %1.4f ] \n',mean(P02s))
fprintf('%1.4f confidence width particle 1 %1.4f \n',ConfFrac,2*SortTab1(FractionBoundary,3))
fprintf('%1.4f confidence width particle 2 %1.4f \n',ConfFrac,2*SortTab2(FractionBoundary,3))