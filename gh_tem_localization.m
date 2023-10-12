clear;
clc;

% tem format: m,n,w,rotation, center_x, center_y

TRIALS = 1000;

% tem_details = [0,0,1,0,0,1;
%                 0,0,1,0,-sin(60*pi/180),-cos(60*pi/180);
%                 0,0,1,0,sin(60*pi/180),-cos(60*pi/180);
%               ];

tem_details = [1,0,1,0,0,0;
                1,1,1,0,0,0;
                0,1,1,0,0,0;
                0,0,1,0,0,0;
                2,2,1,0,0,0;
              ];

num_tems = size(tem_details,1);
% 
EMITTER_XY = 0.5*[-0.6300,-0.1276;
    0.5146,-0.5573]
% 
EMITTER_BRIGHTNESS = [1.,0.3617];

detector_width = 1;

[g1_true, g2_true] = predictGH_G1G2(tem_details, EMITTER_XY, EMITTER_BRIGHTNESS, detector_width);

variab = 0.1;

chi2 = zeros(TRIALS);

%Set options for fminsearch
options = optimset('TolFun',1e-8);

x1s = zeros(TRIALS,2);
x2s = zeros(TRIALS,2);
P02s = zeros(TRIALS);
%Step over multiple experiments
for cts = 1:TRIALS
    
    randn('seed', cts);
    
    g1_meas = g1_true.*(1 + variab*randn(1,length(g1_true)).');
    g2_meas = g2_true.*(1 + variab*randn(1,length(g2_true)).');

    fun = @(xx)calcXiSquareGH_G1G2(xx, tem_details, detector_width, g1_meas, g2_meas);
    
%     fun([EMITTER_XY(1,:),EMITTER_XY(2,:),EMITTER_BRIGHTNESS(2)])
    xx0 = [rand(1,4),0.5]; %Assume that the power ratio is 0.5 as first guess
    
%     xx0 = [EMITTER_XY(1,:),EMITTER_XY(2,:),EMITTER_BRIGHTNESS(2)];
    [xx,chi2(cts)] = fminsearch(fun,xx0,options);
    
    %Record values
%    x1s(cts,1) = xx(1,1);
%    x1s(cts,2) = xx(1,2);
%    x2s(cts,1) = xx(2,1);
%    x2s(cts,2) = xx(2,2);
    if xx(5) < EMITTER_BRIGHTNESS(1)
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
for cts = 1:TRIALS
    plot(x1s(cts,1),x1s(cts,2),'c.')
    plot(x2s(cts,1),x2s(cts,2),'m.')
end

plot(tem_details(:,5), tem_details(:,6), 'ko')

% plot(EMITTER_XY(1,1),EMITTER_XY(1,2),'k+')
% plot(EMITTER_XY(2,1),EMITTER_XY(2,2),'k+')
% for ct = 1:2
%    plot(xx(ct,1),xx(ct,2),'r+')
% end
axis([-2.5 2.5 -2.5 2.5])