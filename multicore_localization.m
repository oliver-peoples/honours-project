clear;
clc;

ConfFrac = 1 - 1/sqrt(exp(1));

%==================================================================================================
% Configuration - edit me!
%==================================================================================================

CORE_OVERRIDE = false;

TRIALS = 1000;

ROWS = 4;
COLS = 4;

CORE_PSF = 1;

G2_CAPABLE_IDX = [1,5,6];

CENTER_IDX = 6;

% G2_CAPABLE_IDX = [6,10,11];
% G2_CAPABLE_IDX = [2,3,5,7,10,11];
% G2_CAPABLE_IDX = [ 6,7,10,11 ];

X_DIFF = 1.25;

if CORE_OVERRIDE
    cores = [0,1,0;
        -sin(60*pi/180),-cos(60*pi/180),0;
        sin(60*pi/180),-cos(60*pi/180),0];
    
    G2_CAPABLE_IDX = [1,2,3];
end

EMITTER_XY = 0.25*[-0.6300,-0.1276,0;
    0.5146,-0.5573,0];

EMITTER_BRIGHTNESS = [1.,0.3617];

%==================================================================================================
% Deduced configuration values - don't touch this!
%==================================================================================================

ROW_SHIFT = X_DIFF / 2;

R_VERTEX = ROW_SHIFT * 2 / 3^0.5;

Y_DIFF = 1.5 * R_VERTEX;

col_indexes = 0:COLS - 1;
row_indexes = 0:ROWS - 1;

if ~CORE_OVERRIDE

    cores = zeros(ROWS * COLS,3);

    cores(:,3) = 0;

    for row=0:(ROWS-1)
   
        for col=0:(COLS-1)
            
            cores(row * COLS + 1:(row + 1) * COLS,1) = 0:(COLS-1);
            cores(row * COLS + 1:(row + 1) * COLS,2) = row;
        end
    end

    cores(:,1) = cores(:,1) * X_DIFF;
    cores(:,1) = cores(:,1) + mod(cores(:,2), 2) * ROW_SHIFT;
    cores(:,1) = cores(:,1) - (ROW_SHIFT + X_DIFF * (COLS - 1)) / 2;
    cores(:,2) = cores(:,2) * Y_DIFF;
    cores(:,2) = cores(:,2) - (Y_DIFF * (ROWS - 1)) / 2;
end

distances = sqrt((cores(:,1)-cores(CENTER_IDX,1)).^2 + (cores(:,2)-cores(CENTER_IDX,2)).^2);
cores = cores(distances < 2.1,:);

GRID_CENTER = mean(cores);

G1_ONLY_CAPABLE_IDX = 1:length(cores);

for g2_capable_idx_idx=1:length(G2_CAPABLE_IDX)
    G1_ONLY_CAPABLE_IDX(G1_ONLY_CAPABLE_IDX == G2_CAPABLE_IDX(g2_capable_idx_idx)) = [];
end

[g1_true, g2_true] = multicorePredictG1G2(EMITTER_XY, EMITTER_BRIGHTNESS, cores, G2_CAPABLE_IDX, CORE_PSF);

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

    

    fun = @(xx)multicoreG1G2_SSE(xx, cores, G2_CAPABLE_IDX, CORE_PSF, g1_meas, g2_meas);
    
    xx0 = [rand(1,4),0.5]; %Assume that the power ratio is 0.5 as first guess
    
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

subplot(1,2,1)
hold on
for cts = 1:TRIALS
    plot(x1s(cts,1),x1s(cts,2),'c.')
    plot(x2s(cts,1),x2s(cts,2),'m.')
end



% plot e1 covariance matrix

e_1_cov = cov(x1s);

s = -2 * log(1 - ConfFrac);

[V, D] = eig(e_1_cov * s);

t = linspace(0, 2 * pi);
a = (V * sqrt(D)) * [cos(t(:))'; sin(t(:))'];

plot(a(1, :) + mean(x1s(:,1)), a(2, :) + mean(x1s(:,2)));

% plot e2 covariance matrix

e_2_cov = cov(x2s);

s = -2 * log(1 - ConfFrac);

[V, D] = eig(e_2_cov * s);

t = linspace(0, 2 * pi);
a = (V * sqrt(D)) * [cos(t(:))'; sin(t(:))'];

plot(a(1, :) + mean(x2s(:,1)), a(2, :) + mean(x2s(:,2)));

plot(EMITTER_XY(1,1),EMITTER_XY(1,2),'k+')
plot(EMITTER_XY(2,1),EMITTER_XY(2,2),'k+')
% for ct = 1:2
%    plot(xx(ct,1),xx(ct,2),'r+')
% end
pbaspect([1 1 1]);
pm_val = 2.5;
axis([GRID_CENTER(1)-pm_val GRID_CENTER(1)+pm_val GRID_CENTER(2)-pm_val GRID_CENTER(2)+pm_val])

% exportgraphics(gca,'good_localization.png','Resolution',500);
% 
subplot(1,2,2)
hold on

for ct = 1:length(cores(G1_ONLY_CAPABLE_IDX))
    cores_subset = cores(G1_ONLY_CAPABLE_IDX,:);
    plot(cores_subset(ct,1),cores_subset(ct,2),'ko')
end

for ct = 1:length(cores(G2_CAPABLE_IDX))
    cores_subset = cores(G2_CAPABLE_IDX,:);
    plot(cores_subset(ct,1),cores_subset(ct,2),'ro')
end

pbaspect([1 1 1]);

%==================================================================================================
% Confidence fraction
%==================================================================================================

rr1 = sqrt((x1s(:,1) - mean(x1s(:,1))).^2 + (x1s(:,2) - mean(x1s(:,2))).^2);
RTab1 = [x1s,rr1];
SortTab1 = sortrows(RTab1,3);
FractionBoundary = ceil(ConfFrac*TRIALS);
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
FractionBoundary = ceil(ConfFrac*TRIALS);
xx2 = SortTab2(1:FractionBoundary,1);
yy2 = SortTab2(1:FractionBoundary,2);
k2 = boundary(SortTab2(1:FractionBoundary,1),SortTab2(1:FractionBoundary,2));
%Now determine area
Area2 = polyarea(xx2(k2),yy2(k2));
Weff2 = 2*sqrt(Area2/pi);
plot(mean(x2s(:,1)),mean(x2s(:,2)),'k+')
plot(xx2(k2),yy2(k2),'r-')
axis([GRID_CENTER(1)-pm_val GRID_CENTER(1)+pm_val GRID_CENTER(2)-pm_val GRID_CENTER(2)+pm_val])

% subplot(2,2,3)
% pcolor(xax,yax,g1Map'); shading interp; colorbar
% hold on
% plot(x1(1),x1(2),'k+')
% plot(x2(1),x2(2),'k+')
% contour(xax,yax,g2Map',[0:0.05:0.5],'ShowText','On')
% 
% subplot(2,2,4)
% % pcolor(xax,yax,g2Map'); shading interp; colorbar
% % hold on
% % plot(x1(1),x1(2),'k+')
% % plot(x2(1),x2(2),'k+')
% 
% 
% noisemodelCF = 1 + variab*randn(size(g1Map));
% g1MapNoise = g1Map.*noisemodelCF;
% noisemodelCF = 1 + variab*randn(size(g2Map));
% g2MapNoise = g2Map.*noisemodelCF;
% 
% pcolor(xax,yax,g1MapNoise'); shading interp; colorbar
% hold on
% plot(x1(1),x1(2),'k+')
% plot(x2(1),x2(2),'k+')
% contour(xax,yax,g2MapNoise',[0:0.05:0.5],'ShowText','On')
% 
% figure(2)
% clf
% hold on
% pcolor(xax,yax,g1Map'); shading interp; colorbar
% for cts = 1:NumSamples
%     plot(x1s(cts,1),x1s(cts,2),'g.')
%     plot(x2s(cts,1),x2s(cts,2),'r.')
% end
% for ct = 1:3
%     plot(x0(ct,1),x0(ct,2),'ko')
% end
% plot(x1(1),x1(2),'k+')
% plot(x2(1),x2(2),'k+')
% 
% plot(xx1(k1),yy1(k1),'k-')
% plot(xx2(k2),yy2(k2),'k-')
    
% G1_ONLY_IDX = [idx for idx in range(np.shape(cores)[0]) if idx not in G2_CAPABLE_IDX]
