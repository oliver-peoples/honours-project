clear;
clc;

ConfFrac = 1 - 1/sqrt(exp(1));

%==================================================================================================
% Configuration - edit me!
%==================================================================================================

CORE_OVERRIDE = false;

TRIALS = 250;

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

cores(:,1) = cores(:,1) - mean(cores(:,1));
cores(:,2) = cores(:,2) - mean(cores(:,2));

GRID_CENTER = mean(cores);

G1_ONLY_CAPABLE_IDX = 1:length(cores);

for g2_capable_idx_idx=1:length(G2_CAPABLE_IDX)
    G1_ONLY_CAPABLE_IDX(G1_ONLY_CAPABLE_IDX == G2_CAPABLE_IDX(g2_capable_idx_idx)) = [];
end

EMITTER_XY(:,1) = EMITTER_XY(:,1) - mean(EMITTER_XY(:,1));
EMITTER_XY(:,2) = EMITTER_XY(:,2) - mean(EMITTER_XY(:,2));

y_offset_linspace = linspace(2,-2,50);
x_offset_linspace = linspace(-2,2,50);

[x_offset_meshgrid, y_offset_meshgrid] = meshgrid(x_offset_linspace, y_offset_linspace);

e_1_dets = zeros(size(x_offset_meshgrid));
e_2_dets = zeros(size(x_offset_meshgrid));

parfor config_idx=1:numel(x_offset_meshgrid)
    
    disp(config_idx)
    
    thread_local_emitter_xy = EMITTER_XY;
    
    thread_local_emitter_xy(:,1) = thread_local_emitter_xy(:,1) + x_offset_meshgrid(config_idx);
    thread_local_emitter_xy(:,2) = thread_local_emitter_xy(:,2) + y_offset_meshgrid(config_idx);
    
    [g1_true, g2_true] = multicorePredictG1G2(thread_local_emitter_xy, EMITTER_BRIGHTNESS, cores, G2_CAPABLE_IDX, CORE_PSF);

    variab = 0.1;

    chi2 = zeros(TRIALS);

    %Set options for fminsearch
    options = optimset('TolFun',1e-8, 'Display', 'off');

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
    end
    
    e_1_covs = cov(x1s);
    e_2_covs = cov(x2s);
    
    e_1_det = det(e_1_covs);
    e_2_det = det(e_2_covs);
    
    e_1_dets(config_idx) = e_1_det;
    e_2_dets(config_idx) = e_2_det;
end

pcolor(x_offset_meshgrid, y_offset_meshgrid, e_1_dets); shading interp; colorbar
hold on
