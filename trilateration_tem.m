clear;
clc;

% functions we'll use:

%   function g1 = g1_TEM(x, y, m, n, w, w_0, I_0)
%   function h_m = hermiteTEM(x, w, m)

% the distance left, right, above, below of point center

displacement_max = 1.5;

% illumination setup

tem_1_w = 1;
inv_tem_1_w = 1 / tem_1_w;

detector_w = 1;
detector_inv_w = 1 / detector_w;

m_1 = 0;
n_1 = 0;

m_2 = 2;
n_2 = 2;

% m_3 = 1;
% n_3 = 2;

m_mat = [ m_1,m_2 ];
n_mat = [ n_1,n_2 ];

% emitter 1 setup

p_0_emitter_1 = 0.3617;
xy_emitter_1 = [ -0.6300,-0.1276 ];

normalization_factor = 2 * 3.14159 * (1/tem_1_w)^2 * 4 * 4;
normalization_factor = 1 / normalization_factor;

% emitter_1_excitation_intensity = normalization_factor * g1_TEM(xy_emitter_1, m_2, n_2, tem_1_w, 1, 1);

% emitter 2 setup

p_0_emitter_2 = 1;
xy_emitter_2 = [ -0.5146,-0.5573 ];

% emitter_2_excitation_intensity = normalization_factor * g1_TEM(xy_emitter_2, m_2, n_2, tem_1_w, 1, 1);

% plot center

xy_center = detector_inv_w * [ 0.5 * (xy_emitter_1(1) + xy_emitter_2(1)),0.5 * (xy_emitter_1(2) + xy_emitter_2(2)) ];

% do a confocal scan, for shits and giggles, because evidently I'm an idiot
% and can't read MATLAB code correctly

x_dims = 500;
y_dims = 500;

x_linspace = linspace(xy_center(1) - displacement_max * detector_inv_w, xy_center(1) + displacement_max * detector_inv_w, x_dims);
y_linspace = linspace(xy_center(2) - displacement_max * detector_inv_w, xy_center(2) + displacement_max * detector_inv_w, y_dims);

g_1_confocal_scan_intensities = zeros(1, y_dims * x_dims);
g_2_confocal_scan_intensities = zeros(1, y_dims * x_dims);

tic
parfor array_idx_1d = 1:(y_dims * x_dims)

    % convert parfor friendly one dimensional index to 2d index

    x_idx = mod(array_idx_1d - 1, x_dims) + 1;
    y_idx = floor((array_idx_1d - 1) / x_dims) + 1;

    % translate from index space to standard deviation space

    x_pos = x_linspace(x_idx);
    y_pos = y_linspace(y_idx);

    objective_pos = [ x_pos,y_pos ];

    % distances to objective
    
    r_1 = sqrt((objective_pos(1) - xy_emitter_1(1))^2 + (objective_pos(2) - xy_emitter_1(2))^2);
    r_2 = sqrt((objective_pos(1) - xy_emitter_2(1))^2 + (objective_pos(2) - xy_emitter_2(2))^2);
    
    % received power
    
    xy_emitter_1_relative = xy_emitter_1 - objective_pos;
    xy_emitter_2_relative = xy_emitter_2 - objective_pos;
    
    emitter_1_excitation_intensity = normalization_factor * g1_TEM(xy_emitter_1_relative, m_2, n_2, tem_1_w, 1, 1);
    emitter_2_excitation_intensity = normalization_factor * g1_TEM(xy_emitter_2_relative, m_2, n_2, tem_1_w, 1, 1);

    p_1 = p_0_emitter_1 * exp(-(r_1^2/2)/(2*detector_inv_w^2)) * emitter_1_excitation_intensity;
    p_2 = p_0_emitter_2 * exp(-(r_2^2/2)/(2*detector_inv_w^2)) * emitter_2_excitation_intensity;

    % g1

    g_1_confocal_scan_intensities(array_idx_1d) = (p_1 + p_2)/(p_0_emitter_1 + p_0_emitter_2);

    % g2 or hanbury-brown-twiss

    alpha = p_1 / p_2;

    g_2_confocal_scan_intensities(array_idx_1d) = (2 * alpha) / (1 + alpha)^2;
end
toc

% reshape these matrices

g_1_confocal_scan_intensities = reshape(g_1_confocal_scan_intensities, x_dims, y_dims).';
g_2_confocal_scan_intensities = reshape(g_2_confocal_scan_intensities, x_dims, y_dims).';

% plot these confocal scans

figure(1)

subplot(1,2,1)
pcolor(x_linspace, y_linspace, g_1_confocal_scan_intensities); shading interp; colorbar
hold on
plot(xy_emitter_1(1), xy_emitter_1(2), 'k+')
plot(xy_emitter_2(1), xy_emitter_2(2), 'k+')
% contour(x_linspace, y_linspace, g_2_confocal_scan_intensities, [0:0.05:0.5], 'ShowText', 'On')
axis equal tight

subplot(1,2,2)
pcolor(x_linspace, y_linspace, g_2_confocal_scan_intensities); shading interp; colorbar
hold on
plot(xy_emitter_1(1), xy_emitter_1(2), 'k+')
plot(xy_emitter_2(1), xy_emitter_2(2), 'k+')
contour(x_linspace, y_linspace, g_2_confocal_scan_intensities, [0.5], 'ShowText', 'On')
axis equal tight



