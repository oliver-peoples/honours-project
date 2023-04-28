clear;
clc;

addpath('matlab-functions/')

plot_vals = false;

s_mat = [0.5 1 4 24];

% functions we'll use:

%   function g1 = g1_TEM(x, y, m, n, w, w_0, I_0)
%   function h_m = hermiteTEM(x, w, m)

% the distance left, right, above, below of point center

displacement_max = 1.5;

% illumination setup

detector_w = 1;
detector_inv_w = 1 / detector_w;

exc_fn_rot = [ 0 ];
exc_fn_rot = 3.14159 .* exc_fn_rot / 180;

m_mat = [ 1 ];
n_mat = [ 1 ];

s_m_mat = s_mat(m_mat+1);
s_n_mat = s_mat(n_mat+1);

w_mat = [ 1 ];
inv_w_mat = 1./w_mat;

% emitter 1 setup

p_0_emitter_1 = 1;
xy_emitter_1 = [-0.6300,-0.1276];

nf_mat = 2 * 3.14159 * inv_w_mat.^2 .* s_m_mat .* s_n_mat;
nf_mat = 1 ./ nf_mat;

% emitter_1_excitation_intensity = normalization_factor * g1_TEM(xy_emitter_1, m_2, n_2, tem_1_w, 1, 1);

% emitter 2 setup

p_0_emitter_2 = 0.1;
xy_emitter_2 = [0.5146,-0.5573];

xy_diff = xy_emitter_2 - xy_emitter_1;

xy_diff = xy_diff * 1. / sqrt(xy_diff(1) * xy_diff(1) + xy_diff(2) * xy_diff(2))

xy_emitter_2 = xy_emitter_1 + xy_diff * 0.1;

p_0_emitter_1 / p_0_emitter_2

% emitter_2_excitation_intensity = normalization_factor * g1_TEM(xy_emitter_2, m_2, n_2, tem_1_w, 1, 1);

% plot center

xy_center = detector_inv_w * [ 0.5 * (xy_emitter_1(1) + xy_emitter_2(1)),0.5 * (xy_emitter_1(2) + xy_emitter_2(2)) ];
% xy_center = [0 0];

% do a confocal scan, for shits and giggles, because evidently I'm an idiot
% and can't read MATLAB code correctly

x_dims = 1500;
y_dims = 1500;

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
    
    p_1 = p_0_emitter_1;
    p_2 = p_0_emitter_2;
    
    for excitation_fn = 1:numel(nf_mat)
        rot_angle = exc_fn_rot(excitation_fn);
        
        rotated_xy_emitter_1 = [ 0 0 ];
        
        rotated_xy_emitter_1(1) = xy_emitter_1_relative(1) * cos(rot_angle) - xy_emitter_1_relative(2) * sin(rot_angle);
        rotated_xy_emitter_1(2) = xy_emitter_1_relative(1) * sin(rot_angle) + xy_emitter_1_relative(2) * cos(rot_angle);
        
        p_1 = p_1 * nf_mat(excitation_fn) * g1_TEM(rotated_xy_emitter_1, m_mat(excitation_fn), n_mat(excitation_fn), w_mat(excitation_fn), 1, 1);
        
        rotated_xy_emitter_2 = [ 0 0 ];
        
        rotated_xy_emitter_2(1) = xy_emitter_2_relative(1) * cos(rot_angle) - xy_emitter_2_relative(2) * sin(rot_angle);
        rotated_xy_emitter_2(2) = xy_emitter_2_relative(1) * sin(rot_angle) + xy_emitter_2_relative(2) * cos(rot_angle);
        
        p_2 = p_2 * nf_mat(excitation_fn) * g1_TEM(rotated_xy_emitter_2, m_mat(excitation_fn), n_mat(excitation_fn), w_mat(excitation_fn), 1, 1);
    end
    
    p_1 = exp(-(r_1^2/2)/(2*detector_w^2)) * p_1;
    p_2 = exp(-(r_2^2/2)/(2*detector_w^2)) * p_2;

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

if plot_vals
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
end

csvwrite('g_1_confocal_scan_intensities.csv', g_1_confocal_scan_intensities);
csvwrite('g_2_confocal_scan_intensities.csv', g_2_confocal_scan_intensities);
csvwrite('x_linspace.csv', x_linspace);
csvwrite('y_linspace.csv', y_linspace);
csvwrite('emitter_positions.csv', [xy_emitter_1;xy_emitter_2]);
