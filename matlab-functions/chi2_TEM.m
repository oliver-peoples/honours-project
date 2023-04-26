function [c2] = chi2_TEM(xx,g_1,g_2,p_0_emitter_1,detector_inv_w,w_mat,m_mat,n_mat,nf_mat, exc_fn_rot)
objective_pos = [ 0,0 ];

X = [ xx(1:2);xx(3:4) ];

xy_emitter_1_relative = X(1,:) - objective_pos;
xy_emitter_2_relative = X(2,:) - objective_pos;

r_1 = sqrt(xy_emitter_1_relative(1)^2 + xy_emitter_1_relative(2)^2);
r_2 = sqrt(xy_emitter_2_relative(1)^2 + xy_emitter_2_relative(2)^2);

experiments = numel(nf_mat);

g_1_pred = zeros(experiments,1);
g_2_pred = zeros(experiments,1);

p_0_emitter_2 = xx(5);

for excitation_fn = 1:numel(nf_mat)
    rot_angle = exc_fn_rot(excitation_fn);

    rotated_xy_emitter_1 = [ 0 0 ];

    rotated_xy_emitter_1(1) = xy_emitter_1_relative(1) * cos(rot_angle) - xy_emitter_1_relative(2) * sin(rot_angle);
    rotated_xy_emitter_1(2) = xy_emitter_1_relative(1) * sin(rot_angle) + xy_emitter_1_relative(2) * cos(rot_angle);

    p_1 = nf_mat(excitation_fn) * g1_TEM(rotated_xy_emitter_1, m_mat(excitation_fn), n_mat(excitation_fn), w_mat(excitation_fn), 1, 1);

    rotated_xy_emitter_2 = [ 0 0 ];

    rotated_xy_emitter_2(1) = xy_emitter_2_relative(1) * cos(rot_angle) - xy_emitter_2_relative(2) * sin(rot_angle);
    rotated_xy_emitter_2(2) = xy_emitter_2_relative(1) * sin(rot_angle) + xy_emitter_2_relative(2) * cos(rot_angle);

    p_2 = nf_mat(excitation_fn) * g1_TEM(rotated_xy_emitter_2, m_mat(excitation_fn), n_mat(excitation_fn), w_mat(excitation_fn), 1, 1);
    
    p_1 = p_0_emitter_1 * exp(-(r_1^2/2)/(2*detector_inv_w^2)) * p_1;
    p_2 = p_0_emitter_2 * exp(-(r_2^2/2)/(2*detector_inv_w^2)) * p_2;
    
    alpha = p_1 / p_2;
    
    g_1_pred(excitation_fn) = (p_1 + p_2) / (p_0_emitter_1 + p_0_emitter_2);
    g_2_pred(excitation_fn) = (2 * alpha) / (1 + alpha)^2;
end

c2 = sum((g_1 - g_1_pred).^2) + sum((g_2 - g_2_pred).^2);
end