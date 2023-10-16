function sse = calcXiSquareGH_G1G2(x_vec, tem_details, detector_w, g1_meas, g2_meas)

sqrt_2 = sqrt(2);

emitter_xy = [x_vec(1:2);x_vec(3:4)];
emitter_brightness = [1,x_vec(5)];

num_tems = size(tem_details,1);

illumination_centers = tem_details(:,5:6);

illumination_center_disp = zeros(num_tems,4);

illumination_center_disp(:,1:2) = emitter_xy(1,:) - illumination_centers;
illumination_center_disp(:,3:4) = emitter_xy(2,:) - illumination_centers;

m = tem_details(:,1);
n = tem_details(:,2);

w = tem_details(:,3);

g1_pred = zeros(num_tems,1);
g2_pred = zeros(num_tems,1);

for tem_idx=1:size(tem_details,1)
    h_m_x = fastGH(m(tem_idx), sqrt_2 * illumination_center_disp(tem_idx,[1,3]) / w(tem_idx));
    h_n_y = fastGH(n(tem_idx), sqrt_2 * illumination_center_disp(tem_idx,[2,4]) / w(tem_idx));
    
    exp_x = exp(-illumination_center_disp(tem_idx,[1,3]).^2/w(tem_idx)^2);
    exp_y = exp(-illumination_center_disp(tem_idx,[2,4]).^2/w(tem_idx)^2);
    
    I_mn = (h_m_x .* exp_x).^2 .* (h_n_y .* exp_y).^2;
    
    emitter_distance = sqrt(emitter_xy(:,1).^2 + emitter_xy(:,2).^2).';
    
    emission_power = emitter_brightness .* I_mn .* exp(-(emitter_distance.^2/2)./(2*w(tem_idx)^2));
    
    g1_pred(tem_idx) = (emission_power(1)+emission_power(2))/(emitter_brightness(1)+emitter_brightness(2));
    
    alpha = emission_power(1)/emission_power(2);
    
    g2_pred(tem_idx) = (2 * alpha)/((1+alpha)^2);
end

sse = sum((g1_pred - g1_meas).^2) + sum((g2_pred - g2_meas).^2);

end