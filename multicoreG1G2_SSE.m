function sse = multicoreG1G2_SSE(x_vec, cores, g2_capable_idx, core_psf, g1_meas, g2_meas)

emitter_xy = [x_vec(1:2),0;x_vec(3:4),0;];

emitter_brightness = [1;x_vec(5)];

emitter_distances = zeros(length(cores),2);

for emitter_idx=1:2
    diffv = cores - emitter_xy(emitter_idx,:);
    
    emitter_distances(:,emitter_idx) = sqrt(diffv(:,1).^2+diffv(:,2).^2+diffv(:,3).^2);
end

powers = exp(-(emitter_distances.^2/2)/(2*core_psf^2));

powers(:,1) = powers(:,1) * emitter_brightness(1);
powers(:,2) = powers(:,2) * emitter_brightness(2);

g1_pred = (powers(:,1)+powers(:,2))./(emitter_brightness(1)+emitter_brightness(2));

alpha = powers(g2_capable_idx,1)./powers(g2_capable_idx,2);

g2_pred = (2 * alpha)./((1+alpha).^2);

sse = sum((g1_pred - g1_meas).^2) + sum((g2_pred - g2_meas).^2);

end