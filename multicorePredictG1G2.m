function [g1_pred, g2_pred] = multicorePredictG1G2(emitter_xy, emitter_brightness, cores, g2_capable_idx, core_psf)

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

end