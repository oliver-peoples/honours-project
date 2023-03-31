function g1 = g1_TEM(emitter_xy, m, n, w, w_0, I_0)

    x = emitter_xy(1);
    y = emitter_xy(2);

    hermite_m = hermiteTEM(x, w, m);
    hermite_n = hermiteTEM(y, w, n);

    g1 = I_0 * (w_0 / w)^2 * (hermite_m * exp(-x^2 / w^2))^2 * (hermite_n * exp(-y^2 / w^2))^2;
        
end