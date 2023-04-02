function h_m = hermiteTEM(x, w, m)
    switch m
    case 0
        h_m = 1;
    case 1
        h_m = (2 * sqrt(2) * x) / w;
    case 2
        h_m = (-2 * w^2 + 8 * x^2) / w^2;
    case 3
        h_m = (-12 * w^2 * x + 16 * x^3 * sqrt(2)) / w^3;
    end
end