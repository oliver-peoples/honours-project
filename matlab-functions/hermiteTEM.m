function h_m = hermiteTEM(x, w, m)
    switch m
    case 0
        h_m = 1;
    case 1
        h_m = (2 * sqrt(2) * x) / w;
    case 2
        h_m = 4 * (sqrt(2) * x / w)^2 - 2;
    case 3
        h_m = 8 * (sqrt(2) * x / w)^3 - 12 * (sqrt(2) * x / w);
    end
end