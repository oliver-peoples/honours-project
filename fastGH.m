function f_x = fastGH(k,x)

if length(k) > 1
    error("k must be a scalar")
end

if k == 0
    f_x = ones(size(x));
    return;
elseif k == 1
    f_x = 2 * x;
    return;
elseif k == 2
    f_x = 4 * x.^2 - 2;
    return
elseif k == 3
    f_x = 8 * x.^3 - 12 * x;
elseif k == 4
    f_x = 16 * x.^4 - 48 * x.^2 + 12;
    return
end

end