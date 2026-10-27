syms s k1 k2 k3 k4
A = [s, -1, 0, 0;
     2+k1, s+0.5+k2, k3-1, k4-0.5;
     0, 0, s, -1;
     -1, -0.5, 1, s+0.5];

res = det(A);

% 核心步骤：按照 s 进行合并同类项，得到标准多项式
final_poly = collect(res, s);

pretty(final_poly)