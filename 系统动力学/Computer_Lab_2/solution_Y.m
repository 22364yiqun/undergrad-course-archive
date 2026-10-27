syms s M m L g I F Y Theta  % 定义符号
% 写下你的两个 Laplace 后的方程
eq1 = (M + m)*s^2*Y + m*L*s^2*Theta == F;
eq2 = m*L*s^2*Y + (I + m*L^2)*s^2*Theta - m*g*L*Theta == 0;

% 让 MATLAB 帮你解出 Theta
sol = solve([eq1, eq2], [Y, Theta]);
G = simplify(sol.Theta / F)  % 这就会直接算出最简的传递函数式子！
T = simplify(sol.Y / F)