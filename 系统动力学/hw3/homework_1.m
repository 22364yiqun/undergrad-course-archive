% 1. 定义系统
s = tf('s');
G = 1 / (s*(s+1)*(s+5));

% 2. 找到临界增益和频率
[Gm, Pm, Wcg, Wcp] = margin(G);
Kcr = Gm;           % 结果应为 30
Pcr = 2*pi / Wcg;   % 结果应为 2.81
disp(Kcr);
disp(Pcr);

% 3. Z-N 法则计算 PID 参数
Kp = 0.6 * Kcr;
Ti = 0.5 * Pcr;
Td = 0.125 * Pcr;

Ki = Kp / Ti;
Kd = Kp * Td;

fprintf('PID 参数结果：\nKp = %.2f\nKi = %.2f\nKd = %.2f\n', Kp, Ki, Kd);