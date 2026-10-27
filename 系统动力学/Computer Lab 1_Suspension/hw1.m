m1 = 465 ;
m2 = 50 ;
k1 = 5700 ;
k2 = 135000 ;
b = 290 ; 
syms s 

Y1 = -k2*(b * s + k1); % problem
Y2 = -k2 * (m1 *s^2 +b * s + k1);
Y = (m2*s^2+b*s+k1+k2)* (m1 * s^2 + b * s + k1) - (b * s + k1 )^2;

G_1 = Y1 / Y;
G_2 = Y2 / Y;

[num_poly_1, den_poly_1] = numden(G_1);
disp('Y_1 :')
pretty(num_poly_1)
disp('Y :')
pretty(den_poly_1)

num_coef_1 = double(coeffs(num_poly_1, s, 'All'));
den_coef_1 = double(coeffs(den_poly_1, s, 'All'));

den_coef_1 = den_coef_1 / den_coef_1(1);
num_coef_1 = num_coef_1 / den_coef_1(1); 

G_1 = tf(num_coef_1, den_coef_1);

[num_poly_2, den_poly_2] = numden(G_2);
disp('Y_2 :')
pretty(num_poly_2)

% 转换为数值系数
num_coef_2 = double(coeffs(num_poly_2, s, 'All'));
den_coef_2 = double(coeffs(den_poly_2, s, 'All'));

den_coef_2 = den_coef_2 / den_coef_2(1);
num_coef_2 = num_coef_2 / den_coef_2(1);  

G_2 = tf(num_coef_2, den_coef_2);


t = 0:0.01:5;          % 时间范围
u_sin = sin(2*pi*1*t);  % 正弦输入 (1Hz)

figure('Name', 'Quarter Car Simulation', 'Color', 'w');

% --- 阶跃响应 ---
subplot(2,2,1);
step(G_1);
title('簧上质量 (y1) 阶跃响应');
grid on;

subplot(2,2,2);
step(G_2);
title('簧下质量 (y2) 阶跃响应');
grid on;

% --- 正弦响应 ---
subplot(2,2,3);
lsim(G_1, u_sin, t);
hold on; plot(t, u_sin, '--r');
title('簧上质量 (y1) 正弦响应');
legend('输出 y1', '路面输入 y');
grid on;

subplot(2,2,4);
lsim(G_2, u_sin, t);
hold on; plot(t, u_sin, '--r'); 
title('簧下质量 (y2) 正弦响应');
legend('输出 y2', '路面输入 y');
grid on;
