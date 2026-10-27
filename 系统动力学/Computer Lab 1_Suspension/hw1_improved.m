m1 = 465 ;
m2 = 50 ;
k1 = 5700 ;
k2 = 135000 ;
b = 290 ; 
syms s ;

u1 = [k2];
u2 = [b,k1];
Y1 = conv(u1,u2);

v1=[k2];
v2=[m1,b,k1];
Y2 = conv(v1,v2);

w1=[m2,b,k1+k2];
w2=[m1,b,k1];
w3 = [b,k1];
Y = conv(w1,w2) - [0,0,conv(w3,w3)];


G_1 = tf(Y1, Y);
disp(G_1);
G_2 = tf(Y2, Y);
disp(G_2);


% --- 仿真参数 ---
t = 0:0.01:5;           % 时间范围
u_sin = sin(2*pi*1*t);  % 1Hz 正弦路面输入

% --- 绘图部分 ---
figure('Name', 'Quarter Car Model Simulation', 'Color', 'w', 'Position', [100, 100, 1000, 600]);

% 1. k1弹簧上质量 (y1) 阶跃响应
subplot(2,2,1);
step(G_1);
title('簧上质量 (y1) 阶跃响应');
ylabel('位移');
grid on;

% 2. k1弹簧下质量 (y2) 阶跃响应
subplot(2,2,2);
step(G_2);
title('簧下质量 (y2) 阶跃响应');
ylabel('位移');
grid on;

% 3. k1弹簧上质量 (y1) 正弦响应
subplot(2,2,3);
[y1_out, ~] = lsim(G_1, u_sin, t);
plot(t, u_sin, '--k', 'LineWidth', 1); hold on;
plot(t, y1_out, 'b', 'LineWidth', 1.5);
title('簧上质量 (y1) 1Hz正弦路面响应');
xlabel('时间'); ylabel('位移');
legend('路面输入', 'y1 响应');
grid on;

% 4. k1弹簧下质量 (y2) 正弦响应
subplot(2,2,4);
[y2_out, ~] = lsim(G_2, u_sin, t);
plot(t, u_sin, '--k', 'LineWidth', 1); hold on;
plot(t, y2_out, 'r', 'LineWidth', 1.5);
title('簧下质量 (y2) 1Hz正弦路面响应');
xlabel('时间'); ylabel('位移');
legend('路面输入', 'y2 响应');
grid on;