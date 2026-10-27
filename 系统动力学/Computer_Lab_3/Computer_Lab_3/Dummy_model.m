%机械臂为六自由度机械臂
%建立机器人模型，指定初始角度
 
%关节长度
L1=0.109;L2=0.146;L3=0.052;L4=0.115;L5=0.072;Lbase = 0.035;
%% DH法建立模型,关节角，连杆偏移，连杆长度，连杆扭转角，关节类型（0转动，1移动）
L(1) = Link( 'd',L1  ,  'a',Lbase,   'alpha',-pi/2, 'offset',0   );
L(2) = Link( 'd',0   ,  'a',-L2, 'alpha',0    , 'offset',pi/2);
L(3) = Link( 'd',0   ,  'a',-L3, 'alpha',pi/2 , 'offset',0   );
L(4) = Link( 'd',L4  ,  'a',0 ,  'alpha',-pi/2, 'offset',0   );
L(5) = Link( 'd',0   ,  'a',0,   'alpha',pi/2 , 'offset',0);
L(6) = Link( 'd',L5  ,  'a',0,   'alpha',0   ,  'offset',0   );
Dummy=SerialLink(L,'name',' ');
%关节角度限制
L(1).qlim=[deg2rad(-170) deg2rad(170)];
L(2).qlim=[deg2rad(-75) deg2rad(90)];
L(3).qlim=[deg2rad(-55) deg2rad(125)];
L(4).qlim=[deg2rad(-180) deg2rad(180)];
L(5).qlim=[deg2rad(-120) deg2rad(120)];
L(6).qlim=[deg2rad(-360) deg2rad(360)];
title("Dummy");
view([1 1 1]); % 解决robot.teach()和plot的索引超出报错
Dummy.teach();%机械臂示教
set(gca,'XLim',[-0.4,0.4]);  %设置坐标轴范围
set(gca,'YLim',[-0.4,0.4]);
set(gca,'ZLim',[-0.1,0.6]);