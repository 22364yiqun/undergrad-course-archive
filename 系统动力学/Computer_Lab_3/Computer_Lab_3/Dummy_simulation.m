%% ===================GUI初始化程序======================%%
function varargout = Dummy_simulation(varargin)
%**************************************************************
%This functions creates a graphical user interface which contains an animation of the Cobot example.
%This function was created for the Control Lab for System Dynamics. 
%Copyright (C) 2023 by HUST                                    
%Created by Prof. Kun Bai and his grad students Yong Zeng and Guangyu Chen                                      
%**************************************************************% 
% DUMMY_SIMULATION MATLAB code for Dummy_simulation.fig
%      DUMMY_SIMULATION, by itself, creates a new DUMMY_SIMULATION or raises the existing
%      singleton*.
%
%      H = DUMMY_SIMULATION returns the handle to a new DUMMY_SIMULATION or the handle to
%      the existing singleton*.
%
%      DUMMY_SIMULATION('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DUMMY_SIMULATION.M with the given input arguments.
%
%      DUMMY_SIMULATION('Property','Value',...) creates a new DUMMY_SIMULATION or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Dummy_simulation_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Dummy_simulation_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Dummy_simulation

% Last Modified by GUIDE v2.5 17-Mar-2024 22:01:12

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Dummy_simulation_OpeningFcn, ...
                   'gui_OutputFcn',  @Dummy_simulation_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before Dummy_simulation is made visible.
function Dummy_simulation_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Dummy_simulation (see VARARGIN)

% Choose default command line output for Dummy_simulation
handles.output = hObject;
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Dummy_simulation wait for user response (see UIRESUME)
% uiwait(handles.figure1);
% uiwait(handles.figure2);
global buttonValue  Jtheta
global trip1 trip2 trip3 T1_roll T1_pitch T1_yaw ;                      %Draw Triangle
global Num T2_pn T2_psi T2_theta T2_phi T2_roll T2_pitch T2_yaw ;        %Throat Swabbing
global T3_psi T3_theta T3_phi T3_roll T3_pitch T3_yaw T3_pn;             %Cone Scan
buttonValue = 0;
Jtheta = zeros(101,6);
pause('off');
%%======================GUI初始化程序========================%%

%% =======================第一问请在此处补全代码======================%%
%%========================第一问请在此处补全代码=======================%%
%%========================第一问请在此处补全代码=======================%%
% 三角形轨迹规划函数

%已给出三个顶点trip1、trip2、trip3，补全各顶点的转换矩阵、顶点间的轨迹规划、运动学逆解
%设置三角形三个顶点
trip1 = [0.25;0.15;0.1];       
trip2 = [0.25;0.25;0.1];
trip3 = [0.30;0.15;0.1];
% RPY世界坐标系欧拉角：
T1_roll = -pi;
T1_pitch = -pi/6; 
T1_yaw = 0;
%% =====================第二问请在此处补全代码=======================%%
%%========================第二问请在此处补全代码=======================%%
%%========================第二问请在此处补全代码=======================%%
%关节末端空间与任务空间之间的映射关系
Num = 60;     %工作空间角度仿真点数
T2_roll = zeros(61,1);
T2_pitch = zeros(61,1);
T2_yaw = zeros(61,1);
%如文档图示绕定点改变姿态。补全T2_psi、T2_theta、T2_roll
%设置固定点T2_pn及工作空间T2_psi、T2_theta、T2_phi(弧度制，deg2rad()),共Num+1个点。
T2_pn = [0.25; 0.0; 0.15];
T2_psi = linspace(0, 2*pi, Num+1);          
T2_theta = deg2rad(12) * ones(1, Num+1);    % 保持12度倾斜绕圈
T2_phi = linspace(0, pi, Num+1);

%末端执行器空间在任务空间下的表示
T2_roll = asin(-sin(T2_theta).*sin(T2_psi)./sqrt(1-sin(T2_theta).*cos(T2_psi).*sin(T2_theta).*cos(T2_psi)));
T2_pitch = asin(sin(T2_theta).*cos(T2_psi))+deg2rad(113.2);
%修改
T2_yaw = asin(-(sin(T2_phi).*cos(T2_theta).*cos(T2_psi) + cos(T2_phi).*sin(T2_psi)) ./ ...
         sqrt(1-(sin(T2_theta).*cos(T2_psi)).^2));
% 关节末端空间与任务空间之间的映射关系
%% =====================第三问请在此处补全代码=======================%%
%%========================第三问请在此处补全代码=======================%%
%%========================第三问请在此处补全代码=======================%%
%%3、代码3
%锥形扫描机器人末端位姿
%r为圆锥底面半径
%N为过原点与圆锥底面平行面的法向量，a、b为过原点与圆锥底面平行面上的正交向量。
N = [-0.9191 0 0.4040];
c = [0 0 1];
a = cross(N, c);
a = a / norm(a);
b = cross(N, a);
b = b / norm(b);
r = 0.04;
Num = 60;
T3_pn    = zeros(61,3);
T3_roll  = zeros(61,1);
T3_pitch = zeros(61,1);
T3_yaw   = zeros(61,1);

T3_psi   = linspace(0, 2*pi, Num+1)';
T3_theta = deg2rad(12) * ones(Num+1, 1);
T3_phi   = T3_psi;

p0 = [0.15; 0.0; 0.18];

h = r / tan(deg2rad(12));
center = p0 - h * (N/norm(N))';

for i = 1:Num+1
    T3_pn(i,:) = center' + r*cos(T3_psi(i))*a + r*sin(T3_psi(i))*b;
end

disp('T3_pn范围：');
disp(['X: ', num2str(min(T3_pn(:,1))), ' ~ ', num2str(max(T3_pn(:,1)))]);
disp(['Y: ', num2str(min(T3_pn(:,2))), ' ~ ', num2str(max(T3_pn(:,2)))]);
disp(['Z: ', num2str(min(T3_pn(:,3))), ' ~ ', num2str(max(T3_pn(:,3)))]);

for i = 1:Num+1
    pe   = T3_pn(i,:)';
    % 修正：末端工具Z轴从pe出发沿-N方向（即朝锥顶反向），
    % 因为机械臂末端是从锥底圆朝锥顶方向延伸
    Znew = pe - p0;            % 从锥顶指向末端
    Znew = Znew / norm(Znew);
    T3_pitch(i) = asin(-Znew(2));
    T3_roll(i)  = atan2(Znew(1), Znew(3));
    T3_yaw(i)   = 0;
end

disp('T3_roll范围(deg)：');
disp([num2str(rad2deg(min(T3_roll))), ' ~ ', num2str(rad2deg(max(T3_roll)))]);
disp('T3_pitch范围(deg)：');
disp([num2str(rad2deg(min(T3_pitch))), ' ~ ', num2str(rad2deg(max(T3_pitch)))]);

x = (tan(T3_theta) .* r .* sin(T3_psi))';
y = (-tan(T3_theta) .* r .* cos(T3_psi))';
z = zeros(1, Num+1);
%% =====================第四问请搭建仿真系统=======================%%
%%=====================第四问请搭建仿真系统=======================%%
%%=====================第四问请搭建仿真系统=======================%%
%%4、simulink仿真
%搭建仿真模型，输入期望角为：theta_ref，输出角为：theta， 命名为：dummy2ndpid

%% =====================请勿修改此条线后的代码=======================%%
%%========================请勿修改此条线后的代码=======================%%
%%========================请勿修改此条线后的代码=======================%%
%% --- Outputs from this function are returned to the command line.
function varargout = Dummy_simulation_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Get default command line output from handles structure
varargout{1} = handles.output;
%% =====================绘制平面三角形=========================%%
% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)view(3)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
uiresume(gcf);
Dummy = Dummymodel();    %启动Dummy模型
global Jtheta buttonValue;
global trip1 trip2 trip3 T1_roll T1_pitch T1_yaw ;  
t = 0:1.25/25:1.25;
clearplot(hObject, eventdata, handles); %清除原有绘图
q0 = [deg2rad(90) 0 0 0 0 0 ];          %设置Dummy初始位置
%对三个顶点间的数值进行插值
num = length(t)-1;
pn = zeros(3*num,3);
p1 = trip1;
p2 = trip2;
p3 = trip3;
for i = 1:3
    switch(i)
        case 1
            subp1 = p2-p1;
            dp = subp1./num;
            for j = 1:25
                p1 = p1+dp;
                pn(j,:) = p1;
            end
            p1 = p1-subp1;
        case 2
            subp2 = p3-p2;
            dp = subp2./num;
            for j = 1:25
                p2 = p2+dp;
                pn(25+j,:) = p2;
            end
        case 3
            subp3 = p1-p3;
            dp = subp3./num;
            for j = 1:25
                p3 = p3+dp;
                pn(50+j,:) = p3;
            end
    end
end
T1 = transl(pn(1,:))*trotz(T1_yaw)*troty(T1_pitch)*trotx(T1_roll);
q1 = Dummy.ikunc(T1);
q = jtraj(q0, q1, t);  %初始位置至运动开始位置的五次插值曲线轨迹规划
axes(handles.axes1);
h1 = quiver3(trip1(1),trip1(2),trip1(3), subp1(1),subp1(2),subp1(3));    %三角形运动方向箭头表示
h2 = quiver3(trip2(1),trip2(2),trip2(3), subp2(1),subp2(2),subp2(3));
h3 = quiver3(trip3(1),trip3(2),trip3(3), subp3(1),subp3(2),subp3(3));
view([1, 1,1]);
if buttonValue
    for i = 1:101
        if  i<27
            axes(handles.axes1);
            title("Draw Triangle");
            hold on
            view([1, 1,1]);
            Dummy.plot(q(i,:));%绘制动画
            hold on
            drawjointangle(hObject, eventdata, handles, q(i,:), i);
        else
            TR=transl(pn(i-26,:))*trotz(T1_yaw)*troty(T1_pitch)*trotx(T1_roll);
            q(i,:)=Dummy.ikunc(TR);
            axes(handles.axes1);
            title("Draw Triangle");
            hold on
            view([1, 1,1]);
            Dummy.plot(q(i,:));%绘制动画
            hold on;
            drawjointangle(hObject, eventdata, handles, q(i,:), i);
        end
    end
elseif ~buttonValue
    for i = 1:26
        axes(handles.axes1);
        title("Draw Triangle");
        hold on
        view([1, 1,1]);
        Dummy.plot(q(i,:));%绘制动画
        hold on;
    end
    for i = 1:75
        TR=transl(pn(i,:))*trotz(T1_yaw)*troty(T1_pitch)*trotx(T1_roll);
        q(i+26,:)=Dummy.ikcon(TR);
        axes(handles.axes1);
        title("Draw Triangle");
        hold on
        view([1, 1,1]);
        Dummy.plot(q(i+26,:));%绘制动画
        hold on;
    end

    drawconjointangle(hObject, eventdata, handles, q);
end
hold off
Jtheta = q;

%%=====================绘制平面三角形（完）=========================%%


%% ====================模拟执行咽拭子任务=======================%%
% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
uiresume(gcf);
% figureHandle = get(handles.axes2, 'Parent')
% disp(['图窗的句柄（编号）是：', num2str(figureHandle)])
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Dummy = Dummymodel();
global Jtheta buttonValue;
global Num T2_pn T2_psi T2_theta T2_phi T2_roll T2_pitch T2_yaw ;
clearplot(hObject, eventdata, handles);
N = [-0.9191 0 0.3939];     %过原点，与拭子末端平面平行面的基向量
c = [0 1 0];
a = cross(N, c);
b = cross(N, a);
% 关节末端空间与任务空间之间的映射关系,末端执行器空间下的姿态由任务空间下的姿态表示
% 关节末端空间与任务空间之间的映射关系
num = length(T2_psi)-1;
conpoint = T2_pn';
%圆锥顶点到圆锥底面圆的距离
r = 0.1;
z = zeros(length(T2_psi),1)';
% 修改
x = tan(T2_theta).*r.*cos(T2_psi);
y = -tan(T2_theta).*r.*sin(T2_psi);
z = z-0.1;
x1 = a(1)*x+b(1)*y+N(1)*z;
y1 = a(2)*x+b(2)*y+N(2)*z;
z1 = a(3)*x+b(3)*y+N(3)*z;
%圆锥顶点到圆锥底面圆的距离
t1 = 0:0.05:2;
q0 = [deg2rad(90) 0 0 0 0 0 ];
%设置轨迹规划
T1 = transl(conpoint)*troty(T2_pitch(1))*trotx(T2_roll(1));
%设置轨迹规划
q1 = Dummy.ikunc(T1);
g= jtraj(q0, q1, t1);
j = 1:6;
axes(handles.axes1);
hold on
set(gca,'XLim',[-0.2,0.4]);  %设置坐标轴范围
set(gca,'YLim',[-0.2,0.4]);
set(gca,'ZLim',[-0.1,0.4]);

if buttonValue
    for i = 1:num+41
        if i<42
            axes(handles.axes1);
            title("Simulate Throat Swabbing");
            view([1, 1,1]);
            hold on
            Dummy.plot(g(i,:));%动画演示
            hold on;
            drawjointangle(hObject, eventdata, handles, g(i,:), i);
        else
            TR=transl(conpoint)*troty(T2_pitch(i-40))*trotx(T2_roll(i-40));
            % P = transl(TR);     %转换为各关节角
            % plot3(P(1),P(2),P(3),'*','LineWidth',2); %读取解析后的位置
            g(i,:)=Dummy.ikcon(TR);
            axes(handles.axes1);
            title("Simulate Throat Swabbing");
            view([1, 1,1]);
            Dummy.plot(g(i,:));%动画演示
            h = quiver3(conpoint(1),conpoint(2),conpoint(3), x1(i-40), y1(i-40), z1(i-40),1);
            hold on
            drawjointangle(hObject, eventdata, handles, g(i,:), i);
        end
    end
elseif ~buttonValue
    for i = 1:41
        axes(handles.axes1);
        title("Simulate Throat Swabbing");
        view([1, 1,1]);
        hold on
        Dummy.plot(g(i,:));%动画演示
        hold on;
    end
    for i = 1:60
        TR=transl(conpoint)*troty(T2_pitch(i+1))*trotx(T2_roll(i+1));
        g(i+41,:)=Dummy.ikcon(TR);
        axes(handles.axes1);
        title("Simulate Throat Swabbing");
        view([1, 1,1]);
        hold on
        Dummy.plot(g(i+41,:));%动画演示
        h = quiver3(conpoint(1),conpoint(2),conpoint(3), x1(i), y1(i), z1(i),1);
        hold on;
    end
    drawconjointangle(hObject, eventdata, handles, g);
end
Jtheta = g;

%%====================模拟执行咽拭子任务（完）========================%%

%% =======================进行锥形扫描===========================%% 
% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
uiresume(gcf);
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global Jtheta buttonValue;
global T3_psi T3_theta T3_roll T3_pitch T3_yaw T3_pn a b N;
Dummy = Dummymodel();
clearplot(hObject, eventdata, handles);
N = [-0.9191 0 0.3939];
c = [0 1 0];
a = cross(N, c);
b = cross(N, a);
r = 0.08;
%锥形扫描位置运动函数
z = zeros(61,1)';
x = tan(T3_theta) .* r .* sin(T3_psi);
y = -tan(T3_theta) .* r .* cos(T3_psi);

%锥形扫描位置运动函数

%锥形扫描顶点到底面的距离，绘制箭头
z = z-0.1;
dx1 = -a(1)*x-b(1)*y+N(1)*z;
dy1 = -a(2)*x-b(2)*y+N(2)*z;
dz1 = -a(3)*x-b(3)*y+N(3)*z;
%锥形扫描顶点到底面的距离，绘制箭头

t1 = 0:0.05:2;
q0 = [deg2rad(90) 0 0 0 0 0 ];
T1 = transl(T3_pn(1,1), T3_pn(1,2), T3_pn(1,3))*troty(T3_roll(1))*trotx(T3_pitch(1));
q1 = Dummy.ikcon(T1);
g = jtraj(q0, q1, t1);
j = 1:6;

axes(handles.axes1);
num = length(T3_psi)-1;
axes(handles.axes1);
hold on
set(gca,'XLim',[-0.2,0.4]);  %设置坐标轴范围
set(gca,'YLim',[-0.2,0.4]);
set(gca,'ZLim',[-0.1,0.4]);

if buttonValue
    for i = 1:num+41
        %设置轨迹规划
        if i<42
            axes(handles.axes1);
            title("Simulate Cone Scan");
            view([1, 1,1]);
            hold on
            Dummy.plot(g(i,:));%动画演示
            hold on;
            drawjointangle(hObject, eventdata, handles, g(i,:), i);
        else
            axes(handles.axes1);
            title("Simulate Cone Scan");
            view([1, 1,1]);
            hold on
            TR = transl(T3_pn(i-40,1), T3_pn(i-40,2), T3_pn(i-40,3))*troty(T3_roll(i-40))*trotx(T3_pitch(i-40));%锥形扫描位置运动函数
            P = transl(TR);
            plot3(P(1),P(2),P(3),'*','LineWidth',2); %读取解析后的位置
            q=Dummy.ikcon(TR);
            view([1, 1,1]);
            Dummy.plot(q);%动画演示
            g(i,:) = q;
            axes(handles.axes1);
            h = quiver3(T3_pn(i-40,1), T3_pn(i-40,2), T3_pn(i-40,3), dx1(i-40),dy1(i-40),dz1(i-40),1);
            hold on;
            drawjointangle(hObject, eventdata, handles, q, i);
        end
    end
elseif ~buttonValue
    for i = 1:41
        axes(handles.axes1);
        title("Simulate Cone Scan");
        view([1, 1,1]);
        hold on
        Dummy.plot(g(i,:));%动画演示
        hold on;
    end
    for i = 1:60
        TR = transl(T3_pn(i+1,1), T3_pn(i+1,2), T3_pn(i+1,3))*troty(T3_roll(i+1))*trotx(T3_pitch(i+1));%锥形扫描位置运动函数
        g(i+41,:) = Dummy.ikcon(TR);
        axes(handles.axes1);
        title("Simulate Cone Scan");
        view([1, 1,1]);
        hold on
        Dummy.plot(g(i+41,:));%动画演示
        h = quiver3(T3_pn(i,1), T3_pn(i,2), T3_pn(i,3), dx1(i),dy1(i),dz1(i),1);
        hold on;
    end

    drawconjointangle(hObject, eventdata, handles, g);
end
Jtheta = g;
%%=======================进行锥形扫描（完）===========================%% 

%% =====================关节角度动态响应==========================%% 
% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global theta_r1 out theta_r2 theta_r3 Jtheta;
global buttonValue
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

open_system('dummy2ndpid');     %打开仿真程序
set_param('dummy2ndpid', 'StartTime', '0', 'StopTime', '5', 'MaxStep', '0.01');
options = simset('SrcWorkspace','current');

g = Jtheta;
theta_ref(:,1) = 0:5/100:5;   %2nd关节角的时间间隔及大小
theta_ref(:,2) = g(:,2);
assignin('base', 'theta_ref', theta_ref);

out = sim('dummy2ndpid',[],options);
theta = out.theta;

axes(handles.axes3);    %在axes3上绘图
indices1 = round(linspace(1, length(theta(:,2)), 101));
selectedData1 = theta(indices1,2);
hold on
plot(theta_ref(:,1),theta_ref(:,2), 'r.','LineWidth',1);    %期望角度
plot(theta_ref(:,1),selectedData1, 'b-','LineWidth',1);   %跟踪角度
% legend('theta','','theta ref','Location','NorthWest');
hold off

%%=====================关节角度动态响应（完）==========================%% 

%% ===========================退出===============================%% 
% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% x = 1
pause(1)  
close (gcf);

%% ===========================清除图像===============================%% 
function clearplot(hObject, eventdata, handles)%清除上一图像
global draw1 draw2 draw3 draw4 draw5 draw6;
q = [deg2rad(90) 0 0 0 0 0 ];
Dummy = Dummymodel();    %启动Dummy模型
axes(handles.axes1);
hold off
title("Calculating......");
set(gca,'XLim',[-0.2,0.4]);  %设置坐标轴范围
set(gca,'YLim',[-0.2,0.4]);
set(gca,'ZLim',[-0.1,0.4]);
view([1, 1,1]);
% view([1, 1,1]);
plot3(10, 10, 10);
hold on
title("Calculating......");
set(gca,'XLim',[-0.2,0.4]);  %设置坐标轴范围
set(gca,'YLim',[-0.2,0.4]);
set(gca,'ZLim',[-0.1,0.4]);
view([1, 1,1]);
% view([1, 1,1]);
Dummy.plot(q);
axes(handles.axes1)
delete(draw1);delete(draw2);delete(draw3);delete(draw4);delete(draw5);delete(draw6);

axes(handles.axes2);
hold off
axis([0 5 -1 1]);
draw1 = plot(-1, -1);
axes(handles.axes3);
hold off
axis([0 5 -1 1]);
draw2 = plot(-1, -1);
axes(handles.axes4);
hold off
axis([0 5 -1 1]);
draw3 = plot(-1, -1);
axes(handles.axes5);
hold off
axis([0 5 -1 1]);
draw4 = plot(-1, -1);
axes(handles.axes6);
hold off
axis([0 5 -1 1]);
draw5 = plot(-1, -1);
axes(handles.axes7);
hold off
axis([0 5 -1 1]);
draw6 = plot(-1, -1);

%% ===========================绘制实时角度图像===============================%% 
function drawjointangle(hObject, eventdata, handles, g, i)
global draw1 draw2 draw3 draw4 draw5 draw6;
hold on
t = 0:5/100:5; 
axes(handles.axes2);
axis([0 5 -inf inf]);
draw1 = plot(t(i), g(1), 'r.','LineWidth',1);
hold on
axes(handles.axes3);
axis([0 5  -inf inf]);
draw2 = plot(t(i), g(2), 'r.','LineWidth',1);
hold on
axes(handles.axes4);
axis([0 5  -inf inf]);
draw3 = plot(t(i), g(3), 'r.','LineWidth',1);
hold on
axes(handles.axes5);
axis([0 5  -inf inf]);
draw4 = plot(t(i), g(4), 'r.','LineWidth',1);
hold on
axes(handles.axes6);
axis([0 5  -inf inf]);
draw5 = plot(t(i), g(5), 'r.','LineWidth',1);
hold on
axes(handles.axes7);
axis([0 5  -inf inf]);
draw6 = plot(t(i), g(6), 'r.','LineWidth',1);
hold on

%% ===========================绘制角度图像===============================%% 
function drawconjointangle(hObject, eventdata, handles, q)
global draw1 draw2 draw3 draw4 draw5 draw6;
hold on
t = 0:5/100:5;
axes(handles.axes2);
axis([0 5 -inf inf]);
draw1 = plot(t, q(:,1), 'r.','LineWidth',1);
hold on
axes(handles.axes3);
axis([0 5  -inf inf]);
draw2 = plot(t, q(:,2), 'r.','LineWidth',1);
hold on
axes(handles.axes4);
axis([0 5  -inf inf]);
draw3 = plot(t, q(:,3), 'r.','LineWidth',1);
hold on
axes(handles.axes5);
axis([0 5  -inf inf]);
draw4 = plot(t, q(:,4), 'r.','LineWidth',1);
hold on
axes(handles.axes6);
axis([0 5  -inf inf]);
draw5 = plot(t, q(:,5), 'r.','LineWidth',1);
hold on
axes(handles.axes7);
axis([0 5  -inf inf]);
draw6 = plot(t, q(:,6), 'r.','LineWidth',1);
hold off

%% ===========================暂停===============================%% 
% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)  %暂停键
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% global f;
uiwait(gcf);

%% ===========================继续===============================%% 
% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
uiresume(gcf);


% --- Executes on button press in radiobutton1.
function radiobutton1_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global buttonValue
buttonValue = get(hObject,'Value');
assignin('base','buttonValue',buttonValue);
% Hint: get(hObject,'Value') returns toggle state of radiobutton1


function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

%% ===========================建立机器人模型===============================%% 
function Dummy = Dummymodel()
%机械臂为六自由度机械臂
%建立机器人模型，指定初始角度
%关节长度
L1=0.109;L2=0.146;L3=0.052;L4=0.115;L5=0.072;Lbase = 0.035;
%%DH法建立模型,关节角，连杆偏移，连杆长度，连杆扭转角，关节类型（0转动，1移动）
L(1) = Link( 'd',L1  ,  'a',Lbase,   'alpha',-pi/2, 'offset',0   );
L(2) = Link( 'd',0   ,  'a',-L2, 'alpha',0    , 'offset',pi/2);
L(3) = Link( 'd',0   ,  'a',-L3, 'alpha',pi/2 , 'offset',0   );
L(4) = Link( 'd',L4  ,  'a',0 ,  'alpha',-pi/2, 'offset',0   );
L(5) = Link( 'd',0   ,  'a',0,   'alpha',pi/2 , 'offset',0);
L(6) = Link( 'd',L5  ,  'a',0,   'alpha',0   ,  'offset',0   );
Dummy=SerialLink(L,'name','Dummy');
%关节角度限制
L(1).qlim=[deg2rad(-170) deg2rad(170)];
L(2).qlim=[deg2rad(-75) deg2rad(90)];
L(3).qlim=[deg2rad(-55) deg2rad(125)];
L(4).qlim=[deg2rad(-180) deg2rad(180)];
L(5).qlim=[deg2rad(-120) deg2rad(120)];
L(6).qlim=[deg2rad(-360) deg2rad(360)];