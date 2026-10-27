function varargout = Inv_Pen(varargin)   % GUI主程序
%**************************************************************%
%This functions creates a graphical user interface which       %
%contains an animation of the Segway example.  This function   %
%was created for the Control Lab for System Dynamics. It       %
%requires the function Inv_Pen.fig to be executed.             %       
%                                                              %
%Copyright (C) 2023 by HUST                                    %
%Created by Prof. Kun Bai and his grad students                %
%Yong Zeng and Guangyu Chen                                    %  
%**************************************************************% 
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Inv_Pen_OpeningFcn, ...
                   'gui_OutputFcn',  @Inv_Pen_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin & isstr(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end
if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end 

%--------------------------------打开界面----------------------------%
function Inv_Pen_OpeningFcn(hObject, eventdata, handles, varargin) 
handles.output = hObject;
guidata(hObject, handles);
global h1 h2;
global v f g_v g_f ll flag reset_flag buttonValue
flag =1;
reset_flag = 0;
buttonValue =0;
axes(handles.sys1);
axis equal;
view([1,0,0]);
axis([0 5 -20 20 0 20]);
grid on;
%view(80,30);
%view([1,0,0]);
reall=15;
ll=4+reall;
v=[0 -2.5 0;0 2.5 0;3 2.5 0;3 -2.5 0;0 -2.5 3;0 2.5 3;3 2.5 3;3 -2.5 3];
f=[1 2 3 4;2 6 7 3;4 3 7 8;1 5 8 4;1 2 6 5;5 6 7 8];
g_v=[1 -0.5 3;1 0.5 3;2 0.5 3;2 -0.5 3;1 -0.5 ll;1 0.5 ll;2 0.5 ll;2 -0.5 ll];
g_f=[1 2 3 4;2 6 7 3;4 3 7 8;1 5 8 4;1 2 6 5;5 6 7 8];
h1=patch('Faces',f,'Vertices',v,'FaceColor','b','FaceAlpha',.9);
h2=patch('Faces',g_f,'Vertices',g_v,'FaceColor','r','FaceAlpha',.9);
%---------------------------------------------------------------------%

%----------------------------------------------------------------------%
function varargout = Inv_Pen_OutputFcn(hObject, eventdata, handles)   %输出函数
varargout{1} = handles.output;
function normal(hObject, eventdata, handles) %公共调用函数
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end
%-------------------------------------------------------------------------%

%-----------------------仿真时间设定------------------------------%
function Tedit_CreateFcn(hObject, eventdata, handles) %仿真时间创建
normal(hObject, eventdata, handles);
function Tedit_Callback(hObject, eventdata, handles)  %仿真时间回调
%-------------------------------------------------------------------%

%---------------------------启动仿真-----------------------------% 
function simulate_Callback(hObject, eventdata, handles)
global draw1 draw2 draw3 draw4 h1 h2;
global v f g_v g_f ll
global out T1 flag reset_flag buttonValue
global theta1 theta2 w1 w2 y1 y2 v10 v20

%将GUI界面参数传递给Simulink模型
%Simulation time
T0 = char(get(handles.Tedit,'String'));
T1 = str2num(get(handles.Tedit,'String'));
%open_system('Invert_Pendulum');
set_param('Invert_Pendulum', 'StopTime',T0);
options = simset('SrcWorkspace','base'); %设置仿真空间
set_param('Invert_Pendulum','SimulationCommand','Update');
out = sim('Invert_Pendulum',[],options);  %第一个参数为模型名，第二个参数为模型开始和结束时间
%out = evalin('base','out');
theta1=min(out.theta(:,1));
theta2=max(out.theta(:,1));
w1=min(out.w(:,1));
w2=max(out.w(:,1));
y1=min(out.y(:,1));
y2=max(out.y(:,1));
v10=min(out.v(:,1));
v20=max(out.v(:,1));
iizz=length(out.tout);
assignin('base','buttonValue',buttonValue);
%plot_theta_w_y_v(1,hObject, eventdata, handles);
if ~buttonValue
    plot_theta_w_y_v(iizz,hObject, eventdata, handles)
end
delete(h1);delete(h2);
axes(handles.sys1);
axis equal;
yy=max(out.y(:,1))*10;
yy=max(yy,20);
axis([0 5 -yy yy 0 20]);
h1=patch('Faces',f,'Vertices',v,'FaceColor','b','EraseMode','normal');
h2=patch('Faces',g_f,'Vertices',g_v,'FaceColor','r','EraseMode','normal');

for ii=1:iizz
    if ~flag||reset_flag==1
        reset_flag = 0;
        break;
    end
    i=out.y(ii,1)*10;
    bbb=[0 i 0;0 i 0;0 i 0;0 i 0;0 i 0;0 i 0;0 i 0;0 i 0;];
    v1=v+bbb;
    set(h1,'Vertices',v1);
    af=deg2rad(out.theta(ii,1));
    gt1=0.5*cos(-af)-0.5;
    gt2=0.5*sin(-af)-0;
    gt3=0.5/cos(af)+(ll-0.5*tan(af))*sin(af)-cos(af)+0.5;
    gt4=(ll-0.5*tan(af))*cos(af)+sin(af)-ll;
    gt5=0.5/cos(af)+(ll-0.5*tan(af))*sin(af)-0.5;
    gt6=(ll-0.5*tan(af))*cos(af)-ll;
    jiao=[0 -gt1 -gt2;0 gt1 gt2;0 gt1 gt2;0 -gt1 -gt2;0 gt3 gt4;0 gt5 gt6;0 gt5 gt6;0 gt3 gt4];
    g_v1=g_v+jiao+bbb;
    set(h2,'Vertices',g_v1);    
    drawnow;    
    if buttonValue
        plot_theta_w_y_v(ii,hObject, eventdata, handles);
    end
    strbar=['T = ',num2str(out.tout(ii)),' s'];
    %waitbar(ii/iizz,hbar,strbar);
    set(handles.Ttext,'string',strbar); 
    pause(0.02)  
end
%close(hbar);

%------------------------------绘图函数-------------------------------%
function plot_theta_w_y_v(ii,hObject, eventdata, handles)
global draw1 draw2 draw3 draw4
global out theta1 theta2 w1 w2 y1 y2 v10 v20
global T1
axes(handles.ag);
draw1=plot(out.tout(1:ii,1),out.theta(1:ii,1),'-r','LineWidth',1,'Marker','o','MarkerSize',0.1);
axis([0 T1 theta1 theta2]);
axes(handles.av);
draw2=plot(out.tout(1:ii,1),out.w(1:ii,1),'-r','LineWidth',1,'Marker','o','MarkerSize',0.1);
axis([0 T1 w1 w2]);
axes(handles.dp);
draw3=plot(out.tout(1:ii,1),out.y(1:ii,1),'-r','LineWidth',1,'Marker','o','MarkerSize',0.1);
axis([0 T1 y1 y2]);
axes(handles.dv);
draw4=plot(out.tout(1:ii,1),out.v(1:ii,1),'-r','LineWidth',1,'Marker','o','MarkerSize',0.1);
axis([0 T1 v10 v20]);

%--------------重置按钮:实现倒立摆模型参数的重置-----------------%
function reset_Callback(hObject, eventdata, handles) 
global draw1 draw2 draw3 draw4 h1 h2 reset_flag;
global f v g_f g_v
set(handles.Tedit,'string','5');
delete(draw1);delete(draw2);delete(draw3);delete(draw4);
delete(h1);delete(h2);
axes(handles.sys1);
axis equal;
view([1,0,0]);
axis([0 5 -20 20 0 20]);
grid on;
%view(80,30);
h1=patch('Faces',f,'Vertices',v,'FaceColor','b','FaceAlpha',.9);
h2=patch('Faces',g_f,'Vertices',g_v,'FaceColor','r','FaceAlpha',.9);
plot_theta_w_y_v(1,hObject, eventdata, handles);
pause(1) 
reset_flag=1;

function exit_Callback(hObject, eventdata, handles)     %退出按钮
global flag
flag = 0;
pause(1)  
close(gcf);

% --- Executes during object creation, after setting all properties.
function figure1_CreateFcn(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function sys3_CreateFcn(hObject, eventdata, handles)

% --- Executes on button press in radiobutton2.
function radiobutton2_Callback(hObject, eventdata, handles)
global buttonValue
buttonValue = get(hObject,'Value');
assignin('base','buttonValue',buttonValue);

function radiobutton2_ButtonDownFcn(hObject, eventdata, handles)
global buttondown_flag
if buttondown_flag
    buttondown_flag=0;
end
% --- Executes when figure1 is resized.
function figure1_SizeChangedFcn(hObject, eventdata, handles)

