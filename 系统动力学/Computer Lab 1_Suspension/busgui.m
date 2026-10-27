function varargout = busgui(varargin)

%**************************************************************%
%This functions creates a graphical user interface which       %
%contains an animation of the Bus Suspension example.  This    %
%function was created for the Control Tutorials for Matlab.  %
%**************************************************************%                
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @busgui_OpeningFcn, ...
                   'gui_OutputFcn',  @busgui_OutputFcn, ...
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


% --- Executes just before busgui is made visible.
function busgui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to busgui (see VARARGIN)

% Choose default command line output for busgui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes busgui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = busgui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes during object creation, after setting all properties.
function axes2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes2


% --- Executes on button press in Start.
function Start_Callback(hObject, eventdata, handles)
% hObject    handle to Start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 
%Callback for the RUN button%
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Define the transfer function here %%%%
  m1=465;
  m2=50;
  k1 = 5700;
  k2 = 135000;
  b1 = 290;
  
  numa = [39150000 769500000];
  %numa = [62775000 39150000 769500000];
  dena = [23250 149350 65710500 39150000 769500000];

%%%%%%%%%%%Do Not change anything below%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
%Run the simulation with a 0.1m step disturbance%     
  t=0:0.04:7;
  u=0.1*ones(size(t));
  [Y]=lsim(numa,dena,u,t);
  
  lt=length(t);
  changet=t;
  for nt=1:lt-1
    changet(1:23)=-0.91:0.04:0;
    changet(nt+23)=t(nt);
  end
  t=changet;  
  
  lY=length(Y);
  change=Y;
  for nc=1:lt-1
    change(1:23)=change(1:23)*0;
    change(nc+23)=Y(nc);
  end  
  Y=change; 
  axes(handles.axes3)
    plot(t(1),Y(1))
    hold on
  title('Bus response to a step disturbance of 0.1m')
  xlabel('Time (sec.)')
  ylabel('Bus oscillation (m)')
  axis([-1 7 -0.1 0.3])
  hold on
  
  %Draw the bus and ground%  
  l=1;
  h=0.5;

  x1=-l/2;
  y1=Y+0.3;
  x2=-l/2;
  y2=Y+0.3+h;
  x3=l/2;
  y3=Y+0.3+h;
  x4=l/2;
  y4=Y+0.3;
  x5=l/10;
  y5=Y+0.3;
  x6=l/10;
  y6=Y+0.4;
  x7=-l/10;
  y7=Y+0.4;
  x8=-l/10;
  y8=Y+0.3;

  st=0.4+Y;
  sb=0.29+Y;
  tire=0.2;

  gxr=1;
  gy=0.105;
  sxr=-1:0.04:7;
  sxl=-1;
  sy=0.205;

  wx1=-0.35;
  wy1=0.6+Y;
  wx2=-0.35;
  wy2=0.7+Y;
  wx3=-0.25;
  wy3=0.7+Y;
  wx4=-0.25;
  wy4=0.6+Y;
  
 axes(handles.axes2)
 
  cla
  A = plot([x1 x2 x3 x4 x5 x6 x7 x8 x1],...
         [y1(1) y2(1) y3(1) y4(1) y5(1) y6(1) y7(1) y8(1) y1(1)],...
         'k','LineWidth',[2]);
     drawnow
     hold on
         
  B = plot([wx1 wx2 wx3 wx4 wx1],[wy1(1) wy2(1) wy3(1) wy4(1) wy1(1)],...         
         'k','LineWidth',[2]);
     drawnow
         
  C = plot([wx1+0.2 wx2+0.2 wx3+0.2 wx4+0.2 wx1+0.2],[wy1(1) wy2(1) wy3(1) wy4(1) wy1(1)],...         
         'k','LineWidth',[2]);
     drawnow

  D = plot([wx1+0.4 wx2+0.4 wx3+0.4 wx4+0.4 wx1+0.4],[wy1(1) wy2(1) wy3(1) wy4(1) wy1(1)],...         
         'k','LineWidth',[2]); 
     drawnow

  E = plot([wx1+0.6 wx2+0.6 wx3+0.6 wx4+0.6 wx1+0.6],[wy1(1) wy2(1) wy3(1) wy4(1) wy1(1)],...         
         'k','LineWidth',[2]);
     drawnow
                         
  L = plot([sxl sxr(1) sxr(1) gxr],[sy sy gy gy], 'g',...
                'LineWidth',[2]);
            drawnow

  I = plot([0 0],[st(1) sb(1)], 'k',...
               'LineWidth',[2]);
           drawnow
                
  radius = 0.09;
  arcstep = 18;
  j = 0:arcstep:(360-arcstep);  
  arcx = radius * cos((j+arcstep) * pi/180);  
  arcy = radius * sin((j+arcstep) * pi/180);  

  J = patch(arcx, arcy+tire(1), 'b'); 
  drawnow
  title('1/4 Bus suspension model animation');
  axis([-1 1 0 1.5])

   lY=length(Y);
  for z=2:lY-1   
    % 如果对象已被删除(比如按了Reset)，则中断动画循环
    if ~ishghandle(L) || ~ishghandle(A)
       break;
    end
    if z == 24
      sb=0.39+Y;
      tire=0.3;
      y1=Y+0.4;
      y4=Y+0.4;
      y2=Y+0.4+h;
      y3=Y+0.4+h;
      y8=Y+0.4;
      y5=Y+0.4;
      y7=Y+0.5;
      y6=Y+0.5;
      st=0.5+Y;
      wy1=0.7+Y;
      wy2=0.8+Y;
      wy3=0.8+Y;
      wy4=0.7+Y;
    end

    set(L, 'XData', [sxl sxr(z) sxr(z) gxr]);
    set(J, 'YData', arcy+tire);
    set(A, 'YData', [y1(z) y2(z) y3(z) y4(z) y5(z) y6(z) y7(z) y8(z) y1(z)]);
    set(B, 'YData', [wy1(z) wy2(z) wy3(z) wy4(z) wy1(z)]);
    set(C, 'YData', [wy1(z) wy2(z) wy3(z) wy4(z) wy1(z)]);
    set(D, 'YData', [wy1(z) wy2(z) wy3(z) wy4(z) wy1(z)]);
    set(E, 'YData', [wy1(z) wy2(z) wy3(z) wy4(z) wy1(z)]); 
    set(I, 'YData', [st(z) sb(1)]);
    drawnow;
    
      axes(handles.axes3)
      plot([t(z),t(z+1)],[Y(z),Y(z+1)])
      hold on
      axis([-1 7 -0.1 0.3])
  end
  guidata(hObject,handles);
% --- Executes on button press in Reset.
function Reset_Callback(hObject, eventdata, handles)
% hObject    handle to Reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
  axes(handles.axes2)
  
  l=1;h=0.5;x1=-l/2;y1=0.3;
  x2=-l/2;y2=0.3+h;x3=l/2;
  y3=0.3+h;x4=l/2;y4=0.3;
  x5=l/10;y5=0.3;x6=l/10;
  y6=0.4;x7=-l/10;y7=0.4;
  x8=-l/10;y8=0.3;st=0.4;
  sb=0.29;tire=0.2;gxr=1;
  gy=0.105;sxr=-1;sxl=-1;
  sy=0.205;wx1=-0.35;wy1=0.6;
  wx2=-0.35;wy2=0.7;wx3=-0.25;
  wy3=0.7;wx4=-0.25;wy4=0.6;

  cla
  A = plot([x1 x2 x3 x4 x5 x6 x7 x8 x1],...
         [y1 y2 y3 y4 y5 y6 y7 y8 y1],...
         'k','LineWidth',[2]);
     drawnow
  hold on
         
  B = plot([wx1 wx2 wx3 wx4 wx1],[wy1 wy2 wy3 wy4 wy1],...         
         'k','LineWidth',[2]);
     drawnow
         
  C = plot([wx1+0.2 wx2+0.2 wx3+0.2 wx4+0.2 wx1+0.2],[wy1 wy2 wy3 wy4 wy1],...      
         'k','LineWidth',[2]);
     drawnow

  D = plot([wx1+0.4 wx2+0.4 wx3+0.4 wx4+0.4 wx1+0.4],[wy1 wy2 wy3 wy4 wy1],...      
         'k','LineWidth',[2]); 
     drawnow

  E = plot([wx1+0.6 wx2+0.6 wx3+0.6 wx4+0.6 wx1+0.6],[wy1 wy2 wy3 wy4 wy1],...      
         'k','LineWidth',[2]); 
     drawnow
                         
  L = plot([sxl sxr sxr gxr],[sy sy gy gy], 'g',...
         'LineWidth',[2]);
     drawnow

  I = plot([0 0],[st sb], 'k',...
         'LineWidth',[2]);
     drawnow
                
  radius = 0.09;
  arcstep = 18;
  j = 0:arcstep:(360-arcstep);  
  arcx = radius * cos((j+arcstep) * pi/180);  
  arcy = radius * sin((j+arcstep) * pi/180);  

  J = patch(arcx, arcy+tire, 'b'); 
  drawnow
                                                               
  title('1/4 Bus suspension model animation')

%Clear the time response plot%
  axes(handles.axes3)
  cla
  title('Bus response to a step disturbance of 0.1m')
  xlabel('Time (sec.)')
  ylabel('Bus oscillation (m)')
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function axes3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
 
% Hint: place code in OpeningFcn to populate axes3

% --- Executes on button press in exit.
function exit_Callback(hObject, eventdata, handles)
% hObject    handle to exit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close(busgui);

