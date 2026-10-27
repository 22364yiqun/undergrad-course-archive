# 系统动力学资料

这里整理了系统动力学课程的笔记、作业和 Matlab/Simulink 上机实验。

| 位置 | 内容 |
| --- | --- |
| [系统动力学.pdf](系统动力学.pdf) | 37 页课程笔记，从拉普拉斯变换的数学复习开始。 |
| [手写笔记.pdf](手写笔记.pdf) | 13 页手写复习笔记，包含拉普拉斯变换、传递函数和机械系统建模等内容。 |
| [Matlab tutorial_v2.pdf](<Matlab tutorial_v2.pdf>) | 14 页 Matlab 入门教程。 |
| HW1–HW3 | 三次作业题目及对应答案（`HWn.pdf` 和 `HWn_sol.pdf`）。 |
| [hw3](hw3/) | 第三次作业用到的 Matlab 脚本：Z-N 法整定 PID 参数，以及求特征多项式。 |
| [Computer Lab 1_Suspension](<Computer Lab 1_Suspension/>) | 实验一：1/4 车辆悬架模型的阶跃和正弦响应仿真，并用 GUI 演示车辆响应。 |
| [Computer_Lab_2](Computer_Lab_2/) | 实验二：倒立摆建模与 Simulink 仿真。 |
| [Computer_Lab_3](Computer_Lab_3/) | 实验三：基于机器人工具箱的六自由度机械臂建模、轨迹仿真与 PID 控制。 |

每个实验文件夹包含：

- `Lab n.pdf`：实验题目。
- `实验报告.md`：实验报告源文件，配图放在 `img/` 中；对应的 PDF 是 `实验一.pdf`、`实验二.pdf`、`实验三.pdf`。
- `.m`、`.fig`、`.slx`：Matlab 代码、GUI 文件和 Simulink 模型。

实验三的代码依赖 Peter Corke 的 Robotics Toolbox for MATLAB（课程使用 10.3.1 版）。工具箱是第三方软件，体积较大，没有放进仓库，请自行下载并在 Matlab 中添加到路径后再运行。Simulink 生成的缓存文件（`slprj/`、`.slxc`）同样没有收录，打开模型时会自动重新生成。
