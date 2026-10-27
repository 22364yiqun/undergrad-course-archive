import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import ptflops
from .result_plot import result_record,tsne_show
from thop import profile

def test(model, test_dataloader, **option):
    model.eval()

    labels, pred_labels, features = [], [], []
    for input, label in test_dataloader:
        input = input.float()
        input = input.unsqueeze(1) 
        if option['use_gpu']:
            input, label = input.type(torch.FloatTensor).cuda(), label.cuda()

        feature,output = model(input,True)

        features.append(feature.data.cpu().numpy())

        pred = torch.max(output, 1)[1]

        labels.append(label.detach().cpu().numpy())
        pred_labels.append(pred.detach().cpu().numpy())
    
    labels = np.concatenate(labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)
    features = np.concatenate(features, axis=0)

    tsne_show(features, labels, option['tsne_path'])

    return result_record(labels, pred_labels, **option)

def test_time(model):
    x = torch.randn(1,1,32,32)
    model.eval()
    num_iterations = 500  # 迭代次数
    # 预热阶段
    for _ in range(30):
        output = model(x)

    # 正式测试阶段 （1）使用Event来测试
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    total_forward_time = 0.0  # 使用time来测试
    start_event.record()  # 记录开始时间
    for _ in range(num_iterations):
        start_forward_time = time.time()
        output = model(x)
        end_forward_time = time.time()
        forward_time = end_forward_time - start_forward_time
        total_forward_time += forward_time * 1000  # 转换为毫秒
    end_event.record()  # 记录结束时间

    # 同步等待GPU操作完成
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
    fps = num_iterations / elapsed_time
    elapsed_time_ms = elapsed_time / (num_iterations * x.shape[0])

    avg_forward_time = total_forward_time / (num_iterations * x.shape[0])

    print(f"\nFPS: {fps}")
    print("elapsed_time_ms:",elapsed_time_ms * 1000)
    print(f"Avg Forward Time per Image: {avg_forward_time} ms \n")
    print('------------------------------------------------')

    return fps,elapsed_time_ms,avg_forward_time

# def test_flops(model):
#     inputs = torch.randn(1,1,32,32)
#     # for name, param in model.named_parameters():
#     #     print(f"{name}: {param.device}")
#     model.eval()
#     # total_param = 0
#     # for param in model.parameters():
#     #     total_param += np.prod(list(param.data.size()))
#     # print('MODEL/Total parameters:', total_param)

#     # bytes_per_param = 4   # 假设参数是一个32位浮点数(4字节)
#     # total_bytes = total_param * bytes_per_param
#     # total_kilobytes = total_bytes / 1024

#     # print("Total parameters in KB:", total_kilobytes)

#     # inputs = inputs.cuda()  #type(torch.FloatTensor)
#     # print(f"{inputs}: {inputs.device}")
#     # flops,p = profile(model,inputs=(inputs,))
#     # print('Total FLOPs:',flops)

#     # return total_kilobytes,flops

#     param_size = 0 
#     total_param = 0
#     for param in model.parameters():
#     # 获取每个参数的元素数量
#         num_elements = param.numel()
#     # 获取每个元素的字节数
#         element_size = param.element_size()
#         total_param += num_elements
#         param_size += num_elements * element_size
#     total_kilobytes = param_size / 1024
#     print("Total parameters in KB:", total_kilobytes)
#     return total_kilobytes

# def calculate_flops_size_time(model):
#     model.eval()
#     input_tensor = torch.randn(1,1,32,32)
#     input_tensor = input_tensor.type(torch.FloatTensor).cuda()
#     # 计算FLOPs
#     # flops, params = profile(model, inputs=(input_tensor,))

#     # 计算模型大小（KB）
#     total_params = 0
#     param_size = 0
#     for p in model.parameters():
#         num_elements = p.numel()
#         dtype_size = p.element_size()
#         total_params += num_elements
#         param_size += num_elements * dtype_size

#     model_size_kb = param_size / 1024  # 转换为KB
#     print("Total params in :", total_params)
#     print("Total parameters in KB:", model_size_kb)

#     return  total_params , model_size_kb

def calculate_flops_and_size(model):
        # 首先确保模型完全在GPU上
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # 获取原始模型
    
    model = model.cuda()  # 将整个模型移到GPU
    
    # 双重检查所有参数都在GPU上
    for param in model.parameters():
        if not param.is_cuda:
            param.data = param.data.cuda()
    
    # 创建GPU上的输入
    input = torch.randn(1, 1, 32, 32).cuda()
    
    # 计算FLOPs和参数量
    macs, params = profile(model, inputs=(input,))
    
    
    # 转换参数
    mmacs = macs / (1000 * 1000)  # 转换为MMac
    params_k = params / 1000  # 转换为k
    
    # 计算模型大小（KB）
    total_size = 0
    for param in model.parameters():
        total_size += param.nelement() * param.element_size()
    total_kilobytes = total_size / 1024  # 转换为KB
    
    # 测量推理时间
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(input)
    
    # 测量实际运行时间
    times = []
    with torch.no_grad():
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.time()
            _ = model(input)
            torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # 转换为ms
    
    avg_forward_time = sum(times) / len(times)
    
    return mmacs, params_k, total_kilobytes, avg_forward_time

def test_tsne(model, criterion, **option):
    model.eval()

    features, labels, logits = [], [], []
    for input, label in option['train_loader']:
        input = input.float()
        input = input.unsqueeze(1) 
        if option['use_gpu']:
            input, label = input.type(torch.FloatTensor).cuda(), label.cuda()
            
        feature, output = model(input, True)
        logic, _, center = criterion(feature, output)        
        features.append(feature.data.cpu().numpy())
        labels.append(label)
        logits.append(logic.data.cpu().numpy())
    
    for input, label in option['test_loader']:
        input = input.float()
        input = input.unsqueeze(1) 
        if option['use_gpu']:
            input, label = input.type(torch.FloatTensor).cuda(), label.cuda()
            
        feature, output = model(input, True)
        logic, _, _ = criterion(feature, output)        
        features.append(feature.data.cpu().numpy())
        labels.append(label)
        logits.append(logic.data.cpu().numpy())

    features = np.concatenate(features, axis=0)
    
    labels = torch.cat(labels,dim=0).detach().cpu().numpy()
    labels = np.array(labels)
    logits = np.concatenate(logits, axis=0)

#    # 假设每个类别的样本数是一样的，类别标签从 0 到 8
#     labels = []
#     for i in range(9):
#         labels += [i] * (len(option['train_loader'].dataset) // 9)  # 假设每个类别的样本数量是一样的
#     labels = np.array(labels)
    labels = [0] * (len(option['train_loader'].dataset))
    labels += [1] * (option['num_classes'] * 50)
    labels += [2] * ((10 - option['num_classes']) * 50)
    labels = np.array(labels)

    tsne_show(features, labels, option['tsne_train'])