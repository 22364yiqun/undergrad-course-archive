################################################################
# Import Relative Libraries
import os, random, torch, shutil
import numpy as np
################################################################
# Main Utils
def seed_setting(option):
    if option['seed_fixed']:
        print('Fixed Random Number Seed!')
        random.seed(option['seed'])
        os.environ['PYTHONHASHSEED'] = str(option['seed']) # 为了禁止hash随机化，使得实验可复现。
        np.random.seed(option['seed'])
        torch.manual_seed(option['seed'])
        torch.cuda.manual_seed(option['seed'])
        torch.cuda.manual_seed_all(option['seed'])
        torch.backends.cudnn.deterministic = True

def device_choice(options):
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        options.update({'use_gpu': use_gpu})
    else:
        print("Currently using CPU")

def make_dir(root, folder): # 创建新目录，数据保存
    folder_name = root  + '/' + folder
    if os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
        os.makedirs(folder_name)
        pass
    else:
        os.makedirs(folder_name)

def writer_hyperparameters(opt):   # 将超参数写入日志
    with open(opt['log_path'], 'a+') as f:
        f.write("Hyperparameters\n")
        f.write("---------------------------------------------------------------------------------------------\n")
        if torch.cuda.is_available():
            f.write('%20s: %s\n' % ('device', torch.cuda.get_device_name(0)))
        else:
            device = torch.device('cpu')
            f.write('%20s: %s\n' % ('device', device.type))

        for key in list(opt.keys()):
            f.write('%20s: %s\n' % (key, opt[key]))
        f.write("---------------------------------------------------------------------------------------------\n")

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()