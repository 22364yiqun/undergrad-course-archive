import time, datetime, torch
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter


from source_data.dataloader import load_data
from own.create_model import init_network,init_loss,init_optimizer
from own.util import writer_hyperparameters,epoch_time,make_dir,device_choice,seed_setting
from own.train import train,evaluate
from own.result_plot import acc_loss_curve,final_record
from own.test import test,test_time,calculate_flops_and_size,test_tsne
import opts
import warnings
from thop import profile

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


warnings.filterwarnings('ignore')
################################################################
# Main Worker

def main_worker(option):

    print('======>> Building Model......')
    model = init_network(option)
    print('Finished. \n')

    print('======>> Choosing Criterion & Optimizer ........ ')
    criterion = init_loss(**option)
    opt = init_optimizer(model, criterion, **option)
    print('Finished. \n')

    scheduler = None
    if option['step_size'] > 0:
        scheduler = lr_scheduler.MultiStepLR(opt, milestones=[int(option['num_epochs'] * 0.3),int(option['num_epochs'] * 0.6),
        int(option['num_epochs'] * 0.9)], gamma=0.1)
    
    writer_hyperparameters(option)

    start_time = time.time()

    writer = SummaryWriter(log_dir=option['tensorboard_path'])
    with open(option['log_path'], 'a+') as fout:
        fout.write('\nModel Train\n')
        fout.write("---------------------------------------------------------------------------------------------\n")
        print('======>> Starting Model Training........ ')

    stop, best_valid_loss = 0, np.inf
    trainloss, trainacc = [], []
    if option['val']:
        valloss, valacc = [], []
    for epoch in range(option['num_epochs']):

        epoch_start = time.time()
        train_loss, train_acc, lr = train(model, option['train_loader'], opt, criterion, scheduler, **option)
        trainloss.append(train_loss)
        trainacc.append(train_acc.item())
        if option['val']:
            val_loss, val_acc = evaluate(model, option['val_loader'], criterion, **option)
            valloss.append(val_loss)
            valacc.append(val_acc.item()) 

        epoch_end = time.time()

        epoch_mins, epoch_secs = epoch_time(epoch_start, epoch_end)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        if option['val']:
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("Acc/Val", val_acc, epoch)
        writer.add_scalar('lr', lr, epoch)

        if option['val']:
            writer.add_scalars('Acc', {'train_acc': train_acc}, epoch)
            writer.add_scalars('Loss', {'train_loss': train_loss}, epoch) 
        else:
            writer.add_scalars('Acc', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)
            writer.add_scalars('Loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch) 


        with open(option['log_path'], 'a+') as fout:
            fout.write(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')
            fout.write(f'\tTrain Loss: {train_loss:.6f} | Train Acc: {train_acc*100:.2f}%\n')
            fout.write(f'\tVal Loss: {val_loss:.6f} | Val Acc: {val_acc*100:.2f}%\n')

            if option['verbose']:
                print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.6f} | Train Acc: {train_acc*100:.2f}%')
                print(f'\tVal Loss: {val_loss:.6f} | Val Acc: {val_acc*100:.2f}%')
        
        # save checkpoint
        if option['val']:
            if val_loss < best_valid_loss:
                stop = 0
                best_valid_loss = val_loss
                torch.save(model.state_dict(), '{}.pth'.format(option['model_path']))
         
            if stop >= option['early_stop'] and option['early_stop'] != 0:
                break
            
    writer.close()
    model.load_state_dict(torch.load('{}.pth'.format(option['model_path'])))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}\n".format(elapsed))

    acc_loss_curve(trainloss, trainacc, valloss, valacc, option['curve_path'])

    return model, criterion

if __name__ == '__main__':
    args = opts.parse()
    my_option = vars(args)

    for i in range(1):  # 每次改变值
        # depth = i + 1
        # my_option.update({'depth':depth})

        # my_option.update({'root_folder':'{}_time{}_depth{}_heads{}'.format(my_option['root_folder'],my_option['data_time'],my_option['depth'],my_option['heads'])})
        my_option.update({'root_folder':'{}_time{}_times{}'.format(my_option['root_folder'],my_option['data_time'],my_option['times'])})
        print(my_option['root_folder'])

        folders = [my_option['tensorboard_folder'], my_option['log_folder'], my_option['model_folder'], my_option['plot_folder']]
        for folder in folders:
            make_dir(my_option['root_folder'], folder)   # 创建根目录

        print('======>> Loading Data........')
        dataloader, label_class = load_data(my_option)
        my_option['train_loader'], my_option['test_loader'] = dataloader['train'], dataloader['test']
        if my_option['val']:
            my_option['val_loader'] = dataloader['val']  

        my_option.update({'label_class': label_class})
        print(label_class)
        print('Finished. \n')

        my_option['tensorboard_file_name'] = f"tensorboard_{my_option['model_name']}_{my_option['num_epochs']}"
        my_option['log_file_name'] = f"log_{my_option['model_name']}_{my_option['num_epochs']}"
        my_option['model_file_name'] = f"model_{my_option['model_name']}_{my_option['num_epochs']}"
        my_option['plot_file_name'] = f"plot_{my_option['model_name']}_{my_option['num_epochs']}"

        seed_setting(my_option)
        device = device_choice(my_option)
        my_option['device'] = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'


        test_acc_list, test_f1_list, test_recall_list, test_precision_list = [], [], [], []

        for cur_time in range(my_option['num_times']):
            my_option.update({'cur_time': cur_time+1})

            my_option['tensorboard_path'] = f"{my_option['root_folder']}/{my_option['tensorboard_folder']}/{my_option['tensorboard_file_name']}_{my_option['cur_time']}"
            my_option['log_path'] = f"{my_option['root_folder']}/{my_option['log_folder']}/{my_option['log_file_name']}_{my_option['cur_time']}.txt"
            my_option['model_path'] = f"{my_option['root_folder']}/{my_option['model_folder']}/{my_option['model_file_name']}_{my_option['cur_time']}"
            my_option['curve_path'] = f"{my_option['root_folder']}/{my_option['plot_folder']}/{'ACC_Loss_Curve'}_{my_option['cur_time']}.png"
            my_option['matrix_path'] = f"{my_option['root_folder']}/{my_option['plot_folder']}/{'Confusion_Matrix'}_{my_option['cur_time']}.png"
            my_option['tsne_train'] = f"{my_option['root_folder']}/{my_option['plot_folder']}/{'TSNE_Train_TEST'}_{my_option['cur_time']}.png"
            my_option['tsne_test'] = f"{my_option['root_folder']}/{my_option['plot_folder']}/{'TSNE_Only_Test'}_{my_option['cur_time']}.png"
            my_option['tsne_path'] = f"{my_option['root_folder']}/{my_option['plot_folder']}/{'TSNE'}_{my_option['cur_time']}.png"
    
            my_option['novelty_score'] = f"{my_option['root_folder']}/{my_option['log_folder']}/{'Novelty_Score'}_{my_option['cur_time']}.png"
            my_option['novelty_score_npy'] = f"{my_option['root_folder']}/{my_option['log_folder']}/{'Novelty_Score'}_{my_option['cur_time']}.npy"
        

            model, criterion = main_worker(my_option)
            # test_tsne(model, criterion, **my_option)  
            # model = init_network(my_option)
            # torch.save(model.state_dict(), '/home/chendezhang1/model_sbcformer_100_1.pth')
            accuracy, f1, recall, precision = test(model, my_option['test_loader'], **my_option)

            fps,elapsed_time_ms,avg_forward_time = test_time(model)
            model = model.to(my_option['device'])
            ################     view_model_param
            mmacs, params_k, total_kilobytes, avg_forward_time = calculate_flops_and_size(model)


            with open(my_option['log_path'], 'a+') as f:
                f.write('\n-----------------Model Statistics-----------------\n')
                f.write(f'FLOPs (MMac)        : {mmacs:.2f}\n')
                f.write(f'Parameters (k)       : {params_k:.2f}\n')
                f.write(f'Total size (KB)      : {total_kilobytes:.2f}\n')
                f.write(f'Inference time (ms)  : {avg_forward_time:.2f}\n')
                f.write('----------------------------------------------\n')

            # 将其写入日志
            ################################

            test_acc_list.append(accuracy)
            test_f1_list.append(f1)
            test_recall_list.append(recall)
            test_precision_list.append(precision)

        final_record(test_acc_list, test_f1_list, test_recall_list, test_precision_list, **my_option)