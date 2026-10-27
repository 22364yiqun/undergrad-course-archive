import argparse

def parse():
    parser = argparse.ArgumentParser(description='Fault Diagnosis')

    # General settings
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_classes', type=int, default=9)
    parser.add_argument('--num_times', type=int, default=3)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--seed_fixed', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--use_gpu', type=bool, default=True)

    parser.add_argument('--val', type=bool, default=True)
    
    # Parameters of root dir
    parser.add_argument('--root_folder', type=str, default='./home/chendezhang/improved/record/test1/') 
    parser.add_argument('--tensorboard_folder', type=str, default='tensorboard') 
    parser.add_argument('--log_folder', type=str, default='log') 
    parser.add_argument('--model_folder', type=str, default='model') 
    parser.add_argument('--plot_folder', type=str, default='plot')
    parser.add_argument('--data_time', type=str, default='2024.12.20')
    # Parameters of DataLoader
    parser.add_argument('--dataset_name', type=str, default='bearing')
    parser.add_argument('--domain', type=str, default='CWRU')
    parser.add_argument('--motor_loader', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--signal_size', type=int, default=1024)
    parser.add_argument('--enc_size', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=4)

    # Parameters of models
    parser.add_argument('--label_class', type=int, default=10)
    parser.add_argument('--times', type=int, default=1)  # 
    parser.add_argument('--model_name', type=str, default='sbcformer')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument("--step_size", type=int, default=50, help='step_size')
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--loss', type=str, default='Softmax', help="Softmax,GCPLoss,RPLoss,ARPLoss,AMPFLoss,SLCPLoss")

    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--out_channel', type=int, default=10)

    opt = parser.parse_args()
    
    return opt