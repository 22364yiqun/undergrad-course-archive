import os, torch
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import islice
import numpy as np
from PyEMD import EMD
from torch.utils.data import Dataset

def s_to_i(tmp):
    # for element in np.nditer(tmp):
    #     new = 32*32*(element - tmp.min())/(tmp.max()-tmp.min())
    #     element = new
    return tmp.reshape(32,32)

class dataset(Dataset):

    def __init__(self, list_data):
        self.seq_data = list_data['data'].tolist()
        self.labels = list_data['label'].tolist()

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        seq = self.seq_data[item]
        label = self.labels[item]
        
        return seq, label

class bearing(object):
    
    def __init__(self, signal_size, enc_size, batch_size, num_workers, domain, num_class,emd_used=False): 
        self.data_dir = r'/home/chendezhang/three_axial_attention/bearing/{}'.format(domain)  # '../dataset/bearing/{}/'/home/chendezhang/improved/bearing/{}

        self.num_classes = num_class  # 总分类数 D:\desktop\three_axial_attention\bearing\source
        self.signal_size = signal_size
        self.enc_size = enc_size

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.emd_used = emd_used
        self.domain = domain 

        # self.dataname = ['Health.csv', 'Inner_1.csv', 'Ball_1.csv', 'Outer_1.csv', 'Combine_1.csv',\
        #                 'Inner_0.5.csv', 'Ball_0.5.csv', 'Outer_0.5.csv', 'Combine_0.5.csv']

        self.dataname = ['B007_1.csv', 'B014_1.csv', 'B021_1.csv', 'IR007_1.csv', 'IR014_1.csv',\
                        'IR021_1.csv','normal_1.csv','OR007@6_1.csv','OR014@6_1.csv','OR021@6_1.csv']
        # # self.dataname = ['Healthy with pulley.csv', '0.7inner-100watt.csv', '0.9inner-100watt.csv', '1.1inner-100watt.csv', '1.3inner-100watt.csv',\
        #                 '1.5inner-100watt.csv','1.7inner-100watt.csv']

        # self.dataname = ['#0_30rpm_collect60s.csv', '#1_30rpm_collect60s.csv', '#2_30rpm_collect60s.csv', '#3_30rpm_collect60s.csv', '#4_30rpm_collect60s.csv',\
        #                 '#5_30rpm_collect60s.csv','#6_30rpm_collect60s.csv','#7_30rpm_collect60s.csv','#8_30rpm_collect60s.csv','#9_30rpm_collect60s.csv']


        self.label = [i for i in range(self.num_classes)]  

        self.label_class = {i: self.dataname[i] for i in range(self.num_classes)} # name
    
    def dataloader_maker(self):
        list_data = self.get_files()

        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        train_pd, val_pd, test_pd = self.train_test_split_order(data_pd)   # 字典
        
        datasets = {}
        datasets['train'] = dataset(list_data=train_pd)
        datasets['val'] = dataset(list_data=val_pd)
        datasets['test'] = dataset(list_data=test_pd)

        self.dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=self.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=self.num_workers)
                            for x in ['train', 'val', 'test']}

        return self.dataloaders
    
    def get_files(self):
        
        data, lab = [], []    # 数据和标签

        for i in tqdm(range(len(self.dataname))):  # tqdm 提示进度
            path2 = os.path.join(self.data_dir, self.dataname[i])   # 名字

            data1, lab1 = self.data_load(path2, label=self.label[i])
            
            data += data1
            lab += lab1
        return [data, lab]
    
    def data_load(self, filename, label):
        file = pd.read_csv(filename, delimiter=",", header=None, skiprows=1)
        file_data = [] 
        col_index_list = file.columns.tolist()
        column_index = col_index_list[0]   #DH 4 CWRU 0
        column_data = file.iloc[:, column_index].tolist()
        for item in column_data:
            file_data.append(float(item))

        # file = open(filename, "r", encoding='gb18030', errors='ignore')
        # file_data = []
        # for line in islice(file, 22, None): 
        #     line = line.rstrip()
        #     word = line.split("\t", 5)    
        #     file_data.append(eval(word[3]))


        file_data = np.array(file_data).reshape(-1)

        data = []
        lab = []
        start, end = 0, self.signal_size
        while end <= file_data.shape[0]:
            tmp =file_data[start:end].reshape(1, -1)

            if self.emd_used:
                tmp = EMD()(tmp.reshape(-1), max_imf=5)
            tmp = s_to_i(tmp)
            data.append(tmp)

            lab.append(label)

            start += self.enc_size 
            end += self.enc_size 

        return data, lab
    
    def train_test_split_order(self, data_pd, split_proportion=[0.7, 0.1, 0.2]):  # 分开
        train_pd = pd.DataFrame(columns=('data', 'label'))
        val_pd = pd.DataFrame(columns=('data', 'label'))
        test_pd = pd.DataFrame(columns=('data', 'label'))
        for i in range(self.num_classes):
            data_pd_tmp = data_pd[data_pd['label'] == i].reset_index(drop=True)

            train_val_point = int(split_proportion[0] * data_pd_tmp.shape[0])
            val_test_point = int((split_proportion[0] + split_proportion[1]) * data_pd_tmp.shape[0])

            tmp_train = data_pd_tmp.loc[:train_val_point, ['data', 'label']]
            train_pd = pd.concat([train_pd, tmp_train], ignore_index=True)

            tmp_val = data_pd_tmp.loc[train_val_point:val_test_point, ['data', 'label']]
            val_pd = pd.concat([val_pd, tmp_val], ignore_index=True)

            tmp_test = data_pd_tmp.loc[val_test_point:, ['data', 'label']]
            test_pd = pd.concat([test_pd, tmp_test], ignore_index=True)
        
        return train_pd, val_pd, test_pd