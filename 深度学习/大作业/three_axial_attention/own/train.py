import torch
import numpy as np
from sklearn import model_selection 

def train(model, dataloaders, optimizer, criterion, scheduler, **option):
    model.train()
    torch.cuda.empty_cache()

    total_loss, correct = 0, 0
    
    for inputs, labels in dataloaders:
        inputs = inputs.float()
        inputs = inputs.unsqueeze(1) 
        if option['use_gpu']:
            inputs, labels = inputs.type(torch.FloatTensor).cuda(), labels.cuda()

        
        # optimizer.zero_grad()
        # output = model(inputs)
        # logic, loss, _ = criterion(output, labels)
        # loss.backward()
        # optimizer.step()
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            feature, output = model(inputs, True)
            logic, loss, _ = criterion(feature, output, labels)

            loss.backward()
            optimizer.step()

        preds = logic.data.max(1)[1]

        total_loss += loss.item() * inputs.size(0)
        correct += torch.sum(preds == labels.data)
    
    epoch_loss = (total_loss / len(dataloaders.dataset))
    epoch_acc = (correct.double() / len(dataloaders.dataset))

    scheduler.step()

    return epoch_loss, epoch_acc, scheduler.get_last_lr()[0]


def evaluate(model, dataloaders, criterion, **option):
    torch.cuda.empty_cache()

    total_loss, correct = 0, 0
    model.eval()
    for inputs, labels in dataloaders:
        inputs = inputs.float()
        inputs = inputs.unsqueeze(1)
        if option['use_gpu']:
            inputs, labels = inputs.type(torch.FloatTensor).cuda(), labels.cuda()
            
        feature,output = model(inputs,True)
        logic, loss, _ = criterion(feature, output, labels)
            
        preds = logic.data.max(1)[1]

        total_loss += loss.item() * inputs.size(0)
        correct += torch.sum(preds == labels.data)

    epoch_loss = (total_loss / len(dataloaders.dataset))
    epoch_acc = (correct.double() / len(dataloaders.dataset))

    return epoch_loss, epoch_acc