
"""
CNN model training, validation and testing - MGR

@author: belovm96
"""

from torch.utils.data import Dataset, DataLoader
import argparse
import torch.nn as nn
import torch
import h5py
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import torch.optim as optim


exp_name = 'CNN_Model_Specs'

#make a directory for the experiment
if not os.path.exists(exp_name):
    os.makedirs(exp_name)
    
    
parser = argparse.ArgumentParser(description=exp_name)

#add hyperparameters to the parser
parser.add_argument('--batch-size', type=int, default=10,
                help='input batch size for training (default: 10)')
parser.add_argument('--epochs', type=int, default=50,
                help='number of epochs to train (default: 50)')
parser.add_argument('--freq-dim', type=int, default=432,
                help='row dimension of the feature')
parser.add_argument('--cuda', action='store_true', default=True,
                help='enables CUDA training (default: True)')  # when you have a GPU
parser.add_argument('--lr', type=float, default=1e-3,
                help='learning rate (default: 1e-3)')
parser.add_argument('--model-save', type=str,  default='best_model_CNN_Spectrogram.pt',
                help='path to save the best model')
parser.add_argument('--tr-data', type=str,  default='training_specs_GTZAN_more.hdf5',
                help='path to training dataset')
parser.add_argument('--val-data', type=str,  default='validation_specs_GTZAN_more.hdf5',
                help='path to training dataset')
parser.add_argument('--test-data', type=str,  default='testing_specs_GTZAN_more.hdf5',
                help='path to training dataset')

args, _ = parser.parse_known_args()
args.cuda = args.cuda and torch.cuda.is_available()

if args.cuda:
    kwargs = {'num_workers': 1, 'pin_memory': True} 
else:
    kwargs = {}
    
#Model Class Inherits from PyTorch
class CNN_Tempogram(nn.Module):
    def __init__(self):
        super(CNN_Tempogram, self).__init__()
    
        self.kernel = 5
        self.channel = 16
        self.output = 10
        
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=self.kernel, stride=1, padding=(self.kernel-1)//2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=(self.kernel-1)//2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=(3-1)//2)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
    
        self.fc = nn.Linear(62208, self.output)
        
    def forward(self, input):
        
        input = input.unsqueeze(1)
        
        out = self.cnn1(input)
        out = self.relu1(out)
        out = self.maxpool1(out)
    
        
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        
        out = self.cnn3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
       
        out = out.view(out.size(0), -1)
        out = self.fc(out)
    
        return out

#Instantiate model object and check model's parameters
model_CNN = CNN_Tempogram()

print(model_CNN)

if args.cuda:
    model_CNN = model_CNN.cuda()

#define the optimizer
optimizer = optim.Adam(model_CNN.parameters(), lr=args.lr)
scheduler  = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
scheduler.step()

class dataset_pipeline(Dataset):
    def __init__(self, path):
        super(dataset_pipeline, self).__init__()
        
        self.h5pyLoader = h5py.File(path, 'r')
        
        self.labels = self.h5pyLoader['labels_spec_int']
        
        self.temp = self.h5pyLoader['spectrogram']
    
        self._len = self.temp.shape[0]
        
    def __getitem__(self, index):
        
        temp_item = torch.from_numpy(self.temp[index].astype(np.float32))
        label_item = torch.from_numpy(np.array(self.labels[index]).astype(np.float32))
        label_item = torch.tensor(label_item, dtype = torch.long)
        
        return (temp_item, label_item)

    def __len__(self):
        return self._len
        
    
#define data loaders
train_loader = DataLoader(dataset_pipeline(args.tr_data), 
                      batch_size=args.batch_size, 
                      shuffle=True, 
                      **kwargs)

validation_loader = DataLoader(dataset_pipeline(args.val_data), 
                           batch_size=args.batch_size, 
                           shuffle=False, 
                           **kwargs)


test_loader = DataLoader(dataset_pipeline(args.test_data), 
                           batch_size=args.batch_size, 
                           shuffle=False, 
                       **kwargs)
    
args.dataset_len = len(train_loader)
args.log_step = args.dataset_len // 4

#Define Loss Function - Using CrossEntropy Loss
criterion = nn.CrossEntropyLoss()

def train(model, epoch, versatile=True):
    #set the model to training mode
    model = model.train() 
    train_loss = 0
    accuracy_train = 0

    #load batch data
    for batch_idx, (data,labels) in enumerate(train_loader):
    
        batch_label = labels
        batch_temp = data
        
        if args.cuda:
            batch_temp = batch_temp.cuda()
    
        #clean up the gradients in the optimizer
        optimizer.zero_grad()
    
        temp_output = model(batch_temp)
    
        #CrossEntropy as loss function
        loss = criterion(temp_output, batch_label)
        
        _, predicted = torch.max(temp_output.data, 1)
        
        correct = 0
        total = 0    
        #total number of labels
        total += batch_label.size(0)
           
        correct += (predicted == batch_label).sum()
        
        accuracy = 100 * correct / total
    
        #automatically calculate the backward pass
        loss.backward()
        #perform the actual back-propagation
        optimizer.step()
        
        accuracy_train += accuracy
        
        train_loss += loss.data.item()
    
        #print the training progress 
        if versatile:
            if (batch_idx+1) % args.log_step == 0:
            
                print('| epoch {:3d} | {:5d}/{:5d} batches | Loss {:5.4f} | Accuracy {} |'.format(
                epoch, batch_idx+1, len(train_loader),
                train_loss / (batch_idx+1), accuracy_train/ (batch_idx+1)
                ))

    train_loss /= (batch_idx+1)
    accuracy_train /= (batch_idx+1)
    print('-' * 99)
    print('    | end of training epoch {:3d} | Loss {:5.4f} | Accuracy {} |'.format(
        epoch, train_loss, accuracy_train))

    return train_loss, accuracy_train

def validate(model, epoch):
    #set the model to evaluation mode, this is important if you have BatchNorm in your model!
    model = model.eval()
    validation_loss = 0.
    accuracy_valid = 0
    #load batch data
    for batch_idx, (data,labels) in enumerate(validation_loader):
        
        batch_temp = data
        batch_label = labels
    
        if args.cuda:
            batch_temp = batch_temp.cuda()
    
        #call torch.no_grad() to only calculate the forward pass, save time and memory
        with torch.no_grad():
    
            temp_output = model(batch_temp)
           
        
            loss = criterion(temp_output, batch_label)
    
            validation_loss += loss.data.item()
            
            _, predicted = torch.max(temp_output.data, 1)
           
            correct = 0
            total = 0    
            
            total += batch_label.size(0)
            
            correct += (predicted == batch_label).sum()
            
            accuracy = 100 * correct / total
            
            accuracy_valid += accuracy
    
    validation_loss /= (batch_idx+1)
    accuracy_valid /= (batch_idx+1)
    print('    | end of validation epoch {:3d} | Loss {:5.4f} | Accuracy {} |'.format(
        epoch, validation_loss, accuracy_valid))
    print('-' * 99)

    return validation_loss, accuracy_valid



training_loss = []
validation_loss = []
accuracy_tr = []
accuracy_vl = []
decay_cnt = 0

for epoch in range(1, args.epochs + 1):
    if args.cuda:
        model_CNN.cuda()
    ls_tr, acc_tr = train(model_CNN, epoch)
    ls_val, acc_val = validate(model_CNN, epoch)
    
    training_loss.append(ls_tr)
    validation_loss.append(ls_val)
    accuracy_tr.append(acc_tr)
    accuracy_vl.append(acc_val)
    
    print(f"Epoch: {epoch}")
    if accuracy_tr[-1] == np.max(accuracy_tr):
        print('      Best training model found.')
    if accuracy_vl[-1] == np.max(accuracy_vl):
        #save current best model
        with open(args.model_save, 'wb') as f:
            torch.save(model_CNN.cpu().state_dict(), f)
            print('      Best validation model found and saved.')

    print('-' * 99)
    decay_cnt += 1
   
    #decay when no best training model is found for 3 consecutive epochs
    if np.min(training_loss) not in training_loss[-3:] and decay_cnt >= 3:
        scheduler.step()
        decay_cnt = 0
        print('      Learning rate decreased.')
        print('-' * 99)
    



