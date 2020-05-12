# frameworks/packages
#This script applies a deep convolutional network including box-convolution to the C-Mapss Turbofan dataset (FD001)
#Module box_convolution is from shrubb/box-convolutions
#Required packages: libgcc, pyqt, git, pytorch, torchvision, C Compiler, OpenCV, requests, gxx_linux-64, scikit-learn
#Tested on Ubuntu 18.04, Pytorch 1.4
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from box_convolution import BoxConv2d
import matplotlib
# # **Data Pre-processing**
# train, test and RUL files (FD001 dataset)
df_train = pd.read_csv("FD001train_V2.txt", sep = ' ', header = None, skipinitialspace=True)
df_test = pd.read_csv("FD001test_V2.txt", sep = ' ', header = None, skipinitialspace=True)
df_RUL = pd.read_csv("RUL_FD001.txt", header = None)

# Cleaning 
col_list = ['unit', 'time', 'os_1', 'os_2', 'os_3', 'sm_1', 'sm_2', 'sm_3', 'sm_4', 'sm_5', 'sm_6', 'sm_7', 'sm_8', 'sm_9', 'sm_10', 'sm_11', 'sm_12', 'sm_13', 'sm_14', 'sm_15', 'sm_16', 'sm_17', 'sm_18', 'sm_19', 'sm_20', 'sm_21']
df_train = df_train[list(range(26))]
df_train.columns = col_list

df_test = df_test[list(range(26))]
df_test.columns = col_list


# missings?
df_train.info(verbose=True)
# Plotting sensor measurements for visual inspection
#fig, ax = plt.subplots(ncols=4, nrows =6, figsize=(24, 20))
#ax = ax.ravel()
#for i, item in enumerate(col_list[2:]):
#  df_train.groupby('unit').plot(kind='line', x = "time", y = item, ax=ax[i])
#  ax[i].get_legend().remove()
#  ax[i].title.set_text(item)
#plt.subplots_adjust(top = 0.99, bottom = 0.01, hspace = 0.3, wspace = 0.2)
#plt.show()


# delete ['os3', 'sm1', 'sm5', 'sm10', 'sm16', 'sm18', 'sm19'] 

new_col_list = ['unit', 'time', 'sm_2', 'sm_3', 'sm_4', 'sm_7', 'sm_8', 'sm_9', 'sm_11', 'sm_12', 'sm_13', 'sm_14', 'sm_15', 'sm_17', 'sm_20', 'sm_21']
df_train = df_train[new_col_list]
df_test = df_test[new_col_list]
print(df_train.shape)
print(df_test.shape)
print(df_train.head())
print(df_test.head())

# Scaling
scale_col_list = new_col_list[2:]
df_train[scale_col_list] = minmax_scale(df_train[scale_col_list])
df_test[scale_col_list] = minmax_scale(df_test[scale_col_list])
print(df_train.head())
print(df_test.head())

# Sequences and network
# Input sequences, find max seq test length
seq_selected = max(df_test.groupby('unit').max()['time'])
print("maximum length of test sequence", seq_selected)
#seq_selected may be used to dilberately chosse length of windows/sequences
seq_selected = 30

#padding
#df_train_padded.iloc = pad_sequence(df_train, batch_first=True, padding_value=0)
#df_test_padded.iloc = pad_sequence(df_test, batch_first=True, padding_value=0)
# ## Input train sequences:
# The train sequences for input are prepared by using the seq_selected length. For every unit sequences of length seq_selected are selected from the start of the cycle and are stacked in the train dataset. Each sequence is in the dimension of [1 x seq_selected x 18]. This is done for every unit until the max cycle - seq_selected cycle.


# Preparing inputs
max_list = list(df_train.groupby('unit').max()['time'])     # creating a list of max cycles in the training dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     # for converting the GPU supported tensors in case of using GPU

i = 0
X_train = Variable(torch.Tensor([]).float()).to(device)
for item in max_list:
  temp_list = Variable(torch.Tensor([]).float()).to(device)
  for j in range(item - seq_selected):
      zero_list = Variable(torch.Tensor(df_train.values[i+j:i+j+seq_selected, 2:]).float()).to(device)
      temp_list = torch.cat((temp_list, zero_list.view(1, seq_selected, 14)), dim=0)
  i += item
  X_train = torch.cat((X_train, temp_list), dim=0)
X_train=X_train.unsqueeze(1)
print("input shape for training model is", X_train.shape)


# Preparing labels
y_train = []
for item in max_list:
  y_train.extend(list(range(item-seq_selected))[::-1])
y_train = Variable(torch.Tensor(y_train).float()).to(device)
y_train = y_train.view(X_train.shape[0], 1)

print("labels shape", y_train.shape)

# Model
batch_size =512
learning_rate = 0.0001
num_epochs = 15

class Box5DCNN(nn.Module):
    def __init__(self):
        super(Box5DCNN, self).__init__()
        self.conv0 = nn.Conv2d(1, 10, 9, padding=4)
        #nn.init.kaiming_normal_(self.conv0.weight.data, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.conv0.weight)
        self.conv1 = nn.Conv2d(10, 10, 9, padding=4)
        #nn.init.kaiming_normal_(self.conv1.weight.data, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv5 = nn.Conv2d(10, 1, 3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        #nn.init.kaiming_normal_(self.conv5.weight.data, nonlinearity='relu')
        self.conv2 = BoxConv2d(1, 10, 14, seq_selected)
        self.conv1_1x1 = nn.Conv2d(10, 10, 1, 1)
        torch.nn.init.xavier_uniform_(self.conv1_1x1.weight)
        #nn.init.kaiming_normal_(self.conv1_1x1.weight.data, nonlinearity='relu')
        self.dropout = nn.Dropout(p=.5)
        self.fc1 = nn.Linear(14*seq_selected*1, 100) #height*width*channels from preceding layer
        #nn.init.kaiming_normal_(self.fc1.weight.data, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(100, 1)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        #self.fc2.bias.requires_grad = False

    def forward(self, x):
        #x = self.conv0(x)
        #print(x.shape)
        #x = F.relu(self.conv1(x))
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.conv1_1x1(x)
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        #x = self.conv1(x)
        #print(x.shape)
        #x = self.conv1(x)
        # print(x.shape)
        #x = self.conv1(x)
        #x = F.relu(self.conv2(x))
        #print(x.shape)
        #x = F.relu(self.conv1_1x1(x))
        #x = self.dropout(x)
        #print(x.shape)
        x = F.relu(self.conv5(x))
        #print(x.shape)
        #x = F.relu(self.conv2(x))
        #print(x.shape)
        #x = F.relu(self.conv1_1x1(x))
        #print(x.shape)
        #x = F.relu(F.max_pool2d(x, 4))
        x = x.view(-1, 14*seq_selected*1)
        #print(x.shape)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        #print(x.shape)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        #print(x.shape)
        return x

model = Box5DCNN().to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
# optimizer = optim.Adam(model.parameters())
optimizer = torch.optim.Adam(model.parameters())

from torchsummary import summary
summary(model, input_size=(1, seq_selected, 14))

#Training
train_loss = []
val_loss = []
epochs = []
for epoch in range(num_epochs):
  permutation = torch.randperm(X_train.size()[0])
  val_permutation = permutation[:X_train.size()[0]//30]
  train_permutation = permutation[X_train.size()[0]//30:]
  
  for i in range(0, len(train_permutation), batch_size):

    # Training model
    optimizer.zero_grad()
    indices = train_permutation[i:i+batch_size]
    batch_x = X_train[indices]
    out = model(batch_x)
    loss = criterion(out, y_train[indices])

    # Validation
    val_predict = model(X_train[val_permutation])
    val_loss_cal = criterion(val_predict, y_train[val_permutation])

    loss.backward()
    optimizer.step()

  if epoch % 2 == 0:
    train_loss.append(loss.item())
    val_loss.append(val_loss_cal.item())
    epochs.append(epoch)
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))






plt.plot(epochs, train_loss, label = "training_loss")
plt.plot(epochs, val_loss, label = "validation_loss")
plt.title("Training Curve (batch_size={})".format(batch_size))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
#plt.show(block=True)


torch.save(model.state_dict(), 'model_weights.pth')

#Prediction
# Preparing test data
max_list_test = list(df_test.groupby('unit').max()['time'])

i = 0
X_test = Variable(torch.Tensor([]).float()).to(device)
for item in max_list_test:
  temp_list = Variable(torch.Tensor([]).float()).to(device)
#  zero_list = Variable(torch.Tensor(df_test.values[i + item - seq_selected : i + item, 2:]).float()).to(device)
  zero_list = Variable(torch.Tensor(df_test.values[i + item - seq_selected : i + item, 2:]).float()).to(device)
  temp_list = torch.cat((temp_list, zero_list.view(1, seq_selected, 14)), dim=0)
  i += item
  X_test = torch.cat((X_test, temp_list), dim=0)
X_test=X_test.unsqueeze(1)
print(X_test.shape)

# Preparing test labels
y_test = df_RUL.values
y_test = Variable(torch.Tensor(y_test).float()).to(device)
print(y_test.shape)

# Predicting loss on the test dataset
y_pred = model(X_test)
test_loss = criterion(y_pred, y_test)
print("The loss on test data is", test_loss.item())


# Plotting predicted vs actual RULs
plt.figure(figsize=(15,10))
plt.plot(list(range(1,101)), y_pred.view(100).tolist(), label='RUL_predicted')
plt.plot(list(range(1,101)), y_test.view(100).tolist(), label="RUL_actual")
plt.title("Model results")
plt.xlabel("Unit")
plt.ylabel("Cycles")
plt.legend()
plt.show(block=True)



#Sources
#https://github.com/shrubb/box-convolutions
#https://github.com/sivaji1233/09_turbofan_rul

