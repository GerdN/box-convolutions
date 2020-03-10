#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing frameworks/packages that are required for the model to run

from __future__ import print_function
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import torch
from torch import nn, optim
from torch.autograd import Variable
#from sklearn.model_selection import train_test_split
from box_convolution import BoxConv2d

# # **Data Pre-processing**

# In[ ]:


# Reading the train, test and RUL files (FD001 dataset)

df_train = pd.read_csv("https://raw.githubusercontent.com/sivaji1233/09_turbofan_rul/master/data/train_FD001.txt", sep = ' ', header = None)
df_test = pd.read_csv("https://raw.githubusercontent.com/sivaji1233/09_turbofan_rul/master/data/test_FD001.txt", sep = ' ', header = None)
df_RUL = pd.read_csv("https://raw.githubusercontent.com/sivaji1233/09_turbofan_rul/master/data/RUL_FD001.txt", header = None)


# In[ ]:


# Cleaning the data (2 extra columns were added due to white space, deleting those and adding the column names)
# for easy and clear understanding I'll work on set1 out of 4 and report the results for other sets in the report)

col_list = ['unit', 'time', 'os_1', 'os_2', 'os_3', 'sm_1', 'sm_2', 'sm_3', 'sm_4', 'sm_5', 'sm_6', 'sm_7', 'sm_8', 'sm_9', 'sm_10', 'sm_11', 'sm_12', 'sm_13', 'sm_14', 'sm_15', 'sm_16', 'sm_17', 'sm_18', 'sm_19', 'sm_20', 'sm_21']

df_train = df_train[list(range(26))]
df_train.columns = col_list

df_test = df_test[list(range(26))]
df_test.columns = col_list


# In[4]:


# Reading first 5 rows of the train dataset

df_train.head()


# In[5]:


# Making sure no missing values are there in the data

df_train.info(verbose=True)


# In[6]:


# Plotting the sensor measurements, to see significant changing trend from healty state and failure.

fig, ax = plt.subplots(ncols=4, nrows =6, figsize=(24, 20))
ax = ax.ravel()
for i, item in enumerate(col_list[2:]):
  df_train.groupby('unit').plot(kind='line', x = "time", y = item, ax=ax[i])
  ax[i].get_legend().remove()
  ax[i].title.set_text(item)
plt.subplots_adjust(top = 0.99, bottom = 0.01, hspace = 0.3, wspace = 0.2)
plt.show()


# In[7]:


# From the above figure it is clearly evident that columns ['os3', 'sm1', 'sm5', 'sm10', 'sm16', 'sm18', 'sm19'] are not contributing to significant change from healthy life to failure life, hence omitting the values for further model development

new_col_list = ['unit', 'time', 'os_1', 'os_2', 'sm_2', 'sm_3', 'sm_4', 'sm_6', 'sm_7', 'sm_8', 'sm_9', 'sm_11', 'sm_12', 'sm_13', 'sm_14', 'sm_15', 'sm_17', 'sm_20', 'sm_21']
df_train = df_train[new_col_list]
df_train['cycle'] = df_train['time']
df_test = df_test[new_col_list]
df_test['cycle'] = df_test['time']
df_train.head()


# In[8]:


# Scaling the values

scale_col_list = new_col_list[2:] + ['cycle']
df_train[scale_col_list] = minmax_scale(df_train[scale_col_list])
df_test[scale_col_list] = minmax_scale(df_test[scale_col_list])

df_train.head()


# # **Approach: Using GRU network**

# First the input sequences that are to be fed to the model, shall be of same size and contains 17 features of 'os's and 'sm's data. Inorder to find the seq lenght I'm assuming the size of the least sequence of my test data set.

# In[9]:


# Minimum length of the sequence in test dataset

seq_selected = min(df_test.groupby('unit').max()['time'])
print("The mimium length of the sequence in test dataset is", seq_selected)


# ## Input train sequences:
# 
# The train sequences for input are prepared by using the seq_selected length. For every unit sequences of length seq_selected are selected from the start of the cycle and are stacked in the train dataset. Each sequence is in the dimension of [1 x seq_selected x 18]. This is done for every unit until the max cycle - seq_selected cycle.

# In[10]:


# Preparation of inputs for training

max_list = list(df_train.groupby('unit').max()['time'])     # creating a list of max cycles in the training dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     # for converting the GPU supported tensors in case of using GPU

i = 0
X_train = Variable(torch.Tensor([]).float()).to(device)
for item in max_list:
  temp_list = Variable(torch.Tensor([]).float()).to(device)
  for j in range(item - seq_selected):
      zero_list = Variable(torch.Tensor(df_train.values[i+j:i+j+seq_selected, 2:]).float()).to(device)
      temp_list = torch.cat((temp_list, zero_list.view(1, seq_selected, 18)), dim=0)
  i += item
  X_train = torch.cat((X_train, temp_list), dim=0)

print("The shape of input data for training model, X_train is", X_train.shape)


# In[11]:


# Preparation of labels for training

y_train = []
for item in max_list:
  y_train.extend(list(range(item-seq_selected))[::-1])
y_train = Variable(torch.Tensor(y_train).float()).to(device)
y_train = y_train.view(X_train.shape[0], 1)

print("The shape of labels data for training model, y_train is", y_train.shape)


# In[ ]:


# Model parameters

batch_size = 250
# learning_rate = 0.00001
num_epochs = 100
hidden_dim = 50
hidden_dim2 = 25


# ## **Model**

# In[13]:

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = BoxConv2d(1, 40, 14, 30)
        self.conv1_1x1 = nn.Conv2d(30, 14, 1, 1)

        self.fc1 = nn.Linear(7*7*40, 10)
#   def __getitem__(self, index):
#        # stuff
#        return (data, RUL)
    
    def forward(self, x):
        # The following line computes responses to 40 "generalized Haar filters"
        x = self.conv1_1x1(self.conv1(x))
        x = F.relu(F.max_pool2d(x, 4))


#class lstm(nn.Module):
#    def __init__(self, hidden_dim, hidden_dim2):
#        super(lstm, self).__init__()
#        self.hidden_dim = hidden_dim
#        self.hidden_dim2 = hidden_dim2
#        self.lstm = nn.GRU(18, hidden_dim, dropout = 0.2, batch_first = True)
#        self.lstm2 = nn.GRU(hidden_dim, hidden_dim2, dropout = 0.2, batch_first=True)
#        self.linear = nn.Linear(hidden_dim2, 1)

#    def forward(self, x):
#        out, _ = self.lstm(x)
#        out, _ = self.lstm2(out)
#        out = out[:, -1, :]
#        out = self.linear(out)
#        return out

model = Net.to(device)
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
# optimizer = optim.Adam(model.parameters())
optimizer = torch.optim.RMSprop(model.parameters())


# ## **Training**

# In[15]:


# training the model

train_loss = []
val_loss = []
epochs = []
for epoch in range(num_epochs):
  permutation = torch.randperm(X_train.size()[0])
  val_permutation = permutation[:X_train.size()[0]//3]
  train_permutation = permutation[X_train.size()[0]//3:]
  
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

  if epoch % 10 == 0:
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
plt.show()


# In[ ]:


torch.save(model.state_dict(), 'model_weights.pth')


# ## **Prediction**

# In[16]:


# Preparing Test data

max_list_test = list(df_test.groupby('unit').max()['time'])

i = 0
X_test = Variable(torch.Tensor([]).float()).to(device)
for item in max_list_test:
  temp_list = Variable(torch.Tensor([]).float()).to(device)
  zero_list = Variable(torch.Tensor(df_test.values[i + item - seq_selected : i + item, 2:]).float()).to(device)
  temp_list = torch.cat((temp_list, zero_list.view(1, seq_selected, 18)), dim=0)
  i += item
  X_test = torch.cat((X_test, temp_list), dim=0)

print(X_test.shape)


# In[ ]:


# Preparing test labels

y_test = df_RUL.values
y_test = Variable(torch.Tensor(y_test).float()).to(device)


# In[18]:


# Predicting loss on the test dataset

y_pred = model(X_test)
test_loss = criterion(y_pred, y_test)

print("The loss on test data is", test_loss.item())


# In[19]:


# Plotting predicted vs actual RULs

plt.figure(figsize=(15,10))
plt.plot(list(range(1,101)), y_pred.view(100).tolist(), label='RUL_predicted')
plt.plot(list(range(1,101)), y_test.view(100).tolist(), label="RUL_actual")
plt.title("Model results")
plt.xlabel("Unit")
plt.ylabel("Cycles")
plt.legend()
plt.show()


# # **2.4 Conclusion**
# 
# Following are a few points that are concluded. 
# 1. Further fine-tuning of the hyperparameters and exploring various architectures are necessary for enhancing the performance of the model on test set. 
# 2. One of the reasons for high loss on test set is because of the less considered sequence length, increasing sequence length to 50 improved the performance, howerer 3 engine IDs don’t have cycles more than 50. 
# 3. Dealing with the noise, taking measures to reduce may also help the models performance.
# 

# In[ ]:


#pip freeze > requirements.txt


# In[ ]:




