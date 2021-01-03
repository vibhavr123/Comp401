
# coding: utf-8

# In[5]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# In[6]:


device = torch.device('cuda') 


# In[10]:


class Comp401Model(nn.Module):
    def __init__(self, sequence_length, n_targets):
        super(Comp401Model, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size = pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(320),
            nn.Dropout(p=0.2))
            
        reduce_by = 2 * (conv_kernel_size - 1)
        pool_kernel_size = float(pool_kernel_size)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.classifier = nn.Sequential(
            nn.Linear(320 * self._n_channels, n_targets),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_targets),
            nn.Linear(n_targets, n_targets),
            nn.Sigmoid())

    def forward(self, x):
        """
        Forward propagation of a batch.
        """
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 320 * self._n_channels)
        predict = self.classifier(reshape_out)
        return predict
            
            


# In[13]:


def criterion():
 return nn.MSELoss()



# In[14]:


def get_optimizer(lr):
   return (torch.optim.SGD,
            {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})

