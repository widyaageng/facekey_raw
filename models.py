import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


######### ADDED METHOD ################
def weights_init(mods):
    if type(mods) == nn.Conv2d:
        torch.nn.init.uniform_(mods.weight, a=-0.1, b=0.1)
    if type(mods) == nn.Linear:
        torch.nn.init.uniform_(mods.weight, a=-0.1, b=0.1)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()        
        
        ch_array = [5,10,20,40,80]
        
        # conv layers
        self.conv1 = nn.Conv2d(1, ch_array[0], 5) # batchsize, 5 ch, 224 - 5 + 1, 224 - 5 + 1 -->pool--> # batchsize, 5 ch, 110, 110
        self.conv2 = nn.Conv2d(ch_array[0], ch_array[1], 5) # batchsize, 10 ch, 110 - 5 + 1, 110 - 5 + 1 -->pool--> # batchsize, 10 ch, 53, 53
        self.conv3 = nn.Conv2d(ch_array[1], ch_array[2], 4) # batchsize, 20 ch, 53 - 4 + 1, 53 - 4 + 1 -->pool--> # batchsize, 20 ch, 25, 25
        self.conv4 = nn.Conv2d(ch_array[2], ch_array[3], 4) # batchsize, 40 ch, 25 - 4 + 1, 25 - 4 + 1 -->pool--> # batchsize, 40 ch, 11, 11
        self.conv5 = nn.Conv2d(ch_array[3], ch_array[4], 2) # batchsize, 80 ch, 11 - 2 + 1, 11 - 2 + 1 -->pool--> # batchsize, 80 ch, 5, 5
        self.pool = nn.MaxPool2d(2,2)
        
        # batchnorm2d layers
        self.bnorm1 = nn.BatchNorm2d(ch_array[0])
        self.bnorm2 = nn.BatchNorm2d(ch_array[1])
        self.bnorm3 = nn.BatchNorm2d(ch_array[2])
        self.bnorm4 = nn.BatchNorm2d(ch_array[3])
        self.bnorm5 = nn.BatchNorm2d(ch_array[4])
        
        # dropout layer
        self.dropout_conv = nn.Dropout(p=0.1)
        self.dropout_fc = nn.Dropout(p=0.2)
        
        # dense layers
        self.fc1 = nn.Linear(ch_array[4]*5*5, 68*8)
        self.fc2 = nn.Linear(68*8, 68*2)
        
        # init-ing net weigths
        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)
        self.conv3.apply(weights_init)
        self.conv4.apply(weights_init)
        self.conv5.apply(weights_init)
        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)

        
    def forward(self, x):
        
        x = self.dropout_conv(self.pool(self.bnorm1(F.relu(self.conv1(x)))))
        x = self.dropout_conv(self.pool(self.bnorm2(F.relu(self.conv2(x)))))
        x = self.dropout_conv(self.pool(self.bnorm3(F.relu(self.conv3(x)))))
        x = self.dropout_conv(self.pool(self.bnorm4(F.relu(self.conv4(x)))))
        x = self.dropout_conv(self.pool(self.bnorm5(F.relu(self.conv5(x)))))
        
        x = x.view(x.shape[0], -1)
        x = self.dropout_fc(self.fc1(x))
        x = self.fc2(x)
        
        
        
        return x