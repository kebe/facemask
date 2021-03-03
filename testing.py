# Python program to test network on a dataset saved in the same format as the CIFAR batches  

#importing files for a basic NN setup
from __future__ import print_function
import torch
import numpy as np
import sys
from statistics import mean
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 1200)  #sqrt(179776/4/16)
        self.fc2 = nn.Linear(1200, 300)
        self.fc3 = nn.Linear(300, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #import pdb; pdb.set_trace()
        x = x.view(x.size(0), 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def test(input_tensor, output_tensor, PATH):
       
        
        net = Net()
        net.load_state_dict(torch.load(PATH))

        out = net(input_tensor)
        accuracy = output_tensor.eq(out.detach().argmax(dim=1)).float().mean()
        print('accuracy:', accuracy.item())

        print(output_tensor)
        print(out.detach().argmax(dim=1))

        #import pdb; pdb.set_trace()
        #image = input_tensor[25];npimg = image.numpy();plt.imshow(np.transpose(npimg, (1, 2, 0)));plt.show()
        #image = input_tensor[21];npimg = image.numpy();plt.imshow(np.transpose(npimg, (1, 2, 0)));plt.show()
        #image = input_tensor[8];npimg = image.numpy();plt.imshow(np.transpose(npimg, (1, 2, 0)));plt.show()



#Test Harness main method
if __name__ == "__main__":

    # # CIFAR BATCHES
    #Input arguments: network file name and data file name
    PATH = "output/epoch_25_trained.pth"
  
    test_input_tensor = torch.load('test_images_tensor.pt')
    test_output_tensor = torch.load('test_labels_tensor.pt')

    Net.test(test_input_tensor, test_output_tensor, PATH)