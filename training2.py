from __future__ import print_function

import numpy as np # linear algebra
import matplotlib.pyplot as plt
#matplotlib inline
import matplotlib.patches as patches
import os
from PIL import Image
import cv2
import warnings
import xmltodict
warnings.filterwarnings("ignore")

import torch
from torchvision import datasets,transforms,models
from torch.utils.data import Dataset,DataLoader


import sys
from statistics import mean
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #in_channels: int, out_channels: int, kernel_size
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


  
    def train(dataset):
       
        train_size=int(len(dataset)*0.7)
        val_size=int(len(dataset)*0.2)
        test_size=len(dataset)-train_size-val_size
        batch_size=4
        trainset,valset,testset=torch.utils.data.random_split(dataset,[train_size,val_size,test_size])
        
        train_loader =DataLoader(dataset=trainset,batch_size=len(trainset),shuffle=True)
        val_loader =DataLoader(dataset=valset,batch_size=len(valset),shuffle=True)
        test_loader =DataLoader(dataset=testset,batch_size=len(testset),shuffle=True)

        dataiter=iter(train_loader) 
        images,labels=dataiter.next()
        images=images.numpy()

        #import pdb; pdb.set_trace()
        val_dataiter=iter(val_loader) 
        val_images,val_labels=val_dataiter.next()
        val_images=val_images.numpy()

        test_dataiter=iter(test_loader) 
        test_images,test_labels=test_dataiter.next()
        test_images=test_images.numpy()

        input_tensor = torch.from_numpy(images)
        output_tensor = labels
        validate_input_tensor = torch.from_numpy(val_images)
        validate_output_tensor = val_labels
        test_input_tensor = torch.from_numpy(test_images)
        test_output_tensor = test_labels
        
        torch.save(input_tensor, 'images_tensor.pt')
        torch.save(output_tensor, 'labels_tensor.pt')
        torch.save(validate_input_tensor, 'val_images_tensor.pt')
        torch.save(validate_output_tensor, 'val_labels_tensor.pt')
        torch.save(test_input_tensor, 'test_images_tensor.pt')
        torch.save(test_output_tensor, 'test_labels_tensor.pt')

        # input_tensor = torch.load('images_tensor.pt')
        # output_tensor = torch.load('labels_tensor.pt')
        # validate_input_tensor = torch.load('val_images_tensor.pt')
        # validate_output_tensor = torch.load('val_labels_tensor.pt')
        # test_input_tensor = torch.load('test_images_tensor.pt')
        # test_output_tensor = torch.load('test_labels_tensor.pt')
        import pdb; pdb.set_trace()
        net = Net()

        criterionCE = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)

        # empty list to store training losses
        train_losses = []
        # empty list to store validation losses
        val_losses = []
        train_accuracies = list()
        val_accuracies = list()
        batch_size = 4
        epoch_stop_train = 203
        train_size = input_tensor.size()[0]//10
        dir_path = "size_"+ str(train_size)
        os.makedirs(dir_path, exist_ok=True) 

        #minibatch calculations
        num_of_batches = input_tensor.size()[0] // batch_size
        if input_tensor.size()[0] % batch_size > 0 :
            last_batch = input_tensor.size()[0] % batch_size 
        else:
            last_batch = 0
        
        for epoch in range(200):  # loop over the dataset multiple times

            print(epoch)
            i =0
            sub_train_accuracies = list()
            while i <= num_of_batches:
                start = (i* batch_size)
                end = (i* batch_size) + batch_size

                #case of last batch
                if i == num_of_batches:
                    if last_batch:
                        start = (i* batch_size) 
                        end = (i* batch_size) + last_batch
                    else:
                        break

                input_tensor_batch = input_tensor[start:end]
                output_tensor_batch = output_tensor[start:end]
                
                i +=1
                #import pdb; pdb.set_trace()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                out_training = net(input_tensor_batch)
                loss_train = criterionCE(out_training, output_tensor_batch)
                
                #append losses
                if i == 1:
                    train_losses.append(loss_train.detach().numpy().mean())

                loss_train.backward()
                optimizer.step()
                #tr_loss = loss_train.item()
                sub_train_accuracies.append(output_tensor_batch.eq(out_training.detach().argmax(dim=1)).float().mean())
                #import pdb; pdb.set_trace()

            train_accuracies.append(sum(sub_train_accuracies) / len(sub_train_accuracies) )
            #validate data    
            out_validation = net(validate_input_tensor)
            loss_validation = criterionCE(out_validation, validate_output_tensor)
            val_losses.append(loss_validation.detach().numpy().mean())
            val_accuracies.append(validate_output_tensor.eq(out_validation.detach().argmax(dim=1)).float().mean())

            #save networks 
            PATH = "./" + dir_path +"/epoch_" + str(epoch) +"_size"+ str(train_size) +"_trained.pth"
            torch.save(net.state_dict(), PATH)
            #if validation error goes up for 5 epochs in a row 
            val_size = len(val_losses)
            #import pdb; pdb.set_trace()
            if epoch >10 and val_losses[val_size-1] > val_losses[val_size-2] > val_losses[val_size-3] > val_losses[val_size-4]:
                epoch_stop_train = epoch
                os.remove("./" + dir_path +"/epoch_" + str(epoch) +"_size"+ str(train_size) +"_trained.pth")
                os.remove("./" + dir_path +"/epoch_" + str(epoch-1) +"_size"+ str(train_size) +"_trained.pth")
                os.remove("./" + dir_path +"/epoch_" + str(epoch-2) +"_size"+ str(train_size) +"_trained.pth")
                os.remove("./" + dir_path +"/epoch_" + str(epoch-3) +"_size"+ str(train_size) +"_trained.pth")
                break

            #import pdb; pdb.set_trace()
            # printing the validation loss
            print('Epoch : ',epoch+1, '\t', 'val loss :', loss_validation)
            print('Epoch : ',epoch+1, '\t', 'train loss :', loss_train)
            print('Epoch : ',epoch+1, '\t', 'val accuracy :', torch.tensor(val_accuracies).mean())
            print('Epoch : ',epoch+1, '\t', 'train accuracy :', torch.tensor(train_accuracies).mean())

        #import pdb; pdb.set_trace()
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.plot(train_accuracies, label='Training accuracies')
        plt.plot(val_accuracies, label='Validation accuracies')
        plt.legend()
        plt.savefig("./" + dir_path +"/size_"+ str(train_size) +"_plot.png")
        print('Finished Training')

        #save trained network to filesystem
        #remove all other files 
        for iteration in range(epoch_stop_train-4):
            if os.path.exists("./" + dir_path +"/epoch_" + str(iteration) +"_size"+ str(train_size) +"_trained.pth"):
                os.remove("./" + dir_path +"/epoch_" + str(iteration) +"_size"+ str(train_size) +"_trained.pth")
            else:
                print("The file does not exist")



#function to visualize images
def draw_bounding_box(input_image):
    with open(path_annotations+input_image[:-4]+".xml") as fd:
        doc=xmltodict.parse(fd.read())
    image=plt.imread(os.path.join(path_images+input_image))
    fig,ax=plt.subplots(1)
    ax.axis("off")
    fig.set_size_inches(10,5)
    temp=doc["annotation"]["object"]
    if type(temp)==list:
        for i in range(len(temp)):
            if temp[i]["name"]=="with_mask":
                a,b,c,d=list(map(int,temp[i]["bndbox"].values()))
                patch=patches.Rectangle((a,b),c-a,d-b,linewidth=1, edgecolor='g',facecolor="none",)
                ax.add_patch(patch)
            if temp[i]["name"]=="without_mask":
                a,b,c,d=list(map(int,temp[i]["bndbox"].values()))     
                patch=patches.Rectangle((a,b),c-a,d-b,linewidth=1, edgecolor='r',facecolor="none",)
                ax.add_patch(patch)
            if temp[i]["name"]=="mask_weared_incorrect":
                a,b,c,d=list(map(int,temp[i]["bndbox"].values()))
                patch=patches.Rectangle((a,b),c-a,d-b,linewidth=1, edgecolor='y',facecolor="none",)
                ax.add_patch(patch)
    else:
        a,b,c,d=list(map(int,temp["bndbox"].values()))
        edgecolor={"with_mask":"g","without_mask":"g","mask_weared_incorrect":"y"}
        patch=patches.Rectangle((a,b),d-b,c-a,linewidth=1, edgecolor=edgecolor[temp["name"]],facecolor="none",)
    ax.imshow(image)
    ax.add_patch(patch)
    plt.show()

#function to make dataset
def make_dataset(no_of_images): 
    image_tensor=[]
    label_tensor=[]
    with_mask_count =0
    without_mask_count =0
    incorrect_mask_count =0
    total_size =100
    #count =0
    for i,j in enumerate(no_of_images):
        # count = count +1
        # if count > 512:
        #     break
        with open(path_annotations+j[:-4]+".xml") as fd:
            doc=xmltodict.parse(fd.read())
        if type(doc["annotation"]["object"])!=list:
            temp=doc["annotation"]["object"]
            a,b,c,d=list(map(int,temp["bndbox"].values()))
            label=options[temp["name"]]
            if label == 0 and with_mask_count < total_size :
                with_mask_count +=1
                image=transforms.functional.crop(Image.open(path_images+j).convert("RGB"), b,a,d-b,c-a)
                image_tensor.append(my_transform(image))
                label_tensor.append(torch.tensor(label))                
            if label == 1 and without_mask_count < total_size :
                without_mask_count +=1
                image=transforms.functional.crop(Image.open(path_images+j).convert("RGB"), b,a,d-b,c-a)
                image_tensor.append(my_transform(image))
                label_tensor.append(torch.tensor(label))
            if label == 2 and incorrect_mask_count < total_size:
                incorrect_mask_count +=1
                image=transforms.functional.crop(Image.open(path_images+j).convert("RGB"), b,a,d-b,c-a)
                image_tensor.append(my_transform(image))
                label_tensor.append(torch.tensor(label))
        else:
            temp=doc["annotation"]["object"]
            for k in range(len(temp)):
                a,b,c,d=list(map(int,temp[k]["bndbox"].values()))
                label=options[temp[k]["name"]]
                if label == 0 and with_mask_count < total_size :
                    with_mask_count +=1
                    image=transforms.functional.crop(Image.open(path_images+j).convert("RGB"), b,a,d-b,c-a)
                    image_tensor.append(my_transform(image))
                    label_tensor.append(torch.tensor(label))                
                if label == 1 and without_mask_count < total_size :
                    without_mask_count +=1
                    image=transforms.functional.crop(Image.open(path_images+j).convert("RGB"), b,a,d-b,c-a)
                    image_tensor.append(my_transform(image))
                    label_tensor.append(torch.tensor(label))
                if label == 2 and incorrect_mask_count < total_size:
                    incorrect_mask_count +=1
                    image=transforms.functional.crop(Image.open(path_images+j).convert("RGB"), b,a,d-b,c-a)
                    image_tensor.append(my_transform(image))
                    label_tensor.append(torch.tensor(label))
                
    final_dataset=[[k,l] for k,l in zip(image_tensor,label_tensor)]
    return tuple(final_dataset)

if __name__ == "__main__":

    path_images="images/"  #path for an image folder 
    path_annotations="annotations/" #path for xmlfiles folder

    imagenames=[] #list of imagefile names
    xmlnames=[] #list of xmlfile names
    # Put images ans xml files into arrays
    for dirname, _, filenames in os.walk(path_annotations):
        for filename in filenames:
            xmlnames.append(filename)

    for dirname, _, filenames in os.walk(path_images):
        for filename in filenames:
            imagenames.append(filename)

    ## Code for finding the total no of labels in our dataset
    listing=[]
    for i in imagenames[:]:
        with open(path_annotations+i[:-4]+".xml") as fd:
            doc=xmltodict.parse(fd.read())
        temp=doc["annotation"]["object"]
        if type(temp)==list:
            for i in range(len(temp)):
                listing.append(temp[i]["name"])
        else:
            listing.append(temp["name"])

    for i in  set(listing):
        print(i)

    # mapping for predictions and analysis purpose
    options={"with_mask":0,"without_mask":1,"mask_weared_incorrect":2} 

    # for i in range(0,5):
    #     draw_bounding_box(imagenames[i])


    #importing neccessary libraries for deeplearning task..

    my_transform=transforms.Compose([transforms.Resize((226,226)),
                                    transforms.ToTensor()])

    #dataset = []
    dataset=make_dataset(imagenames) #making a datset
    
    

    


    # images_tensor = torch.load('images_tensor.pt')
    # labels_tensor = torch.load('labels_tensor.pt')
    # val_images_tensor = torch.load('val_images_tensor.pt')
    # val_labels_tensor = torch.load('val_labels_tensor.pt')

    
    Net.train(dataset)
