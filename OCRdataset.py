import numpy as np

data = np.load('characterfont.npz')

#print(data.files)  # List all arrays stored in the file
#print(data['images'].shape)
#print(data['labels'].shape)
labels = data['labels']
#print(np.unique(labels))
images = data['images']
print(images[0][0][0:10]) #background is pure black and letters are white
shuffle=np.random.permutation(len(labels))
images=images[shuffle]
labels=labels[shuffle]
labels=labels.astype('int64')
if labels.min() >= 65:  # Detects if labels are ASCII (A=65)
    labels = labels - 65
elif labels.min() >= 1: # Detects if labels are 1-indexed (A=1)
    labels = labels - 1

#print(images.min())
#print(images.max())
images =images.astype('float32')/255.0 #NORMALIZE

#yoyo data is smooth no need of filters\data preprocesssing
#import matplotlib.pyplot as plt
#plt.imshow(images[1], cmap='gray')
#plt.show()

#ohhh resnet 18 no greyscale,bhyiiiiii krna pdega ab muje
images=np.expand_dims(images,axis=1)
print(images.shape) 

#sabbb import krlo
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score, confusion_matrix

transform = transforms.Compose([transforms.RandomRotation(10),
                                transforms.RandomAffine
                                (degrees=0,
                                 translate=(0.1,0.1),
                                 scale=(0.9,1.1)),
                                 transforms.GaussianBlur
                                 (kernel_size=3,sigma=(0.1,1.0))])
class OCRDataset(Dataset):

    def __init__(self, images, labels,transform=None):
        self.images = torch.tensor(images)
        self.labels = torch.tensor(labels)
        self.transform= transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image= self.images[idx]
        if self.transform:
            image=self.transform(image)
        return self.images[idx], self.labels[idx]


dataset = OCRDataset(images,labels,transform=transform)
img, label=dataset[0]
print(img.shape)
print(label)

from torch.utils.data import random_split
train_size = int(0.8*len(dataset))
test_size= len(dataset)- train_size
train_dataset, test_dataset=random_split(dataset,[train_size, test_size])
print(len(train_dataset))
print(len(test_dataset))

from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=0)
#num_workers is basically used to divide the cpu to prepare batches for GPU for Training
test_loader= DataLoader(test_dataset,batch_size=32,shuffle=False)

images_batch, labels_batch = next(iter(train_loader))
print(images_batch.shape)
print(labels_batch.shape)

import torchvision
#import torch.nn as nn (Already imported)
model= torchvision.models.resnet18(weights="DEFAULT")
model.conv1=nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)

num_classes=len(np.unique(labels))
print(num_classes)

model.fc = nn.Linear(model.fc.in_features,26)
print(model.fc)

device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model=model.to(device)
print(device)
criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# print(type(images))
# print(type(labels))
num_epochs=5
# print(train_dataset[0][1])
# print(test_dataset[0][1])
# print(len(test_dataset))
# print(len(train_dataset))
from tqdm import tqdm 
# for epoch in range(num_epochs):
#     model.train()
#     running_loss=0.0
#     pbar=tqdm(train_loader,desc=f"Epoch{epoch+1}/{num_epochs}",unit="batch")
#     for images,labels in pbar:
#         images=images.to(device)
#         labels=labels.to(device)
#         optimizer.zero_grad()
#         outputs=model(images)
#         loss=criterion(outputs,labels)
#         loss.backward()
#         optimizer.step()
#         running_loss+=loss.item()
#         pbar.set_postfix(loss=f"{loss.item():.4f}")

#     print(f"Epoch[{epoch+1}/{num_epochs}],Loss:{running_loss/len(train_loader):.4f}")

# #model evaluation
# model.eval()
# correct=0
# total=0
# with torch.no_grad():
#     for images, labels in test_loader:
#         images=images.to(device)
#         labels=labels.to(device)
        
#         outputs=model(images)
#         _, predicted= torch.max(outputs,1)
#         total+=labels.size(0)
#         correct+=(predicted==labels).sum().item()

# accuracy=100*correct/total
# print(f"Accuracy :) - {accuracy:.2f}%")

# # TO SAVE MODEL 
# torch.save(model.state_dict(),"resnet18_ocr.pth")

# LETS TRY MEASURING ACCURACY AFTER EACH EPOCH :)
for epoch in range(num_epochs):
    model.train()
    running_loss=0.0
    pbar=tqdm(train_loader,desc=f"Epoch{epoch+1}/{num_epochs}",unit="batch")
    for images,labels in pbar:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    print(f"Epoch[{epoch+1}/{num_epochs}],Loss:{running_loss/len(train_loader):.4f}")
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for images, labels in test_loader:
            images=images.to(device)
            labels=labels.to(device)
            outputs=model(images)
            _, predicted= torch.max(outputs,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
        
    accuracy=100*correct/total
    print(f"Accuracy :) - {accuracy:.2f}%")
    
   
  
#LETS CHECK EVERYTHING GOING WELL OR NOT
# FINAL TRAINING ACCURACY

model.eval()

# train_correct = 0
# train_total = 0

# with torch.no_grad():
#     for images, labels in train_loader:

#         images = images.to(device)
#         labels = labels.to(device)

#         outputs = model(images)

#         _, predicted = torch.max(outputs, 1)

#         train_total += labels.size(0)
#         train_correct += (predicted == labels).sum().item()

# train_accuracy = 100 * train_correct / train_total

# print(f"Final Train Accuracy: {train_accuracy:.2f}%")

# # FINAL TEST ACCURACY

# test_correct = 0
# test_total = 0

# with torch.no_grad():
#     for images, labels in test_loader:

#         images = images.to(device)
#         labels = labels.to(device)

#         outputs = model(images)

#         _, predicted = torch.max(outputs, 1)

#         test_total += labels.size(0)
#         test_correct += (predicted == labels).sum().item()

# test_accuracy = 100 * test_correct / test_total

# print(f"Final Test Accuracy: {test_accuracy:.2f}%")


# TO SAVE MODEL 
torch.save(model.state_dict(),"resnet18_ocr.pth")
print("MODEL SUCCESSFULLY SAVED")



#SO GUYS 100% ACCURACY IN THE VERY FIRST EPOCH. WE NEED MORE LARGER AND MESSY DATABASES.
#LETS TRY ON MY OWN HANDWRITING IF IT IS ABLE TO DETECT OR NOT
