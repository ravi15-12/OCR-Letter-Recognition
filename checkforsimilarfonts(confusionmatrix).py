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

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score, confusion_matrix

class OCRDataset(Dataset):

    def __init__(self, images, labels):
        self.images = torch.tensor(images)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


dataset = OCRDataset(images,labels)

from torch.utils.data import random_split
train_size = int(0.8*len(dataset))
test_size= len(dataset)- train_size
train_dataset, test_dataset=random_split(dataset,[train_size, test_size])

from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader= DataLoader(test_dataset,batch_size=32,shuffle=False)

import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = models.resnet18()
model.conv1=nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
model.fc = nn.Linear(model.fc.in_features, 26)
model.load_state_dict(torch.load("resnet18_ocr.pth", map_location=device))
model.to(device)
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

#from sklearn.metrics import confusion_matrix.  (PEHLE SE IMPORTED HAIIII)

cm = confusion_matrix(all_labels, all_predictions)

print(cm.shape)
# THIS ONE WAS FOR CHECKING FOR EACH CLASS CONFUSION
# for i in range(26):
#     for j in range(26):
#         if i != j and cm[i,j] > 0:
#             print(
#                 f"{chr(i+65)} -> {chr(j+65)} : {cm[i,j]}"
#             )

#FOR TOP 20 BIG ERRORS

# errors=[]
# for i in range(26):
#     for j in range(26):
#         if i != j and cm[i,j] > 0:
#             errors.append((cm[i,j],chr(i+65),chr(j+65)))
# errors.sort(reverse=True)
# print("\n--- Top 20 Confusion Errors ---")
# for count, actual, predicted in errors[:20]:
#     print(f"{actual} -> {predicted} : {count}")

import matplotlib.pyplot as plt

model.eval()

wrong_images = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                wrong_images.append(images[i].cpu())

        if len(wrong_images) >= 10:
            break

# Display 10 wrong predictions
for i, img in enumerate(wrong_images[:10]):
    img = img.permute(1, 2, 0)   # CHW -> HWC

    plt.figure(figsize=(2,2))
    plt.imshow(img)
    plt.title("Wrong prediction")
    plt.axis("off")
    plt.show()



#FOR GRAPH METHOD (ANALYZE CONFUSION BETWEEN TEXTS FOR GRAPH)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(12,12))
# plt.imshow(cm)
# plt.colorbar()

# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("OCR Confusion Matrix")

# letters = [chr(i + 65) for i in range(26)]

# plt.xticks(range(26), letters)
# plt.yticks(range(26), letters)

# plt.show()