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

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])


image_path = ('/Users/ravi15_12/Desktop/Screenshot 2026-06-25 at 12.39.00 PM.png')
image = Image.open(image_path).convert("L")
image=image.resize((32,32))
img_np=np.array(image,dtype=np.float32)
img_np=img_np/255.0
#img_np=1.0-img_np
img_np=np.expand_dims(img_np,axis=0)
img_np=np.expand_dims(img_np,axis=0)

input_tensor=torch.from_numpy(img_np).to(device)

#input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

print(f"Predicted Class Index: {predicted.item()}")

with torch.no_grad():
    outputs = model(input_tensor)
    
    # 1. Convert raw model outputs to probabilities (0% to 100%)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    
    # 2. Print out the confidence scores for every single index
    print("\n--- Model Confidence Scores ---")
    for i, prob in enumerate(probabilities):
        print(f"Index {i}: {prob.item():.2f}%")
        
    _, predicted = torch.max(outputs, 1)