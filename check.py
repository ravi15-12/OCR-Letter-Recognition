import numpy as np
from PIL import Image

# 1. Load the raw dataset file (takes 1 second)
data = np.load('characterfont.npz')
images = data['images']
labels = data['labels']

# 2. Adjust labels format exactly like your training code does
labels = labels.astype('int64')
if labels.min() >= 65:
    labels = labels - 65
elif labels.min() >= 1:
    labels = labels - 1

# 3. Grab the very first image and its true label
sample_img = images[0]
sample_lbl = labels[0]

# 4. Save it as a clean digital PNG image on your desktop
# (If your dataset images are already 224x224, this saves it perfectly)
if sample_img.max() <= 1.0:
    sample_img = sample_img * 255.0

sample_np = sample_img.astype('uint8')
img_to_save = Image.fromarray(sample_np, mode='L')
img_to_save.save("/Users/ravi15_12/Desktop/dataset_sample.png")

print("--- DONE ---")
print(f"Saved 'dataset_sample.png' to your desktop.")
print(f"The TRUE expected Class Index for this specific image is: {sample_lbl}")