import numpy as np

# Load the dataset
data = np.load("characterfont.npz")
labels = data['labels']

print("Dataset Total Samples:", len(labels))
print("First 20 labels raw data:", labels[:20])
print("Unique labels present:", np.unique(labels))

# Let's see if the labels are strings like ['A', 'B'] or integers like [0, 1]
print("Label Data Type:", labels.dtype)