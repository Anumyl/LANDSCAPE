
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

#Give the full location of the folder containing different folders of images to be classified
#Example: The Animals folder Contains Cat and Dog folders containg their respective images
DATADIR = r"C:\Users\Ananyaa M\Desktop\PROJECT\nature"

#Loading Labels
with open('train_labels.pkl', 'rb') as f:
    CATEGORIES = pickle.load(f)

TEST_FOLDER_NAME = ["test"]
#Enter the image dimensions to be processed
IMG_SIZE = 100

X = []

#Read and load the test images
for category in tqdm(TEST_FOLDER_NAME):
    path = os.path.join(DATADIR,category)
    for img in tqdm(os.listdir(path)):
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
        X.append(np.array(img_array))

X = np.array(X)
X = X.reshape(len(X),-1)

#Preprocessing the data to a particular scale. Eg: Values from [0 to 255] will become values to [0 to 1]
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


#Load Classifier File
pickle_in = open("RFC_trained.pickle","rb")
model = pickle.load(pickle_in)

#Display classified images with their prediction
for i in range(X.shape[0]):
    ran = i
    plt.imshow(X[ran].reshape(IMG_SIZE,IMG_SIZE),cmap='gray')
    plt.title(CATEGORIES[int(model.predict(X[ran].reshape(1,-1)).astype(int))])
    plt.show()