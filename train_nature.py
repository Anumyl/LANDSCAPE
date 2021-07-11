
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

#Enter the folder names containing the images to be classified in the list
CATEGORIES = ["buildings", "forest","glacier","mountain","sea","street"]
with open('train_labels.pkl', 'wb') as f:
    pickle.dump(CATEGORIES, f)

#Enter the image dimensions to be processed
IMG_SIZE = 100

X = []
y = []

#Read and load the image as ML processable array in X-data and y-labels
for category in tqdm(CATEGORIES):
    path = os.path.join(DATADIR,category)
    for img in tqdm(os.listdir(path)):
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
        X.append(np.array(img_array))
        y.append(CATEGORIES.index(category))

X = np.array(X)
y = np.array(y)
X = X.reshape(len(X),-1)

#Preprocessing the data to a particular scale. Eg: Values from [0 to 255] will become values to [0 to 1]
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#Training and testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)


#Random Forest Classifier ML Algorithm
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
model = model.fit(X, y)

#Save Classifier file
pickle_out = open("RFC_trained.pickle","wb")
pickle.dump(model,pickle_out)

#Load Classifier File
pickle_in = open("RFC_trained.pickle","rb")
model = pickle.load(pickle_in)
print("Random Forest", model.score(X_test,y_test))
