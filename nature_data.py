#/Users/ssrinivasaraghavan/AI/Deep_Learning/datasets

from tqdm import tqdm
#https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 #to convert image to number
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler #change scale of image
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

#Give the full location of the folder containing different folders of images to be classified
#Example: The Animals folder Contains Cat and Dog folders containg their respective images
DATADIR = r"C:\Users\Ananyaa M\Desktop\PROJECT\nature" #directory where everything is stored

#Enter the folder names containing the images to be classified in the list
CATEGORIES = ["buildings", "forest","glacier","mountain","sea","street"] #folder names to be trained
with open('train_labels.pkl', 'wb') as f:
    pickle.dump(CATEGORIES, f)
with open('train_labels.pkl', 'rb') as f:
    CATEGORIES = pickle.load(f)


#Enter the image dimensions to be processed
IMG_SIZE = 100 #rezized to 100*100

X = [] #list-data
y = [] #list-names

#Read and load the image as ML processable array in X-data and y-labels
for category in tqdm(CATEGORIES):
    path = os.path.join(DATADIR,category)
    for img in tqdm(os.listdir(path)):
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE) #converted to black n white to reduce computation
        img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
        X.append(np.array(img_array))
        y.append(CATEGORIES.index(category))

X = np.array(X)
y = np.array(y)
X = X.reshape(len(X),-1) #len(shape[0])-60000,28,28

#Preprocessing the data to a particular scale. Eg: Values from [0 to 255] will become values to [0 to 1]
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#Training and testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Support Vector Machine Classifier ML Algorithm
cls = SVC(kernel='linear')
cls.fit(X_train,y_train)
print("SVM",cls.score(X_test,y_test))


#Decision Classifier ML Algorithm
tree = DecisionTreeClassifier(criterion = 'entropy')
tree.fit(X, y)
print("Decision Tree",tree.score(X_test,y_test))

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

for i in range(6): #randomly pics six images and gives the name of the image
    ran = np.random.randint(0,200)
    plt.imshow(X_test[ran].reshape(IMG_SIZE,IMG_SIZE),cmap='gray') #cmap-colour mapping
    plt.title(CATEGORIES[int(model.predict(X_test[ran].reshape(1,-1)).astype(int))])
    plt.show()