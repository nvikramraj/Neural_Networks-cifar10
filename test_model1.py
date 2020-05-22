import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

#Neural Network
# 2 conv layer with batch norm , 3 linear layer 
# Adam optimizer and MSELoss  , one hot vector - labels

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.bn2 = torch.nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward (self,x):
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x,dim =1) #using activation function at output to get % or 0-1 values

model = Model()
save_path = os.path.join("model2-10.pt")
model.load_state_dict(torch.load(save_path))
model.eval()
model.double()

test_data = np.load("test_data2.npy",allow_pickle=True) # loading data set
test_X = torch.tensor([i[0] for i in test_data])
test_y = torch.tensor([np.eye(10)[i[1]] for i in test_data])

#print(test_X[0])
batch_size = 100
acc = 0
label = { "aeroplane":0,"automobile":0,"bird":0,"cat":0,"deer":0,
            "dog":0,"frog":0,"horse":0,"ship":0,"truck":0 }
for i in tqdm(range(0,len(test_X),batch_size)):

    batch_X = test_X[i:i+batch_size].view(-1,3,32,32)
    batch_y = test_y[i:i+batch_size]
    batch_X = batch_X.type(torch.DoubleTensor)
    output = model(batch_X)
    for i,j in zip(output,batch_y):
        x = torch.argmax(i)
        y = torch.argmax(j)
        if x == y :
            acc += 1
            if y == 0:
                label["aeroplane"] += 1
            elif y == 1:
                label["automobile"] += 1
            elif y == 2:
                label["bird"] += 1
            elif y == 3:
                label["cat"] += 1
            elif y == 4:
                label["deer"] += 1
            elif y == 5:
                label["dog"] += 1
            elif y == 6:
                label["frog"] += 1
            elif y == 7:
                label["horse"] += 1
            elif y == 8:
                label["ship"] += 1
            elif y == 9:
                label["truck"] += 1

total_accuracy = acc/len(test_X) *100
print("Total accuracy : ",total_accuracy)
#Getting accuracy of each element
for i in label:
    label[i] = label[i]/1000 *100
    print(f" {i} : {label[i]} ")

#checking for last 10 images

pic = test_X[-10:]
prediction = output[-10:]
titles = { 0:"aeroplane",1:"automobile",2:"bird",3:"cat",4:"deer",
            5:"dog",6:"frog",7:"horse",8:"ship",9:"truck" }
c = 1
for i in range(10):
    x = pic[i].numpy() #plotting the images
    y = torch.argmax(prediction[i]).tolist()
    image = cv2.merge((x[2],x[1],x[0]))
    plt.subplot(2,5,c)
    plt.axis("off")
    plt.title(titles[y])
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    c += 1
plt.show()

#X = test_X[0].view(-1,3,32,32)
#X = X.type(torch.DoubleTensor)
#print(X.dtype)
#output = model(X)

#print(torch.argmax(output) , torch.argmax(test_y[0]))
