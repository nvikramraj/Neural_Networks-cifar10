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

'''
from torchsummary import summary
summary(model, input_size=(3, 32, 32))
'''

print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")

else:
    device = torch.device("cpu")
    print("running on the CPU")

model = Model().to(device)
model.double()


training_data = np.load("training_data2.npy",allow_pickle=True) # loading data set


X = torch.Tensor([i[0] for i in training_data])
y = torch.tensor([np.eye(10)[i[1]] for i in training_data])


validation_per = 0.2 # 20%
val_num = int(len(X)*validation_per)

#seperating data sets for training , validating and testing
train_X = X[:-val_num]
train_y = y[:-val_num]

val_X = X[-val_num:]
val_y = y[-val_num:]


'''
print("Training data : ",len(train_X),len(train_y),train_X.shape)
print("Validation data : ",len(val_X),len(val_y),val_X.shape)
print("Testing data : ",len(test_X),len(test_y),test_X.shape)
'''

#optimization and loss functions

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

def fwd_pass(X,y,train = False):

	X = X.type(torch.DoubleTensor)
	X = X.to(device)
	y = y.to(device)
	#print(X.dtype)
	if train:
		model.zero_grad()
	outputs = model(X)
	check = [torch.argmax(i) == torch.argmax(j) for i,j in zip(outputs,y)]
	acc = check.count(True)/len(check)
	loss = loss_function(outputs,y)

	if train:
		loss.backward()
		optimizer.step()

	return acc,loss

def test(size = 32):
    random_start = np.random.randint(len(val_X)-size)
    X,y = val_X[random_start:random_start+size], val_y[random_start:random_start+size]
    with torch.no_grad():
        val_acc , val_loss = fwd_pass(X, y)
    return val_acc, val_loss	

def train():

    batch_size = 100
    epochs = 10

    with open("model_graph1-10.log","a") as f:
        for epoch in range(epochs):
            
            for i in tqdm(range(0,len(train_X),batch_size)):
                
                batch_X = train_X[i:i+batch_size]
                batch_y = train_y[i:i+batch_size]
                acc,loss = fwd_pass(batch_X , batch_y , train= True)
                
                if i % 50 == 0:
                    val_acc , val_loss = test(size = 100)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")


MODEL_NAME = f"model -{int (time.time())}"
print(MODEL_NAME)
train()

#saving model

save_path = os.path.join("model1-10.pt")
torch.save(model.state_dict(),save_path)
