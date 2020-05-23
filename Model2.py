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
# 3 conv layer with batchnorm , 1 linear layer 
# Adam optimizer and MSELoss  , one hot vector - labels

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1 = nn.Conv2d(3,32,3) 
        self.bn2 = torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(32,64,3)
        self.bn3 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(64,128,3)

        x = torch.rand(3,32,32).view(-1,3,32,32)
        self._to_linear = None 
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear,512)
        self.fc2 = nn.Linear(512,10) 

    def convs(self,x):
        x = self.bn1(x)
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = self.bn2(x)
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = self.bn3(x)
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))
        #print(x.shape)

        if self._to_linear is None:

            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2] # to get the size of 1-d or flattened img
            #print(self._to_linear,x.shape)

        return x

    def forward (self,x):
        x = self.convs(x)
        x = x.view(-1,self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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

test_X = torch.tensor([i[0] for i in test_data])
test_y = torch.tensor([np.eye(10)[i[1]] for i in test_data])
'''
print("Training data : ",len(train_X),len(train_y),train_X.shape)
print("Validation data : ",len(val_X),len(val_y),val_X.shape)
print("Testing data : ",len(test_X),len(test_y),test_X.shape)
'''

#optimization and loss functions

optimizer = optim.Adam(model.parameters() ,lr = 0.001)
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

    with open("model_graph2-10.log","a") as f:
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

save_path = os.path.join("model2-10.pt")
torch.save(model.state_dict(),save_path)
