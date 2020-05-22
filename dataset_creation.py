import pickle
import os
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def getset(label,data):

	data_set_test = []
	for y,x in zip(label,data):
		X_r = np.reshape(x[:1024],(32,32))
		X_g = np.reshape(x[1024:2048],(32,32))
		X_b = np.reshape(x[2048:],(32,32)) #splitting the rgb elements
		X = np.stack((X_r,X_g,X_b),0)# stacking r , g ,b in 3-d 
		data_set_test.append([X,y])
	return data_set_test

#getting raw data from the files
data = []
for i in range(1,6):
	name = "cifar-10-python/cifar-10-batches-py/data_batch_"+ str(i)
	path = os.path.join(name)
	data.append(unpickle(path))

labels_dict = []
for i in data[0] :
	labels_dict.append(i) #getting labels from the dictionary

data_set = []
for i in range(5):
	data_set.append(getset(data[i][labels_dict[1]],data[i][labels_dict[2]]))
#splitting the numpy data and labels from each batch

'''
print("No of batches",len(data_set))
print("No of pictures in a batch ",len(data_set[0]))
print("data, label",len(data_set[0][0]))
print("Label of first pic",data_set[0][0][1])
print("Data of first pic [r,g,b]",data_set[0][0][0])
'''

training_set =[]
for i in range(len(data_set)):
	for j in range(len(data_set[i])):
		training_set.append(data_set[i][j])

print("No of pictures in training set :",len(training_set))

name = "cifar-10-python/cifar-10-batches-py/test_batch"
path = os.path.join(name)
test = unpickle(path)
# the image is of shape 3,32,32 
test_set = getset(test[labels_dict[1]],test[labels_dict[2]])
print("No of pictures in test set :",len(test_set))

np.save("training_data2.npy",training_set) #saving it

np.save("test_data2.npy",test_set) #saving it
'''
name = "cifar-10-python/cifar-10-batches-py/batches.meta"
path = os.path.join(name)
batch_names = unpickle(path)

print(batch_names)

'''

'''
1st index batch
2nd index pic
3rd index data , label
'''