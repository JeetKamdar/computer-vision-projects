#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import cv2
import os


# In[2]:


def convolution(image, operator):
    image_height, image_width = image.shape

    operator_height, operator_width = operator.shape
    
    #center of the filter
    center_i = int((operator_height - 1) / 2) 
    center_j = int((operator_width - 1) / 2)
    
    output = np.zeros((image_height, image_width))
    
    #moving the filter
    for i in range(center_i, image_height - center_i):
        for j in range(center_j, image_width - center_j):
            sum = 0
            #calculate the center value 
            for k in range(-center_i, center_i+1):
                for l in range(-center_j, center_j+1):
                    x = image[i+k, j+l]
                    y = operator[center_i+k, center_j+l]
                    sum = sum + (x * y)
            output[i][j] = sum
    
    return output


# In[3]:


def gradient(grayscale_image):
    prewitt_x = np.array(
        [
            [-1,0,1],
            [-1,0,1],
            [-1,0,1]
        ])

    prewitt_y = np.array(
        [
            [1,1,1],
            [0,0,0],
            [-1,-1,-1]
        ])
    #un-normalised gradient
    Gx = convolution(grayscale_image, prewitt_x)
    Gy = convolution(grayscale_image, prewitt_y)
    
    #normalise
    Gx = np.abs(Gx) / 3
    Gy = np.abs(Gy) / 3
    
    #calculate magnitude
    magnitude = np.sqrt(np.power(Gx, 2) + np.power(Gy, 2))
    #normalise the magnitude
    magnitude = np.rint(magnitude / np.sqrt(2))
    
    return Gx, Gy, magnitude


# In[4]:


def histogram(cell_x, cell_y, magnitude):
    #verifying that both matrices are of the same size
    assert cell_x.shape == cell_y.shape
    
    bins_center = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])
    bins = np.zeros(bins_center.shape[0])
    gradient_angle = np.zeros(cell_x.shape)
    
    for i in range(cell_x.shape[0]):
        for j in range(cell_x.shape[1]):
            #calculate angle
            gradient_angle[i,j] = calc_angle(cell_x[i,j], cell_y[i,j])
            #split the magnitude in appropriate bins
            for k in range(bins.shape[0] - 1):
                if gradient_angle[i,j] < bins_center[k + 1]:
                    bins[k+1] += magnitude[i,j] * abs(1-(bins_center[k+1]-gradient_angle[i,j])/(bins_center[k+1]-bins_center[k]))
                    bins[k] += magnitude[i,j] * abs(1-(gradient_angle[i,j]-bins_center[k])/(bins_center[k+1]-bins_center[k]))
                    break
                    
    return bins


# In[5]:


def calc_angle(dx, dy):
    #calculate th angle within the range (170,-10]
    gradient_angle = np.arctan2(dy, dx) * 180 / np.pi
    if gradient_angle > 170 and gradient_angle < 350:
        gradient_angle -= 180
    elif gradient_angle >= 350:
        gradient_angle -= 360
    
    return gradient_angle


# In[6]:


def normalise_l2(block):
    l2_norm = np.sqrt(np.sum(block * block))
    if l2_norm != 0:
        return block / l2_norm
    else:
        return block


# In[7]:


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# In[8]:


def ReLU(x):
    return x * (x > 0)


# In[9]:


def sigmoid_derivative(x):
    #derivative of sigmoid function with respect to x
    return x * (1.0 - x)


# In[10]:


def ReLU_derivative(x):
    #derivative of the ReLU function, outputs 1 if x is greater than 0, else 0
    np.maximum(x,0,x)
    np.minimum(x,1,x)
    return x


# In[11]:


def shuffle_in_unison(a, b, c):
    #shuffles 2 related arrays in unison
    z = list(zip(a,b,c))
    shuffle_list = np.random.shuffle(z)
    a, b, c = zip(*z)
    return np.array(a), list(b), list(c)


# In[12]:


def HOG(Gx, Gy, magnitude, HOG_descriptor):
    #iterating over the entire image
    #number of overlapping windows = img_size/cell_size - 1
    for i in range(int(magnitude.shape[0]/cell_size) -1):
        for j in range(int(magnitude.shape[1]/cell_size) -1):
            #iterating over the block
            block_vector = np.zeros((0,0))
            for k in range(int(block_size / cell_size)):
                for l in range(int(block_size / cell_size)):
                    #generated a vector for every cell
                    cell_vector = histogram(
                        Gx[(i+k) * cell_size : (i+k+1) * cell_size, (j+l) * cell_size : (j+l+1) * cell_size],
                        Gy[(i+k) * cell_size : (i+k+1) * cell_size, (j+l) * cell_size : (j+l+1) * cell_size],
                        magnitude[(i+k) * cell_size : (i+k+1) * cell_size, (j+l) * cell_size : (j+l+1) * cell_size]
                    )

                    block_vector = np.concatenate((block_vector, cell_vector), axis=None)
            
            normalised_vector = normalise_l2(block_vector)
            HOG_descriptor = np.concatenate((HOG_descriptor, normalised_vector), axis=None)
    return HOG_descriptor


# In[13]:


def train_nn(HOG_descriptor, label, learning_rate, hidden_weights, output_weights):
    #prepare the input, output, and label vectors
    input_vector = HOG_descriptor.T
    bias = np.ones((1, input_vector.shape[1]))
    input_vector = np.concatenate((bias, input_vector),axis=0)
    
    label_vector = np.array([label])
    output_vector = np.zeros(label_vector.shape)
    
    #feedforward
    hidden_nodes = ReLU(np.dot(hidden_weights, input_vector))
    
    bias = np.ones((1, input_vector.shape[1]))
    hidden_nodes = np.concatenate((bias, hidden_nodes),axis=0)
    
    output_nodes = sigmoid(np.dot(output_weights, hidden_nodes))
    
    error = label_vector - output_nodes
    
    #backpropagation
    delta = error * sigmoid_derivative(output_nodes)
    
    output_weights += learning_rate * np.dot(delta, hidden_nodes.T)
    
    hidden_error = np.dot(output_weights.T, error)
    delta = hidden_error * abs(ReLU_derivative(hidden_nodes))
    hidden_weights += learning_rate * np.dot(delta[1:,:], input_vector.T)
    
    return output_nodes, error


# In[14]:


def run_hog(DATA_DIR, DATA):
    label = []
    HOG_descriptor = np.zeros((0,0))
    image_list = []
    counter = 0
    #load the files
    for data in DATA:
        for image in os.listdir('{}/{}'.format(DATA_DIR, data)):
            color_img = cv2.imread('{}/{}/{}'.format(DATA_DIR, data, image), cv2.IMREAD_COLOR)
            grayscale_img = np.dot(color_img[...,:3], [0.299, 0.587, 0.114])
            
            #calculate the gradients and the magnitude
            Gx , Gy, magnitude = gradient(grayscale_img)
            #generate the HOG_descriptor
            hog = HOG(Gx, Gy, magnitude, np.zeros((0,0)))
            if counter == 0:
                HOG_descriptor = np.concatenate((HOG_descriptor, hog), axis=None)
                HOG_descriptor = np.array([HOG_descriptor])
            else:
                hog = np.array([hog])
                HOG_descriptor = np.concatenate((HOG_descriptor, hog), axis=0) 
            
            #generate the label vector
            if 'Positive' in data:
                label.append(1)
            elif 'Neg' in data:
                label.append(0)
            
            image_list.append(image)
            
            counter += 1
    return HOG_descriptor, label, image_list


# In[15]:

#initialise and set the parameters 


DATA_DIR = sys.argv[1]
TRAIN_SETS = ('Train_Positive', 'Train_Negative')

# for HOG
block_size = 16
cell_size = 8

#for the neural network
out_nodes_num = 1
hidden_nodes_num = 250
learning_rate = 0.05

print('Training...')
#call to the HOG function
HOG_descriptor, label, image_list = run_hog(DATA_DIR, TRAIN_SETS)
#shuffle the order so that the positive and negative images are randomised
HOG_descriptor, label, image_list = shuffle_in_unison(HOG_descriptor, label, image_list)

input_nodes_num = HOG_descriptor.shape[1]


# In[25]:

# generate random weights


hidden_weights = np.random.uniform(-0.12,0.12,(hidden_nodes_num, input_nodes_num + 1))
output_weights = np.random.uniform(-0.12,0.12,(out_nodes_num, hidden_nodes_num + 1))


# In[26]:

# train the network


print("Epoch\tMean Squared Error")
ctr = 0
for i in range(200):
    out, error = train_nn(HOG_descriptor, label, learning_rate, hidden_weights, output_weights)
    mse = np.square(error).mean(axis=None)
    print("{}\t{}".format(i, mse))

    if mse <0.0001:
    	if ctr> 5:
    		break
    	else:
    		ctr+=1



# In[27]:

#test code block


TEST_SETS = ('Test_Positive', 'Test_Neg')
print("\nTesting...")
HOG_descriptor, label, image_list = run_hog(DATA_DIR, TEST_SETS)
HOG_descriptor, label, image_list = shuffle_in_unison(HOG_descriptor, label, image_list)
input_vector = HOG_descriptor.T
bias = np.ones((1, input_vector.shape[1]))
input_vector = np.concatenate((bias, input_vector),axis=0)
label_vector = np.array([label]) 
#feedforward
hidden_nodes = ReLU(np.dot(hidden_weights, input_vector))
bias = np.ones((1, input_vector.shape[1]))
hidden_nodes = np.concatenate((bias, hidden_nodes),axis=0)
output_nodes = sigmoid(np.dot(output_weights, hidden_nodes))
error = label_vector - output_nodes


# In[30]:

#print test output
print("\n\n\t\t\t\'1\'is Human, \'0\' is Non-Human\n")
print("Image File\t\t\tExpected_Output\tClassification\tOuput Value")

def classify(x):
	if x == 1:
		return "Human"
	else:
		return "Non-Human"

for i in range(len(label)):
    print('{}\t\t\t{}\t{}\t{}'.format(image_list[i], classify(label[i]), classify(np.rint(output_nodes[0][i])), output_nodes[0][i]))
