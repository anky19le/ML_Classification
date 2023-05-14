import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1 / (1 + np.exp(-z))  # your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    train_data = np.empty((0,784))
    train_label = []
    validation_data = np.empty((0,784))
    validation_label = []
    test_data = np.empty((0,784))
    test_label = []
    n_mats=0
    for key in mat:
      if key.startswith("train"):
        n_mats+=1

    for i in range(int(n_mats)):
      #test data
      test_data = np.vstack([test_data, np.double(np.array(mat['test'+str(i)]))/255])   #convert to double and regularisation
      test_label = np.hstack([test_label, np.full(shape=np.array(mat['test'+str(i)]).shape[0],fill_value=i,dtype=np.int)])

      #training data and validation data
      train_mat = mat['train'+str(i)]
      random_ind = np.random.choice(train_mat.shape[0], size=train_mat.shape[0], replace=False)
      train_data = np.vstack([train_data, np.double(train_mat[random_ind[1000:],:])/255])
      train_label = np.hstack([train_label, np.full(shape = (len(random_ind) - 1000),fill_value=i,dtype=np.int)])
      validation_label = np.hstack([validation_label, np.full(shape=1000,fill_value=i,dtype=np.int)])
      validation_data = np.vstack([validation_data, np.double(train_mat[random_ind[0:1000],:])/255])
    
    test_label = test_label.reshape((test_label.shape[0], 1))
    test_label = test_label.astype(int)
    train_label = train_label.reshape((train_label.shape[0], 1))
    train_label = train_label.astype(int)
    validation_label = validation_label.reshape((validation_label.shape[0], 1))
    validation_label = validation_label.astype(int)

    # Feature selection
    # Your code here.
    alldata = np.vstack([train_data, validation_data])
    delete_ind = np.array(np.where(np.var(alldata,axis=0) < 0.1))
    keep_ind = np.array(np.where(np.var(alldata,axis=0) >= 0.1))
    selected_feature = keep_ind
    train_data = np.delete(train_data,delete_ind, axis=1)
    validation_data = np.delete(validation_data,delete_ind, axis =1)
    test_data = np.delete(test_data,delete_ind, axis=1)
    print('preprocess done')
    #print(type(train_label[0,0]))
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    #print(type(training_label[0,0]))
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    # Your code here
    #
    #
    #
    #
    #
    bias1 = np.full(shape=(np.array(training_data).shape[0],1),fill_value=1,dtype=np.float)
    input_bias = np.append(training_data, bias1, axis=1)
    hidden = sigmoid(np.matmul(input_bias, w1.transpose()))
    bias2 = np.full(shape=(np.array(hidden).shape[0],1),fill_value=1,dtype=np.float)
    hidden_bias = np.append(hidden, bias2, axis=1)
    output = sigmoid(np.matmul(hidden_bias, w2.transpose()))
    
    #one hot encoding
    y = np.zeros(shape= output.shape)
    for i in range(training_label.shape[0]):
      y[i, training_label[i][0]] = 1.0

    #Cost function
    J= -np.sum(y*np.log(output) + ((1-y)*np.log(1-output)))/training_label.shape[0]
    
    #Gradients for weights
    delta = output - y
    grad_w2 = np.dot(delta.transpose(),hidden_bias)
    a = (1-hidden)*hidden
    b = np.matmul(delta,w2[:,:n_hidden])
    c = a*b
    grad_w1 = np.matmul(c.transpose(),input_bias)

    #Regularization
    grad_w1 = (grad_w1 + (lambdaval * w1))/training_label.shape[0]
    grad_w2 = (grad_w2 + (lambdaval * w2))/training_label.shape[0]
    reg_co = np.sum(np.square(w1)) + np.sum(np.square(w2))
    obj_val = J + ((lambdaval * reg_co)/(2*training_label.shape[0]))
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array(obj_grad)
   
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""
    bias1 = np.full(shape=(np.array(data).shape[0],1),fill_value=1,dtype=np.float)
    input_bias = np.append(data, bias1, axis=1)
    hidden = sigmoid(np.matmul(input_bias, w1.transpose()))
    bias2 = np.full(shape=(np.array(hidden).shape[0],1),fill_value=1,dtype=np.float)
    hidden_bias = np.append(hidden, bias2, axis=1)
    output = sigmoid(np.matmul(hidden_bias, w2.transpose()))
    labels= np.argmax(output,axis=1)
    labels=labels.reshape((labels.shape[0],1)) 
    # Your code here
    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 5

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
