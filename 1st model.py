import numpy as np
import math
import pandas as pd
import tensorflow  as tf
import cv2
import matplotlib.pyplot as plt


# get the image filename and labels
def get_imagefilename(disease, annotation, num, BASE_PATH):
    '''
    
    Reads the image from the directory, reshapes it and get the corresponding label
    
    Args:
        disease -- a list containing the filenames
        labels -- the annotations
        BASE_PATH -- path to the image floder
        num --> number of image to retrieve (couldn't use all the images becuse of the computing power )
    
    retruns:
        numpy array of reshaped images, numpy array of disease labels
    '''

    images = []
    disease_label = []
    for i in range(num):
        if disease in annotation['Finding Labels'][i]:
            label = 1
        else:
            label = 0
        image = cv2.imread(BASE_PATH + annotation['Image Index'][i])
        image = cv2.resize(image, (128, 128))
        images.append(image)
        disease_label.append(label)
    
    return np.array(images), np.array(disease_label).reshape(num, 1)



def random_image_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (m, height, width, number of channels)
    Y -- true "label" vector (1 for yes disease / 0 for no disease), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[0]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation].reshape(m, 2)

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[k * num_complete_minibatches:(k+1) * num_complete_minibatches, :, :, :]
        mini_batch_Y = shuffled_Y[k * num_complete_minibatches:(k+1) * num_complete_minibatches]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[mini_batch_size*num_complete_minibatches:m, :, :, :]
        mini_batch_Y = shuffled_Y[mini_batch_size*num_complete_minibatches:m]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def one_hot_matrix(labels, C):
    
    C = tf.constant(C, name='C')    
    one_hot_matrix = tf.one_hot(labels, C, axis=1)    
    sess = tf.Session()    
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot

BASE_PATH = 'images\\'


# extract the annotation
annotation = pd.read_csv('labels.csv')
       
# loading the images
images, disease_label = get_imagefilename('Infiltration', annotation, 2000, BASE_PATH)


diseases = one_hot_matrix(disease_label, 2).reshape(2000, 2)
    
# dividing the data in train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, diseases, test_size=0.33, random_state=1)


def model(X_train, y_train, X_test, y_test, learning_rate = 0.009, num_epochs = 100, 
          minibatch_size = 64, print_cost = True):
    
    X = tf.placeholder(tf.float32, [None, 128, 128, 3], name='X')
    y = tf.placeholder(tf.float32, [None, 2], name='y')
    
    #W1 = [4,4, 3, 8]
    W1 = tf.get_variable("W1", [4,4,3,8], initializer = tf.contrib.layers.xavier_initializer(seed = 0) )
    
    #W2 = [2, 2, 8, 16]
    W2 = tf.get_variable("W2", [2,2,8,16], initializer = tf.contrib.layers.xavier_initializer(seed = 0) )
    
    #first_convolutionl layer
    # stride = 1x1 for convolution and 8x8 for max_pooling, ksize=8x8, padding=SAME
    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    A2 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(Z1, ksize=[1, 8,8,1], strides=[1, 8, 8, 1], padding='SAME')
    
    #second_convolutionl layer
    # stride = 1x1 for convolution and 4x4 for max_pooling, ksize=4x4, padding=SAME
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 8,8,1], strides=[1, 8, 8, 1], padding='SAME')
    
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, num_outputs=2, activation_fn=None, name='Z3')
    
    # cost function 
    Z3 = tf.cast(Z3, tf.float32, name='Z3')
    
    cost =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = y))

    # optimizer/backward pass
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
                    
    # initilizer all variables
    init = tf.global_variables_initializer()
    
    # cost
    costs=[]
    
    saver = tf.train.Saver()
        
    # define the session to run the computational graph
    with tf.Session() as sess:
        
        sess.run(init)
        seed=0
    
        
        for epoch in range(num_epochs):
            # cost for each epoch
            minibatch_cost = 0
            
            # size of the traing data
            m=X_train.shape[0]
            
            # number of minibatches
            num_minibatches = int(m/minibatch_size)       
            seed = seed + 1
            
            #create minibatches 
            minibatches = random_image_mini_batches(X_train, y_train, minibatch_size, seed)
            
            for minibatch in minibatches:
                
                minibatch_X, minibatch_y = minibatch
                
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, y:minibatch_y})
                
                minibatch_cost += temp_cost /num_minibatches
                    

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, y: y_train})
        test_accuracy = accuracy.eval({X: X_test, y: y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        print('parameters learned W1:', W1.eval())
        print('parameters learned W2:', W2.eval())
        
        save_path  = saver.save(sess,	".\\my_model_final.ckpt")

        
        return W1, W2
    
            
tf.reset_default_graph()

w1, w2 = model(X_train, y_train, X_test, y_test, learning_rate = 0.009, num_epochs = 100, 
          minibatch_size = 100, print_cost = True)
    
