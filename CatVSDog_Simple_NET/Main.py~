

#Loading Libraries
from PIL import Image
import numpy as np
import os
#Loading TensorFlow
import tensorflow as tf



## Module for the preprocessing of the dataset
def  load_data(directory): 
    files = os.listdir("./"+directory)
    num = len(files)
    data = np.empty((num,120000),dtype="float32")
    label = np.empty((num,2),dtype ="float32")
    for i in range(num):
        if not files[i].startswith('.'):
            label[i,0]=0 if files[i].split('.')[0]=='cat' else 1
            label[i,1]=1 if files[i].split('.')[0]=='cat' else 0
            img = Image.open("./"+directory+"/"+files[i])
            img = img.resize((200,200), Image.ANTIALIAS)
            Image_array = np.asImage_arrayay (img, dtype ="float32")
            Image_array= Image_array.flatten() 
            data[i,:]= Image_array
            img.close()
    return data, label



#Loading Traing Data
a = load_data("Sample_train")



#Session and Variable Definition
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 120000])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
W = tf.Variable(tf.zeros([120000,2]))
b = tf.Variable(tf.zeros([2]))

#Variable Initialization 
sess.run(tf.initialize_all_variables())

## SoftMax
y = tf.nn.softmax(tf.matmul(x,W) + b)
#Loss Function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#Gredient Descent
train_step = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)




#Training
for i in range(100):
  batch = a
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy: ", accuracy.eval(feed_dict={x: a[0], y_: a[1]}) * 100)




