import numpy as np
import matplotlib
import tensorflow as tf
#import numpy as np
import gzip
import os
import sys
import time

batch_size = 100
test_size = 256
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 30
NUM_CHANNELS = 1
PIXEL_DEPTH = 255


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def encoder(X, w, w2, wd, wd2):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                                 #input 30 x30x1, output 24x24x32
                        strides=[1, 1, 1, 1], padding='VALID'))

    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],                        #input 24x24x32, output 12x12x32
                        strides=[1, 2, 2, 1], padding='SAME')
    

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                               
                        strides=[1, 1, 1, 1], padding='VALID'))          #input 12x12x32, output 8x8x64
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],                        #input 8x8x64, output 4x4x64
                        strides=[1, 2, 2, 1], padding='SAME')

    #nearest neighbour upsampling                                        
    l1da = tf.image.resize_images(l2, 8,                                   #input 4x4x64, output 8x8x64
                        8, 1, align_corners=False)
    # print(l1da.shape)
    output_shapel1d = tf.convert_to_tensor([100, 12, 12, 32], dtype=tf.int32);
    l1d = tf.nn.relu(tf.nn.conv2d_transpose(l1da, wd, output_shapel1d,                         #input 8x8x64, output 12x12x32
                        strides=[1, 1, 1, 1], padding='VALID'))
    
    #nearest neighbour upsampling                                        
    l2da = tf.image.resize_images(l1d, 24,                                   #input 12x12x32, output 24x24x32
                        24, 1, align_corners=False)

    output_shapel2d = tf.convert_to_tensor([100, 30, 30, 1], dtype=tf.int32);
    l2d = tf.nn.relu(tf.nn.conv2d_transpose(l2da, wd2, output_shapel2d,                                 #input 24x24x32, output 30x30x1
                        strides=[1, 1, 1, 1], padding='VALID'))
    return l2d


def extract_data(sess, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    #print('Extracting', filename)
    directory = '/usr/local/lib/python2.7/dist-packages/tensorflow/models/gh_video_prediction/images/images/'
    imagefile_list = []
    for t in range(0, num_images-1):
        imagefile_list.append(directory + 'img_' + `t` + '.jpg')
    filename_queue = tf.train.string_input_producer(imagefile_list)
    image_list = []
    reader = tf.WholeFileReader()
    with sess as sess:
	    init_op = tf.initialize_all_variables()
	    sess.run(init_op)

	    # Start populating the filename queue.
	
	    coord = tf.train.Coordinator()
	    threads = tf.train.start_queue_runners(coord=coord)
	    
	    for i in range(0, num_images):
	        key, value = reader.read(filename_queue)
	        my_img = tf.image.decode_jpeg(value, channels = 1)
	        image = my_img.eval() 
	        image_list.append(image)
	    coord.request_stop()
	    coord.join(threads)

    return image_list


def write_data(sess, num)


sess= tf.Session();
print("extracting data")
complete_image = extract_data(sess, 1000)
trX = complete_image[0:900]
trY = trX
teX = complete_image[900:1000]
teY = teX

print("data extracted");

X = tf.placeholder("float", [100, 30, 30, 1])
Y = tf.placeholder("float", [100, 30, 30, 1])

w = init_weights([7, 7, 1, 32])       
w2 = init_weights([5, 5, 32, 64])     
wd = init_weights([5, 5, 32, 64])
wd2 = init_weights([7, 7, 1, 32])
py_x = encoder(X, w, w2, wd, wd2)

cost = tf.reduce_mean(tf.squared_difference(py_x, Y, name = None))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = py_x;

ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

global_step = tf.Variable(0, name='global_step', trainable=False)

# Call this after declaring all tf.Variables.
saver = tf.train.Saver()

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

    start = global_step.eval() # get last global_step
    print "Start from:", start

    for i in range(start, 500):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX), batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        global_step.assign(i).eval() # set and update(eval) global_step with index, i
        saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
        test_result, act_cost = sess.run([predict_op, cost], feed_dict={X: teX[start:end], 
        print("cost durign epoch " + `i` + "is ", act_cost)        