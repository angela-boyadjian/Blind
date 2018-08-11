import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def recupPixel(path):
    if (path == "resize/2.jpg"or path == "resize/17.jpg" or
    path == "resize/19.jpg" or path == "resize/29.jpg" or path == "resize/31.jpg"):
        return(1)
    im = Image.open(path)
    pixels = list(im.getdata())
    return(pixels)

# entry list
e = []
i = 3
while (i <= 36):
    tmp = recupPixel("resize/" + str(i) + ".jpg")
    if (tmp != 1):
        e.append(tmp)
    i += 1
# outs list
s = [[0], [0], [0], [1], [1], [0], [0], [0], [0], [1], [0], [1], [1], [0], [1], [0], [1], [1], [0], [1], [1], [0], [0], [0], [0], [1], [1], [1], [1], [1]]

# test list
t = []
t.append(recupPixel("resizeTest/1.jpg"))
t.append(recupPixel("resizeTest/2.jpg"))
t.append(recupPixel("resizeTest/17.jpg"))
t.append(recupPixel("resizeTest/19.jpg"))
t.append(recupPixel("resizeTest/29.jpg"))
t.append(recupPixel("resizeTest/31.jpg"))

# test outs
to = [[1], [0], [0], [1], [0], [1]]

# initialisation of neural network
entry = tf.placeholder(tf.float32, (None, 123480))
hidden = tf.layers.dense(entry, 20, activation=tf.nn.selu)
out = tf.layers.dense(hidden, 1)

# initialisation of calc variables
y = tf.placeholder(tf.float32, (None, 1))
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=y), axis=0)
optimizer = tf.train.AdamOptimizer(0.1)
op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

# start a new session
with tf.Session() as sess:
  # run the initilisation of variables
  sess.run(init)
  # learning loop
  for _ in range(100):
    # run the loss calc
    loss_value = sess.run(loss, feed_dict={entry: e, y: s})
    print(loss_value)
    # run the optimizer
    sess.run(op, feed_dict={entry: e, y: s})
  # run the result with neural network prediction
  out_value = sess.run(tf.sigmoid(out), feed_dict={entry: t})
  print("\n\n----- good values -----\n", to, "\n\n----- predictions -----\n", out_value)