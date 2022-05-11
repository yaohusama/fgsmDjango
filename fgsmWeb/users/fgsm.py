import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import math
# from tensorflow.examples.tutorials.mnist import input_data as mnist_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
batch_size=256
from .resizeimageutil import img_pad
import numpy as np
from sklearn.model_selection import train_test_split
import zipfile
def model(x,num,logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)#,padding='valid
    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)#,padding="valid"
    with tf.variable_scope('flat'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])
    # if training:
    #     z=tf.layers.dropout(z,rate=0.2)
    # z = tf.layers.dense(z, units=512, name='logits0',activation="relu")
    l_layer = tf.layers.dense(z, units=num, name='logits')
    y = tf.nn.softmax(l_layer, name='ybar')
    if logits:
        return y, l_layer
    return y

#FGSM
def fgm(model,num, x, eps=0.01, epochs=1, sign=True, clip_min=0, clip_max=1):
    xadv = tf.identity(x)
    ybar = model(xadv,num)
    yshape = ybar.get_shape().as_list()
    ydim = yshape[1]
    indices = tf.argmax(ybar, axis=1)
    target = tf.cond(
     tf.equal(ydim,1),
     lambda: tf.nn.relu(tf.sign(ybar-0.5)),
     lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))
    loss_fn = tf.nn.softmax_cross_entropy_with_logits_v2
    noise_fn = tf.sign
    eps = tf.abs(eps)
    # eps=0
    def cond(xadv, i):
        return tf.less(i, epochs)
    def body(xadv, i):
        ybar,logits= model(xadv, num,logits=True)
        loss = loss_fn(labels=target, logits=logits)
        dy_dx, = tf.gradients(loss, xadv)
        xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx))
        # xadv=tf.stop_gradient(xadv)
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i+1
    xadv, _ = tf.while_loop(cond, body, (xadv, 0), back_prop=False, name='fast_gradient')
    return xadv

# CLASS ENVIRONMENT DEFINITION, BEFORE RUNNING MAIN
class Environment():
	pass


#STEP 4 - Training
def training(sess, ambiente, X_data, Y_data, X_valid=None, y_valid=None, shuffle=True, batch=128, epochs=1):
    Xshape = X_data.shape
    n_data = Xshape[0]
    n_batches = int(n_data/batch)
    # print(n_batches)
    # print(X_data.shape)
    for ep in range(epochs):
        print('epoch number: ', ep+1)
        if shuffle:
            ind = np.arange(n_data)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            Y_data = Y_data[ind]
        for i in range(n_batches):
            # print("y")
            print(' batch {0}/{1}'.format(i + 1, n_batches),end="\r")
            start = i*batch
            end = min(start+batch, n_data)
            sess.run([ambiente.train_op], feed_dict={ambiente.x: X_data[start:end], ambiente.y: Y_data[start:end]})
            # print(ambiente.loss)
        evaluate(sess, ambiente, X_data, Y_data)
		
def evaluate(sess, ambiente, X_test, Y_test, batch=128):
	n_data = X_test.shape[0]
	n_batches = int(n_data/batch)
	totalAcc = 0
	totalLoss = 0
	for i in range(n_batches):
		print(' batch {0}/{1}'.format(i + 1, n_batches), end='\r')
		start = i*batch 
		end = min(start+batch, n_data)
		batch_X = X_test[start:end]
		batch_Y = Y_test[start:end]
		batch_loss, batch_acc = sess.run([ambiente.loss, ambiente.acc], feed_dict={ambiente.x: batch_X, ambiente.y: batch_Y})
		totalAcc = totalAcc + batch_acc*(end-start)
		totalLoss = totalLoss + batch_loss*(end-start)
	totalAcc = totalAcc/n_data
	totalLoss = totalLoss/n_data
	print('acc: {0:.3f} loss: {1:.3f}'.format(totalAcc, totalLoss))
	return totalAcc, totalLoss


def perform_fgsm(sess,num, ambiente, X_data, epochs=1, eps=0.01, batch_size=128):
    print('\nInizio FGSM')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(ambiente.x_fgsm, feed_dict={
            ambiente.x: X_data[start:end],
            ambiente.fgsm_eps: eps,
            ambiente.fgsm_epochs: epochs})
        X_adv[start:end] = adv
    print()
    return X_adv

import skimage.io as io
def mainfgsm(fileParent1):
#STEP 1 - Initial Dataset Collection 
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    subtract_pixel_mean = True

    f = zipfile.ZipFile(os.path.join("uploadfile",fileParent1), 'r')  # 压缩文件位置
    fileParent=fileParent1.split(".")[0]
    if not os.path.exists("extract/"+fileParent):
        os.makedirs("extract/"+fileParent)
    for file in f.namelist():
        f.extract(file, "extract/"+fileParent+"/")  # 解压位置
    f.close()
    wh=0
    # import imageio
    x_train=[]
    y_train=[]
    # pathname=r"D:\downloads\chromefile\fgsmWebTotal\fgsmWeb\extract"
    pathname="extract/"+fileParent
    filenamel=[]
    if os.path.exists(pathname):
        filelist = os.listdir(pathname)
        numclass=len(filelist)
        for f in filelist:
            subf = os.path.join(pathname, f)
            if os.path.isdir(subf):
                filelist1 = os.listdir(subf)
                for f1 in filelist1:
                    if f1.endswith(".bmp") or f1.endswith(".png") or f1.endswith(".jpg") or f1.endswith(".ppm"):
                        filepath=os.path.join(subf,f1)
                        # print()
                        img=io.imread(filepath)
                        if wh==0:
                            wh=img.shape[0]
                        x_train.append(img_pad(img,wh))
                        y_train.append(int(f))
                        filenamel.append(f+"/"+f1)
    x_train = np.array(x_train)
    input_shape = x_train.shape[1:]
    y_train=np.array(y_train)

    x_train = x_train.astype('float32') / 255
    onehotencode=OneHotEncoder()
    y_train=onehotencode.fit_transform(np.array(y_train).reshape(-1,1))
    y_train=y_train.toarray()

    tf.logging.set_verbosity(old_v)
    ambiente = Environment()

    with tf.variable_scope('model'):
        ambiente.x = tf.placeholder(tf.float32, (None, wh, wh, 3))
        ambiente.y = tf.placeholder(tf.float32, (None, numclass), name='y')
        # calls model (STEP 2)
        ambiente.ybar, logits = model(ambiente.x,numclass, logits=True)
        with tf.variable_scope('acc'):
            count = tf.equal(tf.argmax(ambiente.y, axis=1), tf.argmax(ambiente.ybar, axis=1))
            ambiente.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
        with tf.variable_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ambiente.y, logits=logits)
            ambiente.loss = tf.reduce_mean(cross_entropy, name='loss')
        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer()
            ambiente.train_op = optimizer.minimize(ambiente.loss)
    with tf.variable_scope('model', reuse=True):
        ambiente.fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
        ambiente.fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')
        ambiente.x_fgsm = fgm(model, numclass, ambiente.x, epochs=ambiente.fgsm_epochs, eps=ambiente.fgsm_eps)
    sess = tf.InteractiveSession() # ENVIRONMENT -> MODEL -> FGM

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver=tf.train.Saver(max_to_keep=1)
# runs training and evaluating
# STEP 4
    if os.path.exists(r"ckptfgsm"):
        print("loaded")
        saver.restore(sess, "ckptfgsm/fgsm.ckpt")



    training(sess, ambiente, x_train, y_train,  shuffle=False, batch=batch_size, epochs=60)


    # saver.save(sess,'ckptfgsm/fgsm.ckpt')


    X_adv = perform_fgsm(sess, numclass,ambiente, x_train, eps=0.002, epochs=15)
    pathsave=r"download"
    tmp=''
    for i,X_advP in enumerate(X_adv):

        if not os.path.exists(pathsave+"/"+fileParent+"/"+filenamel[i].split("/")[0]):
            os.makedirs(pathsave+"/"+fileParent+"/"+filenamel[i].split("/")[0])
        if len(tmp)==0:
            tmp="static"+"/"+filenamel[i].split("/")[-1].split(".")[0]+".png"
            print(tmp)
            io.imsave(tmp,(X_advP*255).astype(int))
        io.imsave(os.path.join(pathsave+"/"+fileParent,filenamel[i]),(X_advP*255).astype(int))
    evaluate(sess, ambiente, x_train ,y_train)
    evaluate(sess, ambiente, X_adv, y_train)
    return tmp


if __name__ == "__main__":
    mainfgsm()


