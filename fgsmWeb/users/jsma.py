import os
import numpy as np
import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from skimage import io
from .resizeimageutil import img_pad
import pickle
from sklearn.preprocessing import OneHotEncoder
import zipfile

#STEP 2 - Architecture Selection
def model(x,numclass, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[2, 2], padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)
    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)
    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])
    logits_ = tf.layers.dense(z, units=numclass, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')
    if logits:
        return y, logits_
    return y

def jsma(model,numclass, x, y=None, epochs=1, eps=1.0, k=1, clip_min=0.0, clip_max=1.0, score_fn=lambda t, o: t * tf.abs(o)):
    n = tf.shape(x)[0]
    target = tf.cond(tf.equal(0, tf.rank(y)), lambda: tf.zeros([n], dtype=tf.int32) + y, lambda: y)
    target = tf.stack((tf.range(n), target), axis=1) # 2xn
    if isinstance(epochs, float):
        tmp = tf.to_float(tf.size(x[0])) * epochs
        epochs = tf.to_int32(tf.floor(tmp))
    return _jsma_impl(model,numclass, x, target, epochs=epochs, eps=eps, clip_min=clip_min, clip_max=clip_max, score_fn=score_fn)

def _prod(iterable):
    ret = 1
    for x in iterable:
        ret *= x
    return ret

def _jsma_impl(model,numclass, x, yind, epochs, eps, clip_min, clip_max, score_fn):
    def cond(i, xadv):
        return tf.less(i, epochs)
    def body(i, xadv):
        ybar = model(xadv,numclass)
        dy_dx = tf.gradients(ybar, xadv)[0]   
        yt = tf.gather_nd(ybar, yind) 
        dt_dx = tf.gradients(yt, xadv)[0]        
        do_dx = dy_dx - dt_dx
        c0 = tf.logical_or(eps < 0, xadv < clip_max) 
        c1 = tf.logical_or(eps > 0, xadv > clip_min)
        cond = tf.reduce_all([dt_dx >= 0, do_dx <= 0, c0, c1], axis=0)
        cond = tf.to_float(cond)        
        score = cond * score_fn(dt_dx, do_dx) 
        shape = score.get_shape().as_list() 
        dim = _prod(shape[1:]) 
        score = tf.reshape(score, [-1, dim])
        ind = tf.argmax(score, axis=1)
        dx = tf.one_hot(ind, dim, on_value=eps, off_value=0.0)
        dx = tf.reshape(dx, [-1] + shape[1:])
        xadv = tf.stop_gradient(xadv + dx)
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return i+1, xadv
    #STEP 3 - Substitute Dataset Labeling
    _, xadv = tf.while_loop(cond, body, (0, tf.identity(x)), back_prop=False, name='_jsma_batch')
    return xadv

class Environment():
    pass

# CLASS ENVIRONMENT DEFINITION, BEFORE RUNNING MAIN


def evaluate(sess, ambiente, X_data, y_data, batch_size=128):
    print('\nValutazione')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0
    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run([ambiente.loss, ambiente.acc],feed_dict={ambiente.x: X_data[start:end], ambiente.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample
    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc

#STEP 4 - Substitute DNN F Training
def training(sess, ambiente, X_data, y_data, X_valid=None, y_valid=None, epochs=1, load=False, shuffle=True, batch_size=128, name='model'):
    if load:
        if not hasattr(ambiente, 'saver'):
            return print('\nError')
        return ambiente.saver.restore(sess, 'model/{}'.format(name))
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))
        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]
        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(ambiente.train_op, feed_dict={ambiente.x: X_data[start:end], ambiente.y: y_data[start:end], ambiente.training: True})
        if X_valid is not None:
            evaluate(sess, ambiente, X_valid, y_valid)
    # if hasattr(ambiente, 'saver'):
    #     os.makedirs('model', exist_ok=True)
    #     ambiente.saver.save(sess, 'model/{}'.format(name))

#STEP 5 - Jacobian-Based Dataset Augmentation
def make_jsma(sess, numclass,ambiente, X_data, epochs=0.2, eps=1.0, batch_size=128):
    print('\nInizio JSMA')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {ambiente.x: X_data[start:end], ambiente.target: np.random.choice(numclass),ambiente.adv_epochs: epochs,ambiente.adv_eps: eps}
        adv = sess.run(ambiente.x_jsma, feed_dict=feed_dict)
        X_adv[start:end] = adv
    print()
    return X_adv


def mainjsma(fileParent1):
#STEP 1 - Substitute Training Dataset Collection
    tf.reset_default_graph()
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)



    f = zipfile.ZipFile(os.path.join("uploadfile", fileParent1), 'r')  # 压缩文件位置
    fileParent = fileParent1.split(".")[0]
    if not os.path.exists("extract/" + fileParent):
        os.makedirs("extract/" + fileParent)
    for file in f.namelist():
        f.extract(file, "extract/" + fileParent + "/")  # 解压位置
    f.close()
    wh = 0
    # import imageio
    x_train = []
    y_train = []
    # pathname=r"D:\downloads\chromefile\fgsmWebTotal\fgsmWeb\extract"
    pathname = "extract/" + fileParent
    filenamel = []
    if os.path.exists(pathname):
        filelist = os.listdir(pathname)
        numclass = len(filelist)
        for f in filelist:
            subf = os.path.join(pathname, f)
            if os.path.isdir(subf):
                filelist1 = os.listdir(subf)
                for f1 in filelist1:
                    if f1.endswith(".bmp") or f1.endswith(".png") or f1.endswith(".jpg") or f1.endswith(".ppm"):
                        filepath = os.path.join(subf, f1)
                        # print()
                        img = io.imread(filepath)
                        if wh == 0:
                            wh = img.shape[0]
                        x_train.append(img_pad(img, wh))
                        y_train.append(int(f))
                        filenamel.append(f + "/" + f1)
    x_train = np.array(x_train)
    input_shape = x_train.shape[1:]
    y_train = np.array(y_train)



    onehotencode=OneHotEncoder()
    y_train=onehotencode.fit_transform(y_train.reshape(-1,1))
    # print(y_train)
    y_train=(y_train).toarray()
    # print(y_train)



    tf.logging.set_verbosity(old_v)

    print('\nInizializzazione grafo')
    ambiente =  Environment()

    with tf.variable_scope('model'):
        ambiente.x = tf.placeholder(tf.float32, (None, wh,wh, 3),name='x')
        ambiente.y = tf.placeholder(tf.float32, (None, numclass), name='y')
        ambiente.training = tf.placeholder_with_default(False, (), name='mode')
        ambiente.ybar, logits = model(ambiente.x,numclass, logits=True, training=ambiente.training)
        with tf.variable_scope('acc'):
            count = tf.equal(tf.argmax(ambiente.y, axis=1), tf.argmax(ambiente.ybar, axis=1))
            ambiente.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
        with tf.variable_scope('loss'):
            xent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ambiente.y, logits=logits)
            ambiente.loss = tf.reduce_mean(xent, name='loss')
        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer()
            ambiente.train_op = optimizer.minimize(ambiente.loss)
        ambiente.saver = tf.train.Saver()

    with tf.variable_scope('model', reuse=True):
        ambiente.target = tf.placeholder(tf.int32, (), name='target')
        ambiente.adv_epochs = tf.placeholder_with_default(20, shape=(), name='epochs')
        ambiente.adv_eps = tf.placeholder_with_default(0.2, shape=(), name='eps')
        ambiente.x_jsma = jsma(model, numclass,ambiente.x, ambiente.target, eps=ambiente.adv_eps, epochs=ambiente.adv_epochs)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver=tf.train.Saver(max_to_keep=1)
# runs training and evaluating
# STEP 4

    training(sess, ambiente, x_train, y_train, epochs=2)


    # if os.path.exists("ckpt/fgsm.ckpt"):
    #     print("loaded")
    #     saver.restore(sess, "ckpt/fgsm.ckpt")
    #
    # saver.save(sess, 'ckpt_jsma/jsma.ckpt')

    X_adv = make_jsma(sess, numclass,ambiente, x_train/255.0, epochs=40, eps=0.8)

    pathsave = r"download"
    # if not os.path.exists(pathsave+"/extract"):
    #     os.makedirs(pathsave+"/extract")
    tmp = ''
    for i, X_advP in enumerate(X_adv):

        if not os.path.exists(pathsave + "/" + fileParent + "/" + filenamel[i].split("/")[0]):
            os.makedirs(pathsave + "/" + fileParent + "/" + filenamel[i].split("/")[0])
        if len(tmp) == 0:
            tmp = "static" + "/" + filenamel[i].split("/")[-1].split(".")[0] + ".png"
            print(tmp)
            io.imsave(tmp, (X_advP * 255).astype(int))
        io.imsave(os.path.join(pathsave + "/" + fileParent, filenamel[i]), (X_advP * 255).astype(int))
    evaluate(sess, ambiente, x_train, y_train)
    evaluate(sess, ambiente, X_adv, y_train)
    return tmp
if __name__ == "__main__":
    mainjsma("data.zip")

