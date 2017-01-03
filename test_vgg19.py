import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import vgg19
import utils
try:
    import cPickle as pickle
except:
    import pickle


def get_tensors(g):
    i = [
        g.get_tensor_by_name("content_vgg/concat:0"),
        g.get_tensor_by_name("content_vgg/conv1_1/filter:0"),
        g.get_tensor_by_name("content_vgg/conv1_1/Conv2D:0"),
        g.get_tensor_by_name("content_vgg/conv1_1/Relu:0"),
        g.get_tensor_by_name("content_vgg/conv1_2/filter:0"),
        g.get_tensor_by_name("content_vgg/conv1_2/Conv2D:0"),
        g.get_tensor_by_name("content_vgg/pool1:0"),
        g.get_tensor_by_name("content_vgg/conv2_1/filter:0"),
        g.get_tensor_by_name("content_vgg/conv2_1/Conv2D:0"),
        g.get_tensor_by_name("content_vgg/conv2_1/Relu:0"),
        g.get_tensor_by_name("content_vgg/conv2_2/filter:0"),
        g.get_tensor_by_name("content_vgg/conv2_2/Conv2D:0"),
        g.get_tensor_by_name("content_vgg/pool2:0"),
        g.get_tensor_by_name("content_vgg/conv3_1/filter:0"),
        g.get_tensor_by_name("content_vgg/conv3_1/Conv2D:0"),
        g.get_tensor_by_name("content_vgg/conv3_1/Relu:0"),
        g.get_tensor_by_name("content_vgg/conv3_2/filter:0"),
        g.get_tensor_by_name("content_vgg/conv3_2/Conv2D:0"),
        g.get_tensor_by_name("content_vgg/pool3:0"),
        g.get_tensor_by_name("content_vgg/conv4_1/filter:0"),
        g.get_tensor_by_name("content_vgg/conv4_1/Conv2D:0"),
        g.get_tensor_by_name("content_vgg/conv4_1/Relu:0"),
        g.get_tensor_by_name("content_vgg/conv4_2/filter:0"),
        g.get_tensor_by_name("content_vgg/conv4_2/Conv2D:0"),
        g.get_tensor_by_name("content_vgg/pool4:0"),
        g.get_tensor_by_name("content_vgg/conv5_1/filter:0"),
        g.get_tensor_by_name("content_vgg/conv5_1/Conv2D:0"),
        g.get_tensor_by_name("content_vgg/fc6/Reshape:0"),
        g.get_tensor_by_name("content_vgg/fc6/weights:0"),
        g.get_tensor_by_name("content_vgg/fc6/MatMul:0"),
    ]
    return i

img1 = utils.load_image("./test_data/tiger.jpeg")
img2 = utils.load_image("./test_data/puzzle.jpeg")

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)

with tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    images = tf.placeholder("float", [2, 224, 224, 3])
    feed_dict = {images: batch}

    vgg = vgg19.Vgg19()
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    # prob = sess.run(vgg.prob, feed_dict=feed_dict)

    # for profile
    run_options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)
    run_metadata = tf.RunMetadata()
    prob = sess.run(vgg.prob, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
    step_stats = run_metadata.step_stats
    tl = timeline.Timeline(step_stats)
    ctf = tl.generate_chrome_trace_format(show_memory=False, show_dataflow=True)
    with open("timeline.json", "w") as f:
        f.write(ctf)
    print(prob)
    utils.print_prob(prob[0], './synset.txt')
    utils.print_prob(prob[1], './synset.txt')
    g = sess.graph
    i = get_tensors(g)
    for tensor in i:
        print tensor
    # o = sess.run(i, feed_dict=feed_dict)
    # with open('tensors.pickle', mode='wb') as f:
    #     data = {idx:out for (idx, out) in enumerate(o)}
    #     pickle.dump(data, f)
