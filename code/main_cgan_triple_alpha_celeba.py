import os
import numpy as np
import tensorflow as tf
import h5py

from lib.cgan_triple_alpha import GAN

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 128, "The size of batch images")
flags.DEFINE_integer("train_size", 160000, "The size of train images")
flags.DEFINE_string("dataset", "celeba", "The name of dataset")
flags.DEFINE_string("data_dir", "../data/celeba.hdf5", "Directory of dataset")
flags.DEFINE_string("checkpoint_dir", "../checkpoint/cgan_triple_alpha_celeba", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "../samples/cgan_triple_alpha_celeba", "Directory name to save the image samples [samples]")
FLAGS = flags.FLAGS

def main(_):

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        gan = GAN(sess, FLAGS)
        
        HDF5_FILE = FLAGS.data_dir
        f = h5py.File(HDF5_FILE, 'r')
        tr_data = f['images'] # data_num * height * width * channel (float32)
        
        gan.train(tr_data)
        f.close()
        

if __name__ == '__main__':
    tf.app.run()
