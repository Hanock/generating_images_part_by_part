import os
import time
from glob import glob
import tensorflow as tf

from ops import *
from utils import *

class GAN(object):
    def __init__(self, sess, config):

        self.sess = sess
        self.batch_size = config.batch_size
        self.train_size = config.train_size
        self.image_size = 64
        self.image_shape = [64, 64, 3]
        
        self.checked_img = self.create_checked_img(self.image_size)

        self.z_dim = 128

        self.checkpoint_dir = config.checkpoint_dir
        self.sample_dir = config.sample_dir
        self.build_model()

    def build_model(self):

        self.images = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='real_images')
        self.sample_images= tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='sample_images')
        self.z1 = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z1')
        self.z2 = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z2')
        
        # counter
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        #
        self.mean1, self.stddev1, self.mean2, self.stddev2 = self.encoder(self.images)
        self.vae_output, self.O1, self.O2, self.A1, self.A2 = self.generator(self.mean1, self.stddev1, self.mean2, self.stddev2)
        self.D = self.discriminator(self.images)

        self.G, _, _, _, _ = self.generator(self.z1, tf.zeros([self.batch_size, self.z_dim]), self.z2, tf.zeros([self.batch_size, self.z_dim]), reuse=True)
        self.D_ = self.discriminator(self.G, reuse=True)
        
        self.feature_out = self.discriminator(self.vae_output, reuse=True, feature=True)
        self.feature_img = self.discriminator(self.images, reuse=True, feature=True)
        
        
        # alpha loss
        self.alpha_loss1_1 = tf.reduce_sum(tf.abs(tf.reduce_sum(self.A1, [1, 2, 3]) - self.image_size*self.image_size*0.3))
        self.alpha_loss1_2 = tf.reduce_sum(tf.reduce_sum(-tf.square(self.A1 - 0.5) + 0.25, [1, 2, 3]))
        self.alpha_loss2_1 = tf.reduce_sum(tf.abs(tf.reduce_sum(self.A2, [1, 2, 3]) - self.image_size*self.image_size*0.3))
        self.alpha_loss2_2 = tf.reduce_sum(tf.reduce_sum(-tf.square(self.A2 - 0.5) + 0.25, [1, 2, 3]))
        self.alpha_loss = self.alpha_loss1_1 + self.alpha_loss1_2 + self.alpha_loss2_1 + self.alpha_loss2_2
        
        # prior loss
        self.prior_loss1 = tf.reduce_sum(0.5 * (tf.square(self.mean1) + tf.square(self.stddev1) - 2.0 * tf.log(self.stddev1 + 1e-8) - 1.0))
        self.prior_loss2 = tf.reduce_sum(0.5 * (tf.square(self.mean2) + tf.square(self.stddev2) - 2.0 * tf.log(self.stddev2 + 1e-8) - 1.0))
        self.prior_loss = self.prior_loss1 + self.prior_loss2

        # recon loss
        self.recon_loss_raw = tf.reduce_sum(tf.square(self.vae_output - self.images))
        self.recon_loss_adv = tf.reduce_sum(tf.square(self.feature_out - self.feature_img))
        
        # e loss
        self.e_loss = self.prior_loss + self.recon_loss_adv + 0.2*self.recon_loss_raw + 0.2*self.alpha_loss
        
        # g loss
        self.g_gan_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_), self.D_)
        self.g_loss = self.g_gan_loss + 0.0005*self.recon_loss_adv + 0.0001*self.recon_loss_raw + 0.0001*self.alpha_loss
        
        # d loss
        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D), self.D)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_), self.D_)
        self.d_loss = self.d_loss_real + self.d_loss_fake

        
        t_vars = tf.trainable_variables()

        self.e_vars = [var for var in t_vars if 'encoder' in var.name]
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        self.saver = tf.train.Saver()

    def train(self, data):
        """Train DCGAN"""
        
        data_size = len(data)

        e_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.e_loss, var_list=self.e_vars)
        d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(0.0004, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver()

        sample_images = self.transform(data[0:self.batch_size]).astype(np.float32)
        save_images(sample_images, [8, 16], '%s/sample_images.png' % (self.sample_dir))

        
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        counter = self.global_step.eval()

        
        for inf_loop in xrange(1000000):
            
            errD = 0
            errG = 0

            # random mini-batch
            i = np.random.randint(0, self.train_size - self.batch_size)
            batch_images = self.transform(data[i:i + self.batch_size]).astype(np.float32)
            
            # Update E network   
            self.sess.run([e_optim], feed_dict={ self.images: batch_images })

            # Update D network
            batch_z1 = self.random_z()
            batch_z2 = self.random_z()
            self.sess.run(d_optim, feed_dict={ self.images: batch_images, self.z1: batch_z1, self.z2: batch_z2 })

            # Update G network
            self.sess.run(g_optim, feed_dict={ self.images: batch_images, self.z1: batch_z1, self.z2: batch_z2 })

            counter += 1

            if np.mod(counter, 10) == 1:
                # random mini-batch
                i = np.random.randint(0, self.train_size - self.batch_size)
                batch_images = self.transform(data[i:i + self.batch_size]).astype(np.float32)
                batch_z1 = self.random_z()
                batch_z2 = self.random_z()
                
                recon_loss_adv = self.recon_loss_adv.eval({self.images: batch_images})
                recon_loss_raw = self.recon_loss_raw.eval({self.images: batch_images})
                alpha_loss = self.alpha_loss.eval({self.images: batch_images})
                g_gan_loss = self.g_gan_loss.eval({self.z1: batch_z1, self.z2: batch_z2})
                d_loss = self.d_loss.eval({self.z1: batch_z1, self.z2: batch_z2, self.images: batch_images})

                print("[%5d] time: %4.4f, adv: %.1f, raw: %.1f, alpha: %.1f, d: %.8f, g_gan: %.8f" \
                      % (counter, time.time() - start_time, recon_loss_adv, recon_loss_raw, alpha_loss, d_loss, g_gan_loss))

            if np.mod(counter, 100) == 1:
                #for gi in np.arange(1):
                samp_o1, samp_o2, samp = self.sess.run(
                    [self.O1, self.O2, self.vae_output],
                    feed_dict={self.images: sample_images})

                samp_fix = self.sess.run(
                    self.G,
                    feed_dict={self.z1: self.random_fix_z(), self.z2: self.random_z()})
                
                samples = np.concatenate((samp_fix, samp_o1, samp_o2, samp), axis=0)
                
                save_images(samples, [32, 16], '%s/train_%05d.png' % (self.sample_dir, counter))
                
            
            if np.mod(counter, 500) == 1:
                self.save(self.checkpoint_dir, counter)
    
    def encoder(self, image, reuse=False):
        
        def enc(image, name='enc', reuse=False):
            with tf.variable_scope(name, reuse=reuse):
                h0 = lrelu(conv2d(image, 64, name='e_h0_conv'))

                d_bn1 = batch_norm(self.batch_size, name='e_bn1')
                h1 = lrelu(d_bn1(conv2d(h0, 64*2, name='e_h1_conv')))

                d_bn2 = batch_norm(self.batch_size, name='e_bn2')
                h2 = lrelu(d_bn2(conv2d(h1, 64*4, name='e_h2_conv')))

                d_bn3 = batch_norm(self.batch_size, name='e_bn3')
                h3 = lrelu(d_bn3(conv2d(h2, 64*8, name='e_h3_conv')))

                h3_flat = tf.reshape(h3, [self.batch_size, -1])
                u = linear(h3_flat, self.z_dim, 'e_u_lin')
                s = tf.sqrt(tf.exp(linear(h3_flat, self.z_dim, 'e_s_lin')))
                
            return u, s
        
        with tf.variable_scope('encoder', reuse=reuse):
            u1, s1 = enc(image, 'enc1', reuse)
            u2, s2 = enc(image, 'enc2', reuse)

        return u1, s1, u2, s2

    def discriminator(self, image, feature=False, reuse=False):
            
        with tf.variable_scope('discriminator', reuse=reuse):
            h0 = lrelu(conv2d(image, 64, name='d_h0_conv'))

            d_bn1 = batch_norm(self.batch_size, name='d_bn1')
            h1 = lrelu(d_bn1(conv2d(h0, 64*2, name='d_h1_conv')))

            d_bn2 = batch_norm(self.batch_size, name='d_bn2')
            h2 = lrelu(d_bn2(conv2d(h1, 64*4, name='d_h2_conv')))

            d_bn3 = batch_norm(self.batch_size, name='d_bn3')
            h3 = lrelu(d_bn3(conv2d(h2, 64*8, name='d_h3_conv')))

            h3_flat = tf.reshape(h3, [self.batch_size, -1])
            h4 = linear(h3_flat, 1, 'd_h3_lin')

        if feature:
            return h3_flat

        return tf.nn.sigmoid(h4)

    def generator(self, u1, s1, u2, s2, name='generator', reuse=False, feature=False):
        
        def gen(h, gen_name='gen', reuse=False):
            with tf.variable_scope(gen_name, reuse=reuse):
                h0 = linear(h, 512*4*4, 'g_h0_lin')
                h0 = tf.reshape(h0, [-1, 4, 4, 512])
                h0 = tf.nn.relu(h0)

                h1 = deconv2d(h0, [self.batch_size, 8, 8, 256], name='g_h1')
                g_bn1 = batch_norm(self.batch_size, name='g_bn1')
                h1 = tf.nn.relu(g_bn1(h1))

                h2 = deconv2d(h1,[self.batch_size, 16, 16, 128], name='g_h2')
                g_bn2 = batch_norm(self.batch_size, name='g_bn2')
                h2 = tf.nn.relu(g_bn2(h2))

                h3 = deconv2d(h2, [self.batch_size, 32, 32, 64], name='g_h3')
                g_bn3 = batch_norm(self.batch_size, name='g_bn3')
                h3 = tf.nn.relu(g_bn3(h3))

                h4 = deconv2d(h3, [self.batch_size, 64, 64, 3], name='g_h4')
                alpha = deconv2d(h3, [self.batch_size, 64, 64, 1], name='g_a')
                
            return tf.sigmoid(h4), tf.sigmoid(alpha)
        
        e = tf.random_normal([self.batch_size, self.z_dim])
        samples1 = u1 + e * s1
        samples2 = u2 + e * s2
        
        with tf.variable_scope(name, reuse=reuse):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.z_dim, forget_bias=0.0)
            state = lstm_cell.zero_state(self.batch_size, tf.float32)
            
            with tf.variable_scope('g_rnn'):
                (cell_output1, state) = lstm_cell(samples1, state)
                tf.get_variable_scope().reuse_variables()
                (cell_output2, state) = lstm_cell(samples2, state)
                
            rgb1, alpha1 = gen(cell_output1, 'gen1')
            rgb2, alpha2 = gen(cell_output2, 'gen2')
            
            a_norm = alpha1*(1 - alpha2) + alpha2
            a1 = alpha1/a_norm
            a2 = alpha2/a_norm
            
            o1 = self.checked_img*(1 - a1) + rgb1*a1
            o2 = self.checked_img*(1 - a2) + rgb2*a2
            o = rgb1*alpha1*(1 - alpha2) + rgb2*alpha2
            
        return o*2 - 1, o1*2 - 1, o2*2 - 1, alpha1, alpha2

    def random_z(self):
        return np.random.normal(size=(self.batch_size, self.z_dim))
    
    def random_fix_z(self):
        r = np.zeros([self.batch_size, self.z_dim])
        r[0:,:] = np.random.normal(size=(1, self.z_dim))
        return r

    def save(self, checkpoint_dir, step):
        self.sess.run(self.global_step.assign(step))
        
        model_name = "GAN.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
        
    def transform(self, X):
        return X*2 - 1

    def inverse_transform(self, X):
        X = (X+1.)/2.
        return X
    
    def create_checked_img(self, size):
        arr = np.arange(size)
        chk1 = arr[np.mod(arr, size/8) < size/16]
        chk2 = arr[np.mod(arr, size/8) >= size/16]
        a = np.meshgrid(chk1, chk1)
        b = np.meshgrid(chk2, chk2)

        img = np.ones([size, size, 3], dtype=np.float32)

        img[a[0], a[1]] = 0
        img[b[0], b[1]] = 0

        return img