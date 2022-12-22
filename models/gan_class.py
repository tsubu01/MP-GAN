import tensorflow as tf
import numpy as np
from ipywidgets import IntProgress
from IPython.display import display
import time
from IPython.display import display, clear_output
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from .deep_model import *

tf.config.run_functions_eagerly(True)

def debug_func(x):
    #print('shape of input: ', x.shape)
    print('value of input: ', x)
        
def array3ddebug(x):
    print('shape: ', x.shape) 
    print('is 1st element == 2nd element? diff is ', np.sum(x[0]-x[1]))

#single_value_func = tf.function(debug_func)

array_3d_func = tf.function(array3ddebug)
    
                       


class GAN():
    def __init__(self, isplot, 
                 keepshape=True, 
                 is_table=False,
                 lr=0.0002, 
                 lr_steps=1, 
                 decay_rate=0.9, 
                 scaler=None):
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.latent_dim = None
        self.n_outputs = None
        self.is_plot = isplot
        self.keep_shape = keepshape
        self.is_table = is_table
        self.lr = lr
        self.gan_opt = None
        self.disc_opt = None
        self.lr_steps = lr_steps
        self.decay_rate = decay_rate
        self.real_sample_metrics = {}
        self.fake_sample_metrics = {}
        self.decayed_lr = {}
        self.train_fig = None
        self.batches_per_epoch = 0
        self.epoch = 0
        self.decay_rate = decay_rate
        self.scaler = scaler
    
    
    
    def _learning_rate_scheduler(self):
        return(self.lr * self.decay_rate **(-(self.epoch)))

    def set_batches_per_epoch(self, dataset, batch_size):
        self.batches_per_epoch = int(dataset.shape[0]/batch_size)

        
    # define the standalone discriminator model
    def define_discriminator(self, n_inputs, layers):
        #layers is a list of lists; each member is a 3-member list with the format: [layer_type, layer/kernel size, activation]
        self.n_inputs = n_inputs        
        #model needs to be compiled because we use it as a standalone model for training on real data.
        # compile model
        model = DeepModel(self.n_inputs, layers).create_model()
        #uri self.disc_opt = SGD(learning_rate=self.lr_schedule_func(2*self.lr_steps))
        #uri self.disc_opt = SGD(learning_rate=self.lr_schedule_func(2*self.lr_steps))
        self.disc_opt = Adam(learning_rate=self._learning_rate_scheduler(), beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=self.disc_opt, metrics=['accuracy'])
        self.discriminator = model
        print('Discriminator is now defined')
    
    def define_generator(self, latent_dim, layers):
        self.latent_dim = latent_dim        
        model = DeepModel(self.latent_dim, layers).create_model()
        self.generator = model
        print('Generator is now defined')
        
    def define_gan(self):
        #make discriminator weights not trainable.
        discriminator = self.discriminator
        generator = self.generator
        discriminator.trainable = False
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        #uri self.gan_opt = SGD(learning_rate=self.lr_schedule_func(self.lr_steps))
        self.gan_opt = Adam(learning_rate=self._learning_rate_scheduler(), beta_1=0.5)
        #print('>>> decrease lr every {} steps'.format(self.lr_steps))

        model.compile(loss='binary_crossentropy',
                      optimizer=self.gan_opt,
                      metrics=['accuracy'])
        self.gan = model
        print('GAN is now defined')
     
    def generate_real_samples(self, n, realsamplearray):
        keepshape=self.keep_shape
        # generate mnist samples
        #print(realsamplearray.shape)
        if keepshape:
            sampleshape = realsamplearray[0].shape
        else:
            sampleshape = realsamplearray[0].reshape(-1).shape[0]
        array_inds = range(realsamplearray.shape[0])
        random_index_list = []
        if keepshape:
            X = zeros((n, sampleshape[0], sampleshape[1]))
        else:
            X = zeros((n, sampleshape))

        for i in range(n):
            if keepshape:
                X[i] = realsamplearray[np.random.choice(array_inds)]
            else:
                X[i, :] = realsamplearray[np.random.choice(array_inds)].reshape(-1)
        # generate 'true' class labels
        y = ones((n,1))
        #print(X.shape, y.shape)
        if self.is_table:
            return X, y
        else:
            #return X/255, y
            return X, y
    
    # generate points in latent space as input for the generator
    def generate_latent_points(self, latent_dim, n):
        # generate points in the latent space
        x_input = randn(latent_dim * n)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n, latent_dim)
        #array_3d_func(x_input)
        return x_input
 
    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, generator, latent_dim, n):
        # generate points in latent space
        x_input = self.generate_latent_points(latent_dim, n)
        # predict outputs
        # debug:
        X = self.generator.predict(x_input)
        #array_3d_func(X)
        # create class labels
        y = zeros((n,1))
        #print(X.shape, y.shape)
        return X, y
        
        
    # evaluate the discriminator and plot real and fake points
    def summarize_performance(self, epoch, latent_dim, n=200, dataset=None, scaler=None):
        # prepare real samples
        generator = self.generator
        discriminator = self.discriminator
        x_real, y_real = self.generate_real_samples(n, dataset)
        # evaluate discriminator on real examples
        _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake = self.generate_fake_samples(generator, latent_dim, n)
        # evaluate discriminator on fake examples
        _, acc_fake = self.discriminator.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print(epoch, 'acc_real: ', acc_real, ' , acc_fake: ', acc_fake)
        
        self.real_sample_metrics[epoch] = acc_real
        self.fake_sample_metrics[epoch] = acc_fake
        self.decayed_lr[epoch] = self.gan_opt._decayed_lr(tf.float32).numpy()
        clear_output(wait = True)
        print('epoch number: ', epoch)
        print('learning rate: {:.10f}'.format(self.gan_opt._decayed_lr(tf.float32).numpy()))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1) 
        
        epoch_array = [i[0] for i in self.real_sample_metrics.items()]
        real_sample_metrics_array = [i[1] for i in self.real_sample_metrics.items()]
        fake_sample_metrics_array = [i[1] for i in self.fake_sample_metrics.items()]

        ax.cla()
        plt.plot(epoch_array, real_sample_metrics_array)
        plt.plot(epoch_array, fake_sample_metrics_array)
        ax.set_ylim([0,1])
        plt.legend(['real sample acc', 'synth sample accuracy'])
        self.train_fig = fig
        display(fig)        
        
        # scatter plot real and fake data points
        if self.is_plot:
            #descaling real and fake data with scaler
            
            
            if len(x_real.shape) == 3:
                #x_fake for some reason is 4d, so we squeeze it
                dim0, dim1, dim2 = x_real.shape
                x_fake = np.squeeze(x_fake)
                
                x_real_vect = x_real.reshape(-1, x_real.shape[-2]*x_real.shape[-1])
                x_fake_vect = x_fake.reshape(-1, x_fake.shape[-2]*x_fake.shape[-1])
                
                x_real_vect_descaled = scaler.inverse_transform(x_real_vect)
                x_fake_vect_descaled = scaler.inverse_transform(x_fake_vect)
                
                x_real = x_real_vect_descaled.reshape(dim0, dim1, dim2)
                x_fake = x_fake_vect_descaled.reshape(dim0, dim1, dim2)
                
                x_real = np.clip(x_real, 0, 255).astype(int)
                x_fake = np.clip(x_fake, 0, 255).astype(int)
            


            print('shape of fake data: ', x_fake.shape)
            print('checking fake data: size 0, size 1, are both arrays identical. ', x_fake[0].shape, 
                  x_fake[1].shape, np.all(x_fake[0]==x_fake[1]))
            disparray = np.zeros((28*2,28*n))
            print(x_fake.shape)
            
            for k in range(n):
                #disparray[:28,28*k:28*(k+1)] = np.clip((np.squeeze(x_fake[k])*255).astype(int),0,255)
                #disparray[28:, 28*k:28*(k+1)] = np.clip((np.squeeze(x_real[k])*255).astype(int),0,255)
                
                disparray[:28, 28*k:28*(k+1)] = x_fake[k, :, :]
                disparray[28:, 28*k:28*(k+1)] = x_real[k, :, :]
            
            
            
                
            fig, axs = pyplot.subplots(1,1)
            axs.imshow(disparray, cmap='gray_r')
            
            """
            fig, (ax0, ax1) = pyplot.subplots(1,2)
            ax0.imshow(x_fake[0], cmap='gray_r')
            ax1.imshow(x_real[0], cmap='gray_r')
            """
            
            
            if self.is_plot:
                plt.show()
            res = "{:.2f}".format(acc_fake)
            plt.imsave(f'./epoch_{epoch}_accfake{res}.png', disparray)
            plt.close('all')
        if self.is_table:
            fake_then_real = np.concatenate([x_fake[:5], x_real[:5]], axis=0)
            temp_out = pd.DataFrame(fake_then_real)
            print(temp_out.shape)
            if scaler:
                display(pd.DataFrame(scaler.inverse_transform(temp_out)))
            else:
                display(temp_out)

            temp_out['fake/real'] = (5*['fake'])+(5*['real'])
            
        return "{:.2f}".format(acc_real), "{:.2f}".format(acc_fake)
        
    # train the generator and discriminator
    def train(self, 
              dataset, 
              scaler=None, 
              n_epochs=1, 
              n_batch=8,
              n_eval=400, #num of epoch intervals to do an eval process
              progress_bar=True,
              save_after_epoch_mult=10, 
              start_epoch=0, 
              file_prefix=None):
        
        print('**** now training gan ***')
        self.scaler = scaler
        # determine half the size of one batch, for updating the discriminator
        latent_dim = self.latent_dim
        g_model = self.generator
        d_model = self.discriminator
        gan_model = self.gan
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        #uri batches_per_epoch = int(dataset.shape[0]/n_batch)
        batches_per_epoch = self.batches_per_epoch

        for i in range(start_epoch, n_epochs):
            if progress_bar:
                f = IntProgress(min=0, max=batches_per_epoch) # instantiate the bar
                display(f) # display the bar
                count = 0
            for j in range(int(batches_per_epoch/1)):
                if progress_bar:
                    f.value += 1 # signal to increment the progress bar
                    count += 1
                x_real, y_real = self.generate_real_samples(half_batch, dataset)
                # prepare fake examples
                #single_value_func(n_batch)
                x_fake, y_fake = self.generate_fake_samples(g_model, latent_dim, half_batch)
                # update discriminator

                d_model.train_on_batch(x_real, y_real)

                d_model.train_on_batch(x_fake, y_fake)

                # prepare points in latent space as input for the generator
                x_gan = self.generate_latent_points(latent_dim, n_batch)
                # create inverted labels for the fake samples
                y_gan = ones((n_batch, 1))
                # update the generator via the discriminator's error
                gan_model.train_on_batch(x_gan, y_gan)
            if (i+1) % n_eval == 0:
                acc_real, acc_fake = self.summarize_performance(i, latent_dim, n=10, dataset=dataset, scaler=scaler)
            if (i+1) % save_after_epoch_mult == 0 or i+1 == n_epochs:
                print('>>> saving intermediate model')
                self.generator.save(f'{file_prefix}_temp_generator_epoch_{i}_{acc_real}_{acc_fake}.model')
                self.discriminator.save(f'{file_prefix}_temp_discriminator_epoch_{i}_{acc_real}_{acc_fake}.model')
                self.gan.save(f'{file_prefix}_temp_gan_epoch_{i}_{acc_real}_{acc_fake}.model')
        
        
    def lr_schedule_func(self, initial_lr):
        print('batches per epoch', self.batches_per_epoch)
        #print('in lr_scheduler, ', batches_per_epoch, self.decay_rate, self.lr)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=self.lr,
        decay_steps=float(self.batches_per_epoch,),#self.batches_per_epoch,
        decay_rate=self.decay_rate)
        return lr_schedule
    """
    def lr_schedule_func(self, batches_per_epoch=20):
        print('in lr_scheduler, ', batches_per_epoch, self.decay_rate, self.lr)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=self.lr,
        decay_steps=batches_per_epoch,
        decay_rate=self.decay_rate)
        return lr_schedule
    """
