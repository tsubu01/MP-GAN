#sources:
# The training flow implemented in this class is based on the following source:
#https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/

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


class GAN():
    """
    This class handles the creation and training of a GAN model.
    To train a GAN from scratch, first define a class instance. Then
    use the define_discriminatror() and define_generator() functions. with
    the API defined in deep_model.py. Once the GAN is prepared, call the train()
    function.
    To generate new samples with a trained GAN, take the GAN's generator object
    and use its built-in predict() function, with a latent-dim - dimensionl vector as input
    (or an array of vectors, for multiple simultaneous predictions).
    is_table indicates if the data is tabular or image.
    For standardizing the data you can use sklearn.StandardScaler before training and
    pass the scaler to the GAN. Images currently don't take a scaler (pass None) and are standardized
    internally by division by 255.
    """
    def __init__(self, isplot, 
                 keepshape=True, 
                 is_table=False,
                 lr=0.0002, 
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
        self.decay_rate = decay_rate
        self.real_sample_metrics = {}
        self.fake_sample_metrics = {}
        self.decayed_lr = {}
        self.train_fig = None
        self.batches_per_epoch = 0
        self.epoch = 0
        self.decay_rate = decay_rate
        self.scaler = scaler
        self.train_discriminator = True
    
    
    def _learning_rate_scheduler(self):
        return(self.lr * self.decay_rate ** (-(self.epoch)))

    def define_discriminator(self, n_inputs, layers,optimizer=Adam):
        #layers is a list of lists; each member is a 3-member list with the format: [layer_type, layer/kernel size, activation]
        self.n_inputs = n_inputs        
        #model needs to be compiled because we use it as a standalone model for training on real data.
        # compile model
        model = DeepModel(self.n_inputs, layers).create_model()
        if optimizer == Adam:
            self.disc_opt = Adam(learning_rate=self._learning_rate_scheduler(), beta_1=0.5)
        else:
            self.disc_opt = optimizer
        print('discriminator optimizer: ', self.disc_opt)
        model.compile(loss='binary_crossentropy', optimizer=self.disc_opt, metrics=['accuracy'])
        self.discriminator = model
        print('Discriminator is now defined')
    
    def define_generator(self, latent_dim, layers):
        self.latent_dim = latent_dim        
        model = DeepModel(self.latent_dim, layers).create_model()
        self.generator = model
        print('Generator is now defined')
        
    def define_gan(self, optimizer=Adam):
        #make discriminator weights not trainable.
        discriminator = self.discriminator
        generator = self.generator
        discriminator.trainable = False
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        if optimizer == Adam:
            self.gan_opt = Adam(learning_rate=self._learning_rate_scheduler(), beta_1=0.5)
        else:
            self.gan_opt = optimizer
        print('gan optimizer: ', self.gan_opt)
        model.compile(loss='binary_crossentropy',
                      optimizer=self.gan_opt,
                      metrics=['accuracy'])
        self.gan = model
        print('GAN is now defined')
     
    def generate_real_samples(self, n, realsamplearray):
        keepshape=self.keep_shape
        # generate mnist samples
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
        y = ones((n, 1))
        if self.is_table:
            return X, y
        else:
            if self.scaler:
                return X, y
            else:
                return X.astype('float32') / 255, y
         
    def generate_latent_points(self, latent_dim, n):
        # generate points in the latent space
        x_input = randn(latent_dim * n)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n, latent_dim)
        return x_input
 
    def generate_fake_samples(self, generator, latent_dim, n):
        # generate points in latent space
        x_input = self.generate_latent_points(latent_dim, n)
        # predict outputs
        X = self.generator.predict(x_input, verbose=False)
        # create class labels
        y = zeros((n, 1))
        return X, y
                
    def summarize_performance(self, 
                              epoch, 
                              latent_dim, 
                              n=200, 
                              dataset=None, 
                              scaler=None, 
                              save_after_epoch_mult=10,
                              file_prefix=''):
        
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
        self.decayed_lr[epoch] = self._learning_rate_scheduler()
        clear_output(wait = True)
        print('epoch number: ', epoch)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1) 
        
        epoch_array = [i[0] for i in self.real_sample_metrics.items()]
        real_sample_metrics_array = [i[1] for i in self.real_sample_metrics.items()]
        fake_sample_metrics_array = [i[1] for i in self.fake_sample_metrics.items()]

        ax.cla()
        plt.plot(epoch_array, real_sample_metrics_array)
        plt.plot(epoch_array, fake_sample_metrics_array)
        plt.xlabel('epoch')
        plt.ylabel('discriminator accuracy')
        ax.set_ylim([0, 1])
        plt.legend(['on real samples', 'on synthetic samples'])
        self.train_fig = fig
        display(fig)        
        
        # scatter plot real and fake data points
        if self.is_plot:
            #descaling real and fake data with scaler            
            if len(x_real.shape) == 3:
                #x_fake is 4d, so we squeeze it
                dim0, dim1, dim2 = x_real.shape
                x_fake = np.squeeze(x_fake)
                if scaler:
                    print('*** scaler is found')
                    x_real_vect = x_real.reshape(dim0, dim1 * dim2)
                    x_fake_vect = x_fake.reshape(dim0, dim1 * dim2)
                
                    x_real_vect_descaled = scaler.inverse_transform(x_real_vect)
                    x_fake_vect_descaled = scaler.inverse_transform(x_fake_vect)
                
                    x_real = x_real_vect_descaled.reshape(dim0, dim1, dim2)
                    x_fake = x_fake_vect_descaled.reshape(dim0, dim1, dim2)
                
                if not scaler:
                    print('*** scaler is none')
                    x_real = 255 * x_real
                    x_fake = 255 * x_fake
                  
                x_real = np.clip(x_real, 0, 255).astype(int)
                x_fake = np.clip(x_fake, 0, 255).astype(int)
            
            disparray = np.zeros((28 * 2, 28 * n))
            for k in range(n):     
                disparray[:28, 28 * k:28 * (k + 1)] = x_fake[k, :, :]
                disparray[28:, 28 * k:28 * (k + 1)] = x_real[k, :, :]

            fig, axs = pyplot.subplots(1, 1)
            axs.imshow(disparray, cmap='gray_r')

            if self.is_plot:
                plt.show()
            if (epoch + 1) % save_after_epoch_mult == 0:
                res = "{:.2f}".format(acc_fake)
                plt.imsave(f'./{file_prefix}_epoch_{epoch}_accfake{res}.png', disparray, cmap='gray_r')
            plt.close('all')
            
        if self.is_table:
            fake_then_real = np.concatenate([x_fake[:5], x_real[:5]], axis=0)
            temp_out = pd.DataFrame(fake_then_real)
            if scaler:
                display(pd.DataFrame(scaler.inverse_transform(temp_out)))
            else:
                display(temp_out)
            temp_out['fake/real'] = (5*['fake'])+(5*['real'])
            
        return "{:.2f}".format(acc_real), "{:.2f}".format(acc_fake)
        
    def train(self, 
              dataset, 
              scaler=None, 
              n_epochs=1, 
              n_batch=8,
              n_eval=400, #num of epoch intervals to do an eval process
              progress_bar=True,
              save_after_epoch_mult=10,
              train_discriminator=True,
              file_prefix=''):
        """
        The training method.
        If you are training an already trained GAN, the epoch count is remembered. Pass the
        new number of epochs you want the training to reach in n_epochs.
        Args:
            dataset - np array - [n, h, w] for images or [n, w] for tables.
            scaler - see __init__()
            n_batch - batch size (int)
            n_eval - number of epochs (int)
            progress_bar - (bool) display a bar that fills every epoch. Note that you may need to run
            jupyter nbextension enable --py --sys-prefix widgetsnbextension
            after installing the requirements.txt in order for it to work in jupyter.
            save_after_epoch_mult - the epoch increments for checkpoint saving (int).
            train_discriminator - whether the discriminator should be also trained. (bool)
            file_prefix - to identify your saved files quickly when running multiple experiments. (str).
        Returns:
            None.
            The trained generator will be found in self.generator and can be used to generate data
            by self.generator.predict(...)
            
        """
        self.train_discriminator = train_discriminator
        print('**** now training gan ***')
        print('**** training discriminator: {}'.format(self.train_discriminator))
        epoch_array = [i[0] for i in self.real_sample_metrics.items()]
        if len(epoch_array) > 0:
            start_epoch = epoch_array[-1] + 1
        else:
            start_epoch = 0
        self.scaler = scaler
        # determine half the size of one batch, for updating the discriminator
        latent_dim = self.latent_dim
        g_model = self.generator
        d_model = self.discriminator
        gan_model = self.gan
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        self.batches_per_epoch = int(dataset.shape[0] / n_batch)
        print('**** batches per epoch: ', self.batches_per_epoch)

        for i in range(start_epoch, n_epochs):
            self.epoch = i
            if progress_bar:
                f = IntProgress(min=0, max=self.batches_per_epoch) # instantiate the bar
                display(f) # display the bar
                count = 0
            for j in range(int(self.batches_per_epoch)):
                if progress_bar:
                    f.value += 1 # signal to increment the progress bar
                    count += 1
                if self.train_discriminator:
                    x_real, y_real = self.generate_real_samples(half_batch, dataset)
                    # prepare fake examples
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
                acc_real, acc_fake = self.summarize_performance(i, 
                                                                latent_dim, 
                                                                n=10, 
                                                                dataset=dataset, 
                                                                scaler=scaler, 
                                                                save_after_epoch_mult=save_after_epoch_mult,
                                                                file_prefix=file_prefix)
            if (i+1) % save_after_epoch_mult == 0 or i + 1 == n_epochs:
                print('>>> saving intermediate model')
                self.generator.save(f'{file_prefix}_temp_generator_epoch_{i}_{acc_real}_{acc_fake}.model')
                self.discriminator.save(f'{file_prefix}_temp_discriminator_epoch_{i}_{acc_real}_{acc_fake}.model')
                self.gan.save(f'{file_prefix}_temp_gan_epoch_{i}_{acc_real}_{acc_fake}.model')
