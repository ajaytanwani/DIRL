import pickle as pkl
import scipy.io as sio
import h5py
import numpy as np
from skimage.transform import resize
import os

class MNISTDataset(object):

    def __init__(self, datasets_directory=''):
      mnist = self.load_dataset("mnist", datasets_directory)
      mnistm = self.load_dataset("mnistm", datasets_directory)
      svhn = self.load_dataset("svhn", datasets_directory)
      usps = self.load_dataset("usps", datasets_directory)

      mnistm_train = mnistm['train']/255
      mnistm_test = mnistm['test']/255
      mnistm_valid = mnistm['valid']/255

      svhn_x_train, svhn_y_train, svhn_x_test, svhn_y_test = svhn
      usps_x_train, usps_y_train, usps_x_test, usps_y_test = usps

      mnist_train = mnist.train.images.reshape(-1, 28, 28, 1)
      mnist_test = mnist.test.images.reshape(-1, 28, 28, 1)
      mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
      mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    
      usps_x_train = np.concatenate([usps_x_train, usps_x_train, usps_x_train], 3)
      usps_x_test = np.concatenate([usps_x_test, usps_x_test, usps_x_test], 3)

      def get_respective_sets(dtype):
          if dtype == "mnist":
              return mnist_train, mnist_test, mnist.train.labels, mnist.test.labels
          elif dtype == "mnistm":
              return mnistm_train, mnistm_test, mnist.train.labels, mnist.test.labels
          elif dtype == "svhn":
              return svhn_x_train, svhn_x_test, svhn_y_train, svhn_y_test    
          elif dtype == "usps":
              return usps_x_train, usps_x_test, usps_y_train, usps_y_test
          else:
              print("data type not supported:", dtype)

      self.get_dataset = get_respective_sets


    def imgs_reshape(self, data, t_size = 28):
        side_dim = data.shape[1]
        channels = data.shape[3]
        data = data.transpose((1, 2, 3, 0)) 
        data = resize(data.reshape(side_dim, side_dim, -1), (t_size, t_size)) 
        data = data.reshape(t_size, t_size, channels, -1)
        data = data.transpose((3, 0, 1, 2))
        return data

    def load_dataset(self, name, datasets_directory=''):

        if name == "mnist":
            #read the data in
            from tensorflow.examples.tutorials.mnist import input_data
            return input_data.read_data_sets(os.path.join(datasets_directory, "MNIST_data/"), one_hot=True)
        elif name == "mnistm":
            return pkl.load(open(os.path.join(datasets_directory, "mnistm_data.pkl"), 'rb'))
        elif name == "svhn":
            train_data = sio.loadmat(os.path.join(datasets_directory, 'svhn/train_32x32.mat'))
            test_data = sio.loadmat(os.path.join(datasets_directory, 'svhn/test_32x32.mat'))
            x_train = train_data['X']
            y_train = train_data['y']
            x_test = test_data['X']
            y_test = test_data['y']
            x_train = np.transpose(x_train, (3, 0, 1, 2))
            x_test = np.transpose(x_test, (3, 0, 1, 2))
            
            #convert to onehot
     
            y_train %= 10
            y_test %= 10
            
            n_values = 10
            y_train_oh = np.eye(n_values)[y_train]
            y_test_oh = np.eye(n_values)[y_test]
            y_train_oh = np.reshape(y_train_oh, (-1, n_values))
            y_test_oh = np.reshape(y_test_oh, (-1, n_values))

            
            return self.imgs_reshape(x_train), y_train_oh, self.imgs_reshape(x_test), y_test_oh
        elif name == "usps":
            with h5py.File(os.path.join(datasets_directory, "usps/usps.h5"), 'r') as hf:
                train = hf.get('train')
                test = hf.get('test')
                x_train = train.get('data')[:]
                y_train = train.get('target')[:]
                x_test = test.get('data')[:]
                y_test = test.get('target')[:]
                
                x_train = x_train.reshape(-1, 16, 16, 1)
                x_test = x_test.reshape(-1, 16, 16, 1)
                
                n_values = 10
                y_train_oh = np.eye(n_values)[y_train]
                y_test_oh = np.eye(n_values)[y_test]
                y_train_oh = np.reshape(y_train_oh, (-1, n_values))
                y_test_oh = np.reshape(y_test_oh, (-1, n_values))

                return self.imgs_reshape(x_train), y_train_oh, self.imgs_reshape(x_test), y_test_oh
