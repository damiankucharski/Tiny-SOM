import numpy as np
import copy 

class Map:
    """
    Class implementing Kohonnen Neural Network - Self organising map.
    """

    def __init__(self, shape, input_shape, lr, sigma = None, _pilambda = None):
        """
        Kohonnen neural network initializer

        Parameters:

        shape (tuple): shape of neural network. Typically it would be some square for example (20,20)
        input_shape (int): input features dimention.
        lr (float): learning rate
        sigma (float, optional): neighbourhood function radius. If not specified ut will be set as max(shape) / 2
        _pilambda (float, optional): time constant for neighbourhood function. If not specified then it will be calculated as n_iterations / log(sigma)
        size (int): the length of a square grid of neurons

        """
        self.lr = lr
        if sigma:
            self.sigma = sigma
        else:
            self.sigma = max(shape) / 2
        self._pilambda = _pilambda

        self.shape = shape
        self.input_shape = input_shape

        self.weights = np.random.random((*shape, input_shape))
        self.indices_matrix = np.indices(self.shape)
        self.indices_matrix = self.indices_matrix.reshape(self.indices_matrix.shape[0],-1).T

    
    def calc_distances(self, matrix, point):

        distances = np.sqrt(np.sum(np.square(matrix - point.reshape(1,-1)), axis = -1))

        return distances

    def find_bmu(self, distances):

        min_arg = np.argmin(distances)
        index = np.unravel_index(min_arg, self.shape)
        return index
    
    def calc_topological_nbh(self, point, epoch):
        
        nbh_size = self.sigma * np.exp(-epoch/self._pilambda)
        distances = self.calc_distances(self.indices_matrix, point)
        distances = distances.reshape(self.shape)
        nbh =  np.exp(-distances/2/np.square(nbh_size))
        
        nbh *= nbh <= nbh_size
        assert nbh.shape == self.shape
    
        return nbh

    def update_weights(self, point, epoch, sample, epochs):
        """
        Performs single update step

        Arguments:

        point (array-like of shape (input_shape)): this epoch's BMU
        epoch (int): epoch number
        sample (int): random sample to which weights are optimized in current epoch
        epochs (int): integer specifing number of epochs som is being trained for
        """        
        lr = self.lr * np.exp(-epoch/epochs)
        nbh = self.calc_topological_nbh(point, epoch)
        delta_weights = lr * np.expand_dims(nbh,-1) * (sample - self.weights)
        
        assert delta_weights.shape == (*self.shape, self.input_shape)

        self.weights += delta_weights
        return

    def fit(self, X, epochs = 100):
        
        if not self._pilambda:
            self._pilambda = epochs / np.log(self.sigma)
        self.cache = []
        for epoch in range(epochs):
            self.cache.append(copy.deepcopy(self.weights))
            sample = X[np.random.randint(X.shape[0]),...]
            assert len(sample) == self.input_shape
            distances = self.calc_distances(self.weights, sample)
            bmu_index = self.find_bmu(distances)
            point = self.weights[bmu_index]

            assert point.shape == sample.shape
            bmu_index = np.array(bmu_index)
            self.update_weights(bmu_index, epoch, sample, epochs)
        return 


