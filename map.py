import numpy as np

class Map:
    """
    Class implementing Kohonnen Neural Network - Self organising map.
    """

    def __init__(self, shape, input_shape, _alpha, _sigma, _lambda, _pilambda):
        """
        Kohonnen neural network initializer

        Parameters:

        shape (tuple): shape of neural network. Typically it would be some square for example (20,20)
        input_shape (int): input features dimention.
        _alpha (float): learning rate
        _sigma (float): neighbourhood function radius
        _lambda (float): time constant for learning rate
        _pilambda (float): time constant for neighbourhood function
        size (int): the length of a square grid of neurons

        """
        self._alpha = _alpha
        self._sigma = _sigma
        self._lambda = _lambda
        self._pilambda = _pilambda

        self.shape = shape
        self.input_shape = input_shape

        self.weights = np.random.randn(*shape, input_shape)

    
    def calc_distances(self, point):
        distances = np.sqrt(np.sum(np.square(self.weights - point.reshape(1,-1)), axis = -1))
        
        assert distances.shape == self.shape
        
        return distances

    def find_bmu(self, distances):

        min_arg = np.argmin(distances)
        index = np.unravel_index(min_arg, self.shape)
        return index
    
    def calc_topological_nbh(self, point, epoch):
        
        nbh_size = self._sigma * np.exp(-epoch/self._pilambda)
        distances = self.calc_distances(point)
        nbh =  np.exp(-np.square(distances)/2/np.square(nbh_size))
    
        assert nbh.shape == self.shape
    
        return nbh

    def update_weights(self, point, epoch, sample):
        """
        Performs single update step

        Arguments:

        point (array-like of shape (input_shape)): this epoch's BMU
        epoch (int): epoch number
        sample (int): random sample to which weights are optimized in current epoch

        """        
        lr = self._alpha * np.exp(-epoch/self._lambda)
        nbh = self.calc_topological_nbh(point, epoch)

        delta_weights = lr * np.expand_dims(nbh,-1) * (sample - self.weights)
        
        assert delta_weights.shape == (*self.shape, self.input_shape)

        self.weights += delta_weights

        return

    def fit(self, X, epochs = 100):
        
        self.cache = []
        for epoch in range(epochs):
            self.cache.append(self.weights)
            sample = X[np.random.randint(X.shape[0]),...]
            assert len(sample) == self.input_shape
            distances = self.calc_distances(sample)
            bmu_index = self.find_bmu(distances)
            point = self.weights[bmu_index]

            # print(bmu_index)
            # print(sample.shape)
            # print(point.shape)
            assert point.shape == sample.shape
            self.update_weights(point, epoch, sample)
        return 


