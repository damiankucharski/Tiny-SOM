import numpy as np

class Map:
    """
    Class implementing Kohonnen Neural Network - Self organising map.
    """

    def __init__(self, shape, input_shape, _alpha, _sigma, _lambda):
        """
        Kohonnen neural network initializer

        Parameters:

        shape (tuple): shape of neural network. Typically it would be some square for example (20,20)
        input_shape (int): input features dimention.
        _alpha (float): learning rate
        _sigma (float): neighbourhood function radius
        _lambda (float): time constant
        size (int): the length of a square grid of neurons

        """
        self._alpha = _alpha
        self._sigma = _sigma
        self._lambda = _lambda

        self.shape = shape
        self.input_shape = input_shape

        self.weights = np.random.randn(*shape, input_shape)

    
    def calc_distances(self, point):
        distances = np.sqrt(np.sum(np.square(self.weights - point.reshape(1,-1)), axis = -1))
        
        assert distances.shape == self.shape
        
        return distances

    def find_bmu(self, distances):
        min_arg = np.argmin(distances)
        return np.unravel_index([min_arg], self.shape)[0]

    def update_weights(self, epoch):
        lr = self._alpha * np.exp(-epoch/self._lambda)

    def fit(self, X, epochs = 100):
        pass



data = np.array([
    [255,0,0],
    [0,255,0],
    [0,0,255]
])



test_map = Map((10,10),3,0.5,0.5,0.5)
print(test_map.weights.shape)
test_map.calc_distances(data[0])