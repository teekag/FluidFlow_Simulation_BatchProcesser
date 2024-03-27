# models/rom_builder.py

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ROMBuilder:
    def __init__(self):
        pass

    def train_rom(self, training_data):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2))
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        self.model.fit(training_data['inputs'], training_data['outputs'])

    def predict(self, inputs):
        return self.model.predict(inputs)
