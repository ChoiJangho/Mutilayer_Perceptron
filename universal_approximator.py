import numpy as np
import matplotlib.pyplot as plt

def identity(xs):
    return xs

def heaviside_step_func(xs):
    return np.array(xs >= 0, dtype=int)

def get_sample_data(func, xs):
    return func(xs)

class TwoLayerNetwork:
    def __init__(self, input_dim=1, output_dim=1, hidden_units=3):
        """
        :param input_dim:
        :param output_dim:
        :param hidden_units:
        activation is fixed as tanh
        """
        self._input_dim = input_dim
        self._hidden_units = hidden_units
        self._output_dim = output_dim
        self._w_first = 0.2 * np.random.randn(input_dim+1, hidden_units)
        self._activation = np.tanh
        self._w_second = 0.2 * np.random.randn(hidden_units+1, output_dim)
        self._learning_rate= 0.05
        self._error = 0.
        self._last_error = 0.

    def forward_propagation(self, inputs):
        """
        :param inputs: numpy array of input (num of data X dim)
        :return:
        """
        inputs = np.concatenate((np.ones((inputs.shape[0], 1)), inputs), axis=1)
        activations = np.matmul(self._w_first.T, inputs.T).T
        hidden_units = self._activation(activations)
        hidden_units = np.concatenate(
            (np.ones((hidden_units.shape[0], 1)), hidden_units), axis=1
        )
        outputs = np.matmul(self._w_second.T, hidden_units.T).T
        return activations, hidden_units, outputs

    def evaluate(self, inputs, targets):
        _, _, outputs = self.forward_propagation(inputs)
        errors = np.zeros(targets.shape[0])
        for i in range(targets.shape[0]):
            errors[i] = np.linalg.norm(outputs[i, :] - targets[i, :])
        error = np.sum(errors)
        print(error)
        return error

    def backward_propagation(self, inputs, targets):
        activations, hiddens, outputs = self.forward_propagation(inputs)
        inputs = np.concatenate((np.ones((inputs.shape[0], 1)), inputs), axis=1)
        delta_output = outputs - targets
        temp = np.tile(hiddens, (self._output_dim, 1, 1)).transpose(1, 0, 2)
        first_layer_derivative = np.zeros(
            (targets.shape[0], self._hidden_units, self._input_dim+1))
        second_layer_derivative = np.zeros((targets.shape[0], self._output_dim, self._hidden_units+1))
        for i in range(targets.shape[0]):
            second_layer_derivative[i, :, :] = np.multiply(delta_output[i, :], temp[i, :, :])
            delta_hidden = np.tile((1 - hiddens[i, :] ** 2) * np.matmul(
                self._w_second, delta_output[i, :].T), (1, self._input_dim)).T
            first_layer_derivative[i, :, :] = np.multiply(delta_hidden[1:, :], inputs[i, :])
        return np.reshape(first_layer_derivative, (self._input_dim + 1, self._hidden_units)), \
            np.reshape(second_layer_derivative, (self._hidden_units + 1, self._output_dim))

    def train(self, inputs, targets):
        """ Stochastic Gradient Descent
        """
        while True:
            for i in range(targets.shape[0]):
                first_layer_derivative, second_layer_derivative = \
                    self.backward_propagation(
                        np.expand_dims(inputs[i, :], axis=0).T,
                        np.expand_dims(targets[i, :], axis=0).T
                    )

                self._w_first = self._w_first - self._learning_rate * first_layer_derivative
                self._w_second = self._w_second - self._learning_rate * second_layer_derivative
            self._last_error = self._error
            self._error = self.evaluate(inputs, targets)
            if abs(self._error) < 0.07:
                break



xs = np.expand_dims(np.linspace(-1, 1, num=20), axis=0).T
ts = get_sample_data(identity, xs)

two_layer_network = TwoLayerNetwork()
two_layer_network.train(xs, ts)
_, _, ys = two_layer_network.forward_propagation(xs)

plt.figure(1)
plt.plot(xs, ys, 'r--')
plt.show()


