# Teaching a 3 level neural network to work as Full Adder
from numpy import random, exp, array, dot
import pandas as pd

class NeuralNetwork():
  def __init__(self, gateInput, gateOutput, ):
    random.seed(1)
    self.input = gateInput
    self.output = gateOutput
    self.input_shape = (1,3)
    self.output_shape = (1,2)
    self.layer_1_nodes = 10
    self.layer_2_nodes = 10
    self.layer_3_nodes = 10

    self.weights_1 = 2 * random.random((self.input_shape[1], self.layer_1_nodes)) - 1
    self.weights_2 = 2 * random.random((self.layer_1_nodes, self.layer_2_nodes)) - 1
    self.weights_3 = 2 * random.random((self.layer_2_nodes, self.layer_3_nodes)) - 1
    self.out_weights = 2 * random.random((self.layer_3_nodes, self.output_shape[1])) - 1

  def sigmoid(self, x):
    return 1 / (1 + exp(-x))

  def sigmoid_derivative(self, x):
    return x * (1 - x)

  def think(self, x):
    layer1 = dot(x, self.weights_1)
    layer2 = dot(self.sigmoid(layer1), self.weights_2)
    layer3 = dot(self.sigmoid(layer2), self.weights_3)
    output = dot(self.sigmoid(layer3), self.out_weights)
    return self.sigmoid(output)

  def train(self, num_steps):
    for x in range(num_steps):
      layer1 = dot(self.input, self.weights_1)
      layer2 = dot(self.sigmoid(layer1), self.weights_2)
      layer3 = dot(self.sigmoid(layer2), self.weights_3)
      output = dot(self.sigmoid(layer3), self.out_weights)

      error = self.output - self.sigmoid(output)

      out_weights_adjustment = dot(layer3.T, error * self.sigmoid_derivative(output))

      temp = self.out_weights
      self.out_weights += out_weights_adjustment
      error = self.out_weights - temp
      
      weight_3_adjustment = dot(layer2.T, error * self.sigmoid_derivative(layer3))

      temp = self.weights_3
      self.weights_3 += weight_3_adjustment
      temp = self.weights_3 - temp

      weight_2_adjustment = dot(layer1.T, temp * self.sigmoid_derivative(self.weights_3))

      temp = self.weights_2
      self.weights_2 += weight_2_adjustment
      temp = self.weights_2 - temp

      weight_1_adjustment = dot(self.input.T, temp * self.sigmoid_derivative(self.weights_2))
      
      self.weights_1 += weight_1_adjustment


if __name__ == '__main__':
  file = pd.read_csv("dataset.txt", delimiter=',')

  dataset = file.values

  gateInput = dataset[:,:3]
  gateOutput = dataset[:,3:]

  neural_network = NeuralNetwork(gateInput, gateOutput)

  neural_network.train(1000)

  # Should be 0 , 1
  print(neural_network.think([[1,0,0]]))