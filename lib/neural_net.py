from IPython import embed

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet

import numpy
from image_vectorizer import ImageVectorizer

class NeuralNet:
  def __init__(self, hidden_neuron_num=1, hidden_type='sigmoid'):
    self.hidden_neuron_num = hidden_neuron_num
    self.hidden_type = hidden_type

    self.net = FeedForwardNetwork()
    self.samples = SupervisedDataSet(784, 784)

    self.vectorizer = ImageVectorizer()

    self.add_layers()
    self.add_connections()
    self.sort()

  def add_layers(self):
    self.inLayer = LinearLayer(784, name='in')
    self.outLayer = LinearLayer(784, name='out')

    if self.hidden_type == 'sigmoid':
      self.hiddenLayer = SigmoidLayer(self.hidden_neuron_num, name='hidden')
    else:
      self.hiddenLayer = LinearLayer(self.hidden_neuron_num, name='hidden')

    self.net.addInputModule(self.inLayer)
    self.net.addModule(self.hiddenLayer)
    self.net.addOutputModule(self.outLayer)

  def add_connections(self):
    self.in_to_hidden = FullConnection(self.inLayer, self.hiddenLayer)
    self.hidden_to_out = FullConnection(self.hiddenLayer, self.outLayer)

    self.net.addConnection(self.in_to_hidden)
    self.net.addConnection(self.hidden_to_out)

  def sort(self):
    self.net.sortModules()

  def activate(self, vector):
    return self.net.activate(vector)

  def train(self, paths):
    for path in paths:
      vector = self.vectorizer.image_to_vector(path) 
      vector = [el / 255.0 for el in vector]
      self.samples.addSample(vector, vector)

    trainer = BackpropTrainer(self.net, self.samples, learningrate=.5, lrdecay=0.98)
    for i in range(1,20):
      error = trainer.train()
      print "error for %(i)ith iteration: %(error)f" % locals()

  def input_weights_of_hidden_layer(self):
    weights = self.in_to_hidden.params
    hidden_weights_by_neuron = numpy.split(weights, self.hidden_neuron_num)
    return hidden_weights_by_neuron

  def input_weights_of_out_layer(self):
    weights = self.hidden_to_out.params
    hidden_weights_by_neuron = numpy.split(weights, self.hidden_neuron_num)
    return hidden_weights_by_neuron