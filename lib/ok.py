from IPython import embed

# from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet

from PIL import Image
import sys
import numpy
import os

sys.path.append("/Users/joshsilverman/Dropbox/Apps/ai-hw4")
sys.path.append("/Users/joshsilverman/Dropbox/Apps/ai-hw4/lib")
sys.path.append("/Users/joshsilverman/Dropbox/Apps/ai-hw4/images")


class NN:
  def __init__(self, hidden_neuron_num=1, hidden_type='sigmoid'):
    self.hidden_neuron_num = hidden_neuron_num
    self.hidden_type = hidden_type

    self.net = FeedForwardNetwork()
    self.samples = SupervisedDataSet(784, 784)

    self.add_layers()
    self.add_connections()
    self.sort()

  def add_layers(self):
    self.inLayer = LinearLayer(784, name='in')
    self.outLayer = LinearLayer(784, name='out')

    print self.hidden_type
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
      vector = get_image_vector(path) 
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

def get_image_paths():
  images = []
  for name in os.listdir("images"):
    if name.endswith(".png") and name.startswith("number5"): 
      image_path = "images/" + name
      images.append(image_path)
      print name

  return images

def get_image_vector(path):
  image = Image.open(path) #Can be many different formats.
  image_vector = numpy.asarray(image).reshape(-1)
  return image_vector

def vectors_to_images(vectors):
  for vector in vectors:
    vector_to_image(vector)

def vector_to_image(vector):
  projection = vector - min(vector)
  projection = projection * 255 / max(projection)

  # print projection
  pxs = numpy.reshape(projection, (28, 28))
  im = Image.fromarray(pxs)
  im.show()

paths = get_image_paths()

net_3lin = NN(3, 'linear')
net_3lin.train(paths)
weights_by_neuron = net_3lin.input_weights_of_hidden_layer()
vectors_to_images(weights_by_neuron)

# net_3sig = NN(3, 'sigmoid')
# net_3sig.train(paths)
# weights_by_neuron = net_3sig.input_weights_of_hidden_layer()
# vectors_to_images(weights_by_neuron)

# net_6sig = NN(6, 'sigmoid')
# net_6sig.train(paths)
# weights_by_neuron = net_6sig.input_weights_of_hidden_layer()
# vectors_to_images(weights_by_neuron)

# net_9sig = NN(9, 'sigmoid')
# net_9sig.train(paths)
# weights_by_neuron = net_9sig.input_weights_of_hidden_layer()
# vectors_to_images(weights_by_neuron)
