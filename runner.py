import sys
sys.path.append("/Users/joshsilverman/Dropbox/Apps/ai-hw4/lib")

from IPython import embed
from neural_net import NeuralNet
from image_vectorizer import ImageVectorizer


vectorizer = ImageVectorizer()
paths = vectorizer.get_image_paths()

############################################################
################### Assignment Questions ###################
############################################################

############################################################
# Train a feedforward neural network with one hidden layer #
# of size 3 to learn representations of those digits.      #
# Try using (a) Linear transform function                  #
############################################################ 

# net_3lin = NeuralNet(3, 'linear')
# net_3lin.train(paths)
# weights = net_3lin.input_weights_of_hidden_layer()
# vectorizer.vectors_to_images(weights)

############################################################
# (b) Sigmoid transform function for the hidden layer      #
############################################################

net_3sig = NeuralNet(3, 'sigmoid')
net_3sig.train(paths)
weights = net_3sig.input_weights_of_hidden_layer()
vectorizer.vectors_to_images(weights, '3_hidden_layer_sigmoid')

############################################################
# Change the size of hidden layer to 6 and retrain         #
############################################################

net_6sig = NeuralNet(6, 'sigmoid')
net_6sig.train(paths)
weights = net_6sig.input_weights_of_hidden_layer()
vectorizer.vectors_to_images(weights, '6_hidden_layer_sigmoid')

############################################################
# Change the size of hidden layer to 9 and retrain         #
############################################################

net_9sig = NeuralNet(9, 'sigmoid')
net_9sig.train(paths)
weights = net_9sig.input_weights_of_hidden_layer()
vectorizer.vectors_to_images(weights, '9_hidden_layer_sigmoid')
