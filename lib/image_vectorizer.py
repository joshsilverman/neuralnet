from IPython import embed
from PIL import Image
import numpy
import sys
import os

sys.path.append("/Users/joshsilverman/Dropbox/Apps/ai-hw4")
sys.path.append("/Users/joshsilverman/Dropbox/Apps/ai-hw4/lib")
sys.path.append("/Users/joshsilverman/Dropbox/Apps/ai-hw4/images")

class ImageVectorizer:
  def get_image_paths(self):
    images = []
    for name in os.listdir("images"):
      if name.endswith(".png"): #and name.startswith("number5"): 
        image_path = "images/" + name
        images.append(image_path)

    return images

  def image_to_vector(self, path):
    image = Image.open(path) #Can be many different formats.
    image_vector = numpy.asarray(image).reshape(-1)
    return image_vector

  def vectors_to_images(self, vectors, prefix):
    for i, vector in enumerate(vectors):
      name = "output/%(prefix)s_neuron_%(i)i.png" % locals()
      self.vector_to_image(vector, name)

  def vector_to_image(self, vector, name):
    projection = vector - min(vector)
    projection = projection * 255 / max(projection)

    pxs = numpy.reshape(projection, (28, 28))
    im = Image.fromarray(pxs).convert('RGB')
    im.save(name, "PNG")