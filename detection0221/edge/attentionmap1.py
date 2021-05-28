import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.resnet50 import preprocess_input
import os
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

resnet_50 = tf.keras.applications.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
                                               
                                               
filePath = 'D:/myworklist/deeplearning/paper/refinenetdeblur/paper/debluredge/AttentioNN-master/images/heatmap/'
count=0
for i in os.listdir(filePath):
  #img = cv2.imread("cat.jpg")[:,:,::-1]
  img = cv2.imread("D:/myworklist/deeplearning/paper/refinenetdeblur/paper/debluredge/AttentioNN-master/images/heatmap/"+i)[:,:,::-1]
  #img = cv2.resize(img, (224, 224))
  #plot_heatmaps(range(164, 169))
  #plot_heatmaps(range(76, 81))
  #plot_heatmaps(range(3, 8))
  #ax = plt.imshow(img)
  cv2.imwrite("D:/myworklist/deeplearning/paper/refinenetdeblur/paper/debluredge/AttentioNN-master/images/result1231/"+i+".png",img)


def preprocess(img):
  # use the pre processing function of ResNet50 
  img = preprocess_input(img)
  
  #expand the dimension
  return np.expand_dims(img, 0)

input_image = preprocess(img)

def get_activations_at(input_image, i):
  #index the layer 
  out_layer = resnet_50.layers[i]
  
  #change the output of the model 
  model = tf.keras.models.Model(inputs = resnet_50.inputs, outputs = out_layer.output)
  
  #return the activations
  return model.predict(input_image)

def postprocess_activations(activations):

  #using the approach in https://arxiv.org/abs/1612.03928
  output = np.abs(activations)
  output = np.sum(output, axis = -1).squeeze()

  #resize and convert to image 
  output = cv2.resize(output, (224, 224))
  output /= output.max()
  output *= 255
#  return 255 - output.astype('uint8')
  return output.astype('uint8')
def apply_heatmap(weights, img):
  #generate heat maps 
  heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
  heatmap = cv2.addWeighted(heatmap, 0.7, img, 0.3, 0)
  return heatmap

def plot_heatmaps(rng):
  level_maps = None
  
  #given a range of indices generate the heat maps 
  for j in rng:
    activations = get_activations_at(input_image, j)
    weights = postprocess_activations(activations)
    heatmap = apply_heatmap(weights, img)
    if level_maps is None:
      level_maps = heatmap
    else:
      level_maps = np.concatenate([level_maps, heatmap], axis = 1)
  plt.figure(figsize=(15, 15))
  plt.axis('off')
  ax = plt.imshow(level_maps)
  #cv2.imwrite("D:/myworklist/deeplearning/paper/refinenetdeblur/paper/debluredge/AttentioNN-master/"+str(i)+".png",level_maps)
  cv2.imwrite("D:/myworklist/deeplearning/paper/refinenetdeblur/paper/debluredge/AttentioNN-master/images/result1231/"+count+".png",level_maps)
  count=count+1
  
#plot_heatmaps(range(164, 169))
#cv2.imwrite("D:/myworklist/deeplearning/paper/refinenetdeblur/paper/debluredge/AttentioNN-master/"+str(2)+".png",img)
#plot_heatmaps(range(76, 81))
#cv2.imwrite("D:/myworklist/deeplearning/paper/refinenetdeblur/paper/debluredge/AttentioNN-master/"+str(3)+".png",img)
#plot_heatmaps(range(3, 8))
#cv2.imwrite("D:/myworklist/deeplearning/paper/refinenetdeblur/paper/debluredge/AttentioNN-master/"+str(4)+".png",img)

filePath = 'D:/myworklist/deeplearning/paper/refinenetdeblur/paper/debluredge/AttentioNN-master/images/heatmap/'
count=0
for i in os.listdir(filePath):
  #img = cv2.imread("cat.jpg")[:,:,::-1]
  img = cv2.imread("D:/myworklist/deeplearning/paper/refinenetdeblur/paper/debluredge/AttentioNN-master/images/heatmap/"+i)[:,:,::-1]
  img = cv2.resize(img, (224, 224))
  plot_heatmaps(range(164, 169))
  plot_heatmaps(range(76, 81))
  plot_heatmaps(range(3, 8))
  #ax = plt.imshow(img)
  cv2.imwrite("D:/myworklist/deeplearning/paper/refinenetdeblur/paper/debluredge/AttentioNN-master/images/result1231/"+i+".png",img)


























