import numpy as np
from keras.datasets import mnist
import warnings

warnings.filterwarnings('ignore')

#mnist data
#https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

NUM_TRAIN_IMGS = train_images.shape[0]  #60000
NUM_TEST_IMGS = test_images.shape[0]  #10,000
IMG_DIM = train_images.shape[1]  #28
INPUT_SIZE = IMG_DIM**2  #784
NUM_NODES = 50 #number of nodes per layer
OUTPUT_SIZE = 10

#hyperparameters
epoch = 500
learn_rate = 1.5

#-------matrix dimensions---------

#train_images (NUM_NODES, 60000)
#train_labels (60000, )
#input_images (784, 60000)
#vector_labels (10, 60000)

#iw (NUM_NODES, 784)
#ib (NUM_NODES, 1)
#ow (10, NUM_NODES)
#ob (10, 1)



#initial weights and biases
def init_params():
  iw = np.random.uniform(-0.5, 0.5, (NUM_NODES, INPUT_SIZE))
  ib = np.zeros((NUM_NODES, 1))
  ow = np.random.uniform(-0.5, 0.5, (OUTPUT_SIZE, NUM_NODES))
  ob = np.zeros((OUTPUT_SIZE, 1))

  return iw, ib, ow, ob


#to compress input to range: (0, 1)
def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
  return sigmoid(x) * (1 - sigmoid(x))


#creating a prediction based on given weights and biases
def fwd_prop(iw, ib, ow, ob, train_images):
  #input -> hidden
  iz = iw.dot(train_images) + ib
  h_img = sigmoid(iz)

  #hidden -> output
  oz = ow.dot(h_img) + ob
  o_img = sigmoid(oz)

  return iz, h_img, o_img


#function to convert a number to a vector (size 10) of zeros where  the number index is one
# eg. 4 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] 
def convert_labels(train_labels):
  #create a (60000, 10) matrix to store all output vectors in training set
  #dimensions are swapped to facilitate use of for loop
  vector_labels = np.zeros((train_labels.size, OUTPUT_SIZE))

  #assigning element @ index label value to 1
  for value, vector in enumerate(vector_labels):
    vector[train_labels[value]] = 1

  return vector_labels.T #return transpose to have output vector in column


def bckwd_prop(iz, h_img, o_img, ow, train_images, train_labels):
  vector_labels = convert_labels(train_labels)

  #cost/error calculations
  doz = o_img - vector_labels
  dow = 1 / NUM_TRAIN_IMGS * doz.dot(h_img.T)
  dob = 1 / NUM_TRAIN_IMGS * np.sum(doz)
  diz = ow.T.dot(doz) * deriv_sigmoid(iz)
  diw = 1 / NUM_TRAIN_IMGS * diz.dot(train_images.T)
  dib = 1 / NUM_TRAIN_IMGS * np.sum(diz)

  return diw, dib, dow, dob


def update_params(iw, ib, ow, ob, diw, dib, dow, dob, learn_rate):
  #updating weights and biases to minimize cost/error
  iw -= learn_rate * diw
  ib -= learn_rate * dib
  ow -= learn_rate * dow
  ob -= learn_rate * dob

  return iw, ib, ow, ob


#converting training images numpy array from (60000, 28, 28) -> (784, 60000)
def convert_image(train_images, num_images):
  #create a new matrix of correct size to store input values
  vector_images = np.zeros((num_images, INPUT_SIZE))

  #iterate through the loop and reshape it
  for i, image in zip(range(num_images), train_images):
    vector_images[i] = image.reshape(INPUT_SIZE, )

  return vector_images.T #return transpose to have input in column


def get_prediction(prediction_vector):
  return np.argmax(prediction_vector) #returns the index of the elment model predicted with highest possibility

def get_accuracy(predictions, labels):
  total_correct = 0

  #iterating through the the predictions and comparing the output to the label
  for i, prediction_vector in zip(range(labels.size), predictions.T):
    if get_prediction(prediction_vector) == labels[i]:
      total_correct += 1
      

  return round(total_correct / labels.size * 100, 2) #returning accuracy as a rounded percentage


def train(train_images, train_labels, num_images, learn_rate, epoch):
  iw, ib, ow, ob = init_params()
  max_accuracy = 0 #maximum accuracy during training
  max_epoch = 0 #epoch number with the maximum training accuracy

  train_images = convert_image(train_images, num_images)

  for e in range(epoch):
    #iw and oz never used
    iz, h_img, o_img = fwd_prop(iw, ib, ow, ob, train_images)
    diw, dib, dow, dob = bckwd_prop(iz, h_img, o_img, ow, train_images,
                                    train_labels)
    iw, ib, ow, ob = update_params(iw, ib, ow, ob, diw, dib, dow, dob,
                                   learn_rate)

    accuracy = get_accuracy(o_img, train_labels)

    #calculating iteration with max accuracy throughout training
    if accuracy > max_accuracy:
      max_accuracy = accuracy
      max_epoch = e + 1

    print("E " + str(e + 1) + ", Accuracy: ", accuracy, "\tMax @ E " + str(max_epoch) + ": ", max_accuracy)

  #these are the best weights and biases after training
  return iw, ib, ow, ob


def test(iw, ib, ow, ob, test_images, test_labels):
  test_images = convert_image(test_images, NUM_TEST_IMGS)

  #run forward prop with testing data to create prediction
  _, _, output = fwd_prop(iw, ib, ow, ob, test_images)

  print("\n\ntest sample accuracy: ", get_accuracy(output, test_labels))

final_iw, final_ib, final_ow, final_ob = train(train_images, train_labels, NUM_TRAIN_IMGS, learn_rate, epoch)


test(final_iw, final_ib, final_ow, final_ob, test_images, test_labels)



