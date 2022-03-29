import numpy as np
import keras
import ROOT
import talos
import tensorflow as tf
from ROOT import TMVA, TFile, TTree, TCut
from subprocess import call
from os.path import isfile
 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
from keras.models import Sequential
from tensorflow.keras.utils import Sequence

variables = ["var1","var2"]

# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

output = TFile.Open('TMVA.root', 'RECREATE')
factory = TMVA.Factory('TMVARegression', output,
        '!V:!Silent:Color:DrawProgressBar:Transformations=D,G:AnalysisType=Regression')
        
# Load data
if not isfile('tmva_reg_example.root'):
    call(['curl', '-L', '-O', 'http://root.cern.ch/files/tmva_reg_example.root'])
    
data = TFile.Open('tmva_reg_example.root')
tree = data.Get('TreeR')

#loading data in numpy format and also readable my any machine learning framework tool
tree_data = ROOT.RDataFrame("TreeR",data).AsNumpy()
x_tree = np.array([tree_data[var] for var in variables]).T
num_tree = x_tree.shape[0]

np.save('file.npy',x_tree)

print(x_tree)
  
batch_size = 32 
         
def generator_function(array, batch_size):
  inputs = []
  targets = []
  batchcount = 0
  while True:
       for x in array:
         inputs.append(x[0])
         targets.append(x[1])
         batchcount += 1
         if batchcount > batch_size:
                  X = np.array(inputs, dtype='float32')
                  y = np.array(targets, dtype='float32')
                  yield (X,y)
                  inputs = []
                  targets = []
                  batchcount = 0

#Define model for generator
model = Sequential()
model.add(Dense(64, activation='tanh', input_dim=1))
model.add(Dense(1, activation='linear'))
 
# Set loss and optimizer
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01))

#fitting the generator function
model.fit_generator(generator_function(x_tree,batch_size),steps_per_epoch=num_tree / batch_size, epochs=20)

#saving the model
model.save('model.h5')
model.summary()

   




























