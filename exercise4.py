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

dataloader = TMVA.DataLoader('dataset')
for branch in tree.GetListOfBranches():
    name = branch.GetName()
    if name != 'fvalue':
        dataloader.AddVariable(name)
dataloader.AddTarget('fvalue')
 
dataloader.AddRegressionTree(tree, 1.0)
dataloader.PrepareTrainingAndTestTree(TCut(''),
        'nTrain_Regression=4000:SplitMode=Random:NormMode=NumEvents:!V')
  
batch_size = 100  
  
#Create generator function accessing values from x_tree(loaded in the form of numpy array) 
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
                  yield (X, y)
                  inputs = []
                  targets = []
                  batchcount = 0
         
    
# Define model
model = Sequential()
model.add(Dense(64, activation='tanh', input_dim=2))
model.add(Dense(1, activation='linear'))
 
# Set loss and optimizer
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01))

#fitting the generator function
model.fit(generator_function(tf.expand_dims(x_tree,axis = 0),batch_size),steps_per_epoch=num_tree / batch_size, epochs=10)

#saving the model
model.save('model.h5')
model.summary()

#booking tmva methods
factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras',
        'H:!V:VarTransform=D,G:FilenameModel=model.h5:NumEpochs=20:BatchSize=32')
factory.BookMethod(dataloader, TMVA.Types.kBDT, 'BDTG',
        '!H:!V:VarTransform=D,G:NTrees=1000:BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=4')

#running tmva
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
   




























