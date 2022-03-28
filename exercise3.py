import ROOT
from ROOT import TMVA, TFile, TTree, TCut, gROOT ,gSystem
from os.path import isfile
import numpy as np
from subprocess import call
import pickle
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
  
#Set_up_TMVA____________________________________________________________________________________________________________________  
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
   
outputFile = TFile.Open("Higgs_ClassificationOutput.root", "RECREATE")
   
factory = TMVA.Factory("TMVA_Higgs_Classification", outputFile,
                         "!V:ROC:!Silent:Color:AnalysisType=Classification")
                         
# Setup Dataset(s)______________________________________________________________________________________________________________
   
inputFileName = ROOT.TString("Higgs_data.root")
inputFileLink = ROOT.TString("http://root.cern.ch/files/" + inputFileName)
    
if not isfile('Higgs_data.root'):
  call(['curl', '-L', '-O', 'http://root.cern.ch/files/Higgs_data.root'])
  
inputFile = TFile.Open('Higgs_data.root')
      
signalTree = inputFile.Get('sig_tree')
backgroundTree = inputFile.Get('bkg_tree')

signalTree.Print();    

dataloader = TMVA.DataLoader('dataset')


#for branch in signalTree.GetListOfBranches():
#    dataloader.AddVariable(branch.GetName())

dataloader.AddVariable("m_jj");
dataloader.AddVariable("m_jjj");
dataloader.AddVariable("m_lv");
dataloader.AddVariable("m_jlv");
dataloader.AddVariable("m_bb");
dataloader.AddVariable("m_wbb");
dataloader.AddVariable("m_wwbb");

signalWeight     = 1.0;
backgroundWeight = 1.0;

dataloader.AddSignalTree    ( signalTree,     signalWeight     );
dataloader.AddBackgroundTree( backgroundTree, backgroundWeight );

dataloader.PrepareTrainingAndTestTree( TCut(''), TCut(''),
                                       "nTrain_Signal=7000:nTrain_Background=7000:SplitMode=Random:NormMode=NumEvents:!V" ); 
                                       
# Define model_________________________________________________________________________________________________________________
model = nn.Sequential()
model.add_module('linear_1', nn.Linear(in_features=7, out_features=64))
model.add_module('relu', nn.Tanh())
model.add_module('linear_2', nn.Linear(in_features=64, out_features=2))
                                       
#Define_loss_function_and_optimizer___________________________________________________________________________________________                                       
loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD

#Training________________________________________________________________________________________________________________
def train(model, train_loader, val_loader, num_epochs,
          batch_size, optimizer, criterion, save_best, scheduler):
    trainer = optimizer(model.parameters(), lr=0.01)
    schedule, schedulerSteps = scheduler
    best_val = None

    for epoch in range(num_epochs):
        # Training Loop
        # Set to train mode
        model.train()
        running_train_loss = 0.0
        running_val_loss = 0.0
        for i, (X, y) in enumerate(train_loader):
            trainer.zero_grad()
            output = model(X)
            train_loss = criterion(output, y)
            train_loss.backward()
            trainer.step()

            # print train statistics
            running_train_loss += train_loss.item()
            if i % 64 == 63:    # print every 64 mini-batches
                print(f"[Epoch {epoch+1}, {i+1}] train loss:"
                      f"{running_train_loss / 64 :.3f}")
                running_train_loss = 0.0

        if schedule:
            schedule(optimizer, epoch, schedulerSteps)

        # Validation Loop
        # Set to eval mode
        model.eval()
        with torch.no_grad():
            for i, (X, y) in enumerate(val_loader):
                output = model(X)
                val_loss = criterion(output, y)
                running_val_loss += val_loss.item()

            curr_val = running_val_loss / len(val_loader)
            if save_best:
               if best_val==None:
                   best_val = curr_val
               best_val = save_best(model, curr_val, best_val)

            # print val statistics per epoch
            print(f"[Epoch {epoch+1}] val loss: {curr_val :.3f}")
            running_val_loss = 0.0

    print(f"Finished Training on {epoch+1} Epochs!")

    return model 
                                      
#Define_Predict_function_______________________________________________________________________________________________                                       
def predict(model, test_X, batch_size=32):
    # Set to eval mode
    model.eval()
   
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_X))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            X = data[0]
            outputs = model(X)
            predictions.append(outputs)
        preds = torch.cat(predictions)
   
    return preds.numpy()                                      
                                       
load_model_custom_objects = {"optimizer": optimizer, "criterion": loss,
                             "train_func": train, "predict_func": predict}

#Saving_the_model___________________________________________________________________________________________________________________
           
print(model)
m = torch.jit.script(model)
torch.jit.save(m, "model.pt")

#Booking_methods_____________________________________________________________________________________________________________________________   
factory.BookMethod(dataloader, TMVA.Types.kLikelihood, "Likelihood",
                           "H:!V:TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10:NSmooth=1:NAvEvtPerBin=50" );
                           
factory.BookMethod(dataloader, TMVA.Types.kFisher, "Fisher", "H:!V:Fisher:VarTransform=None:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10" );

factory.BookMethod(dataloader,TMVA.Types.kBDT, "BDT",
                      "!V:NTrees=200:MinNodeSize=2.5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );

factory.BookMethod(dataloader, TMVA.Types.kPyTorch, 'PyTorch',
                   'H:!V:VarTransform=D,G:FilenameModel=model.pt:'
                   'NumEpochs=30:BatchSize=32')
         
#Run_TMVA_____________________________________________________________________________________________________________________________          

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

roc = factory.GetROCCurve(dataloader)
roc.Draw()
                                    



    





















