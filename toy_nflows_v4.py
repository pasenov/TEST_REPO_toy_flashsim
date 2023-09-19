# -*- coding: utf-8 -*-

#!pip install nflows
import matplotlib.pyplot as plt
import corner
import torch
from torch import nn
from torch import optim
from scipy import stats
from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal,StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform,MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.nn.nets import ResidualNet

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Model wrapper

class model_from_flow(torch.nn.Module):
    def __init__(self, flow):
        super().__init__()  
        self.flow = flow

    def forward(self, x):
        self.flow.eval()
        output_dist = self.flow.sample(num_samples=1, context=x)
        return output_dist

def profile(datasets,labels, ix, iy, nbins, filename, logScale=True):
    # Create the figure and the axis
    fig, ax = plt.subplots()
    ax.set_ylim(0.0,0.5)
    if logScale :
        ax.set_xscale('log')

    # Get the minimum and maximum values of column iy in the first dataset
    ymin = datasets[0][:, iy].min()
    ymax = datasets[0][:, iy].max()

    # Create the bin edges
    if logScale :
        bin_edges = np.logspace(np.log10(ymin), np.log10(ymax), nbins + 1)
    else:
        bin_edges = np.linspace(ymin, ymax, nbins + 1)

    # Loop through the datasets
    for i, dataset in enumerate(datasets):
        # Calculate the mean and standard deviation of column ix in each bin
        means, edges, _ = stats.binned_statistic(dataset[:, iy], dataset[:, ix], statistic='mean', bins=bin_edges)
        stds, _, _ = stats.binned_statistic(dataset[:, iy], dataset[:, ix], statistic='std', bins=bin_edges)

        # Calculate the bin centers
        bin_centers = (edges[:-1] + edges[1:]) / 2

        # Plot the mean and the error band
        ax.errorbar(bin_centers, stds,  fmt='o', label=labels[i])

    # Add a legend
    ax.legend()

    # Save the figure to the specified filename
    plt.savefig(filename)

    # Close the figure
    plt.close(fig)

#N=500000
batchsize=2000

import numpy as np
dataset=np.load("gen.npy")
#dataset=np.load("gen200k.npy")
print("dataset: ",dataset)
print("dataset shape: ", dataset.shape)
attrExtractors = [
#GEN LEVEL
lambda j : j.pt(),  #0
lambda j : j.eta(), #1
lambda j : j.phi(), #2
lambda j : j.e(), #3
lambda j : j.flavour, #4
#RECO LEVEL
lambda j : j.btag, #5
lambda j : j.recoPt, #6
lambda j : j.recoPhi, #7
lambda j : j.recoEta, #8
lambda j : j.muonsInJet[0].pT() if len(j.muonsInJet) >0 else -1, #9
lambda j : j.recoNConstituents, #10
]
X=dataset[:,(7,6,5,8,10)]
print("X: ",X)
Y=dataset[:,(0,1,2,3,4,9)]
print("Y: ",Y)
X[:,1] = X[:,1] / Y[:,0]
Y[:,4] = np.abs(Y[:,4])
#N = Y.shape[0] #Set N to the size of Y along axis 0
N = 6000 # With 6000 the program is running
print(Y.shape)


def buildFlow():
  num_layers = 15 #was 10 and 5
  from nflows.distributions.normal import ConditionalDiagonalNormal,StandardNormal

  base_dist = StandardNormal(shape=[5])


  transforms = []

  for _ in range(num_layers):
      transforms.append(RandomPermutation(features=5))
      transforms.append(MaskedAffineAutoregressiveTransform(features=5, 
                                                        use_residual_blocks=False,
                                                        num_blocks=8,
                                                        hidden_features=64, #was 4, 20
                                                        context_features=6))

  transform = CompositeTransform(transforms)

  flow = Flow(transform, base_dist).to("cpu")
  optimizer = optim.Adam(flow.parameters() )# ,lr=0.001)
  return flow,optimizer

fa,oa=buildFlow()

import sys
start=0
if len(sys.argv) > 1 :
          fa=torch.load( "flow%s_%s"%(0,sys.argv[1]))
          start=int(sys.argv[1])+1
          oa = optim.Adam(fa.parameters() )# ,lr=0.001)
          print("loaded")

print(sum(p.numel() for p in fa.parameters() if p.requires_grad))

bins=np.linspace(0.5,10.5,11)/10.
binsMany=np.linspace(0.5,20.5,200)/10.

#num_iter = 5000
num_iter = 6000
indices=Y[:N//2,4].argsort()
indices=np.append(indices,np.arange(N//2,N,1, dtype=int))
print(indices)
print(X[indices,2])
print(Y[indices,4])
xa = torch.tensor(X[:N], dtype=torch.float32).to("cpu")
y = torch.tensor(Y[:N], dtype=torch.float32).to("cpu")
print("y: ", y)
np.savetxt("y.csv", y, delimiter=",")
ytest = torch.tensor(Y[N:], dtype=torch.float32).to("cpu")
scheduler = optim.lr_scheduler.ReduceLROnPlateau(oa, 'min',verbose=True, factor=0.5, patience=50)
for i in range(start,start+num_iter):
  for optimizer,flow,x in [(oa,fa,xa)]: #,(ob,fb,xb),(oc,fc,xc)]:
    totloss=0
    totcount=0
    if True :
      for b in range(N//batchsize ) :
        if (b+1)*batchsize < N :
            optimizer.zero_grad()
            loss= -flow.log_prob(inputs=x[b*batchsize:(b+1)*batchsize], context=y[b*batchsize:(b+1)*batchsize]).mean()
            loss.backward()
            optimizer.step()
            totloss+=loss.item()
            totcount+=1
    else:
            optimizer.zero_grad()
            loss= -flow.log_prob(inputs=x, context=y).mean()
            loss.backward()
            optimizer.step()
            totloss+=loss.item()
            totcount+=1
    scheduler.step(totloss)
    if (i + 1) % 10 == 0:
        print("train loss:",i,totloss/totcount)
  
  if  (i + 1) % 100 == 0:
        [flow.eval() for flow in [fa]] #,fb,fc]]
        fig, ax = plt.subplots(4, 3,figsize=(20, 15))
        xline = torch.linspace(0, 20,100)/10
        yline = torch.linspace(-5, 5,100)

        with torch.no_grad():
          for j,flow in enumerate([fa,]) : #fb,fc]):
              torch.save(flow, "flow%s_%s"%(j,i))
              flow=fa

        print(y.shape)
        ii=0
        context0 = y[ii*500:(ii+1)*500,:]
        print("context0: ", context0)
        np.savetxt("context0.csv", context0, delimiter=",")
        print("Context shape:",y[ii*500:(ii+1)*500,:].shape)
        samples = flow.sample(1,y[ii*500:(ii+1)*500,:]).detach().cpu().numpy()
        print("First Samples shape:",samples.shape)
        for ii in range(1,N//500) :
            context = y[ii*500:(ii+1)*500,:]
            np.savetxt("context%s.csv"%ii, context, delimiter=",")
            samples = np.concatenate((samples,flow.sample(1,context).detach().cpu().numpy()),axis=0)

        print("Samples shape:",samples.shape)
        plt.title('iteration {}'.format(i + 1))
        print(Y.shape)
        print(samples[:,0,:].shape)
        ranges=[
                (0,4), #0
                (0,6), #1 pt
                (-1,1), #2
                (-5.0, 5.0), #3
                (0,200), #4
                (0,500), #5 ptgen
                (-5,5), #6 eta 
                (0,6), # 7 phi
                (0,1000), #8 energy
                (-10,10), #9  flav
                (0,250), #10 const

                ]
        print("Corner1")
        YN = Y[:N,:]
        print("Y[:N,:] = ", YN)
        np.savetxt("YN.csv", YN, delimiter=",")
        samples0 = samples[:,0,:]
        print("samples[:,0,:] = ", samples0)
        np.savetxt("samples0.csv", samples0, delimiter=",")
        flash=np.concatenate((samples[:,0,:],Y[:N,:]), axis=1)
        print("flash = ", flash)
        np.savetxt("flash_Python.csv", flash, delimiter=",")
        figure = corner.corner(flash,range=ranges,color='blue',bins=20)
        print(xa.shape)
        print("Corner2")
        XN2N = X[N:2*N]
        print("X[N:2*N] = ", XN2N)
        np.savetxt("XN2N.csv", XN2N, delimiter=",")
        YN2N = Y[N:2*N]
        print("Y[N:2*N]) = ", YN2N)
        np.savetxt("YN2N.csv", YN2N, delimiter=",")
        real=np.concatenate((X[N:2*N],Y[N:2*N]),axis=1)
        print("real = ", real)
        np.savetxt("real.csv", real, delimiter=",")
        figure2 = corner.corner(real,fig=figure,range=ranges,color='red',bins=20)
        figure.savefig("corner_%s.png"%i)
        plt.close()

        # Separate events based on category
        bjet_indices_real = np.where(np.abs(real[:, 9]) == 5)
        light_indices_real = np.where(real[:, 9] == 0)

        bjet_indices_flash = np.where(np.abs(flash[:, 9]) == 5)
        light_indices_flash = np.where(flash[:, 9] == 0)

        # Get discriminator values
        discriminator_real_bjet = real[bjet_indices_real, 2].flatten()
        discriminator_real_light = real[light_indices_real, 2].flatten()

        discriminator_flash_bjet = flash[bjet_indices_flash, 2].flatten()
        discriminator_flash_light = flash[light_indices_flash, 2].flatten()

        # Calculate ROC curves
        fpr_real, tpr_real, _ = roc_curve(np.concatenate((np.ones(len(discriminator_real_bjet)), np.zeros(len(discriminator_real_light)))), np.concatenate((discriminator_real_bjet, discriminator_real_light)))
        fpr_flash, tpr_flash, _ = roc_curve(np.concatenate((np.ones(len(discriminator_flash_bjet)), np.zeros(len(discriminator_flash_light)))), np.concatenate((discriminator_flash_bjet, discriminator_flash_light)))

        # Calculate AUC
        auc_real = auc(fpr_real, tpr_real)
        auc_flash = auc(fpr_flash, tpr_flash)

        # Plot ROC curves
        plt.figure()
        plt.plot(tpr_real, fpr_real, label=f'Real (AUC = {auc_real:.2f})')
        plt.plot(tpr_flash, fpr_flash, label=f'Flash (AUC = {auc_flash:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0001, 1.05])
        plt.yscale('log')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curves for Real and Flash events')
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig("roc_%s.png"%i)
        plt.close()
        print("ROC done")

        profile([flash,real],["flash","real",],1,5,30,"responseVsPt_%s.png"%i,logScale=True)
        profile([flash[bjet_indices_flash],real[bjet_indices_real],flash[light_indices_flash],real[light_indices_real]],["flashB","realB","flashLight","realLight"],1,5,30,"responseVsPtFlavour_%s.png"%i,logScale=True)
        profile([flash,real],["flash","real",],1,6,30,"responseVsEta_%s.png"%i,logScale=False)
        [flow.train() for flow in [fa,] ] #fb,fc]]

# Save the model

wrapped_model = model_from_flow(flow)
with torch.no_grad():
    fake_input = torch.rand(128, 6)
    wrapped_model.eval()
    traced_model = torch.jit.trace(wrapped_model, fake_input)
    traced_model.eval()
    traced_model.save("flow_model.pt")
    torch.save(flow.state_dict(), "flow_state_dict_model.pt")
