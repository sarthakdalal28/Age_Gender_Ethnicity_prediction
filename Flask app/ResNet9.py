import os
import torch
import torchvision
#import tarfile
import torch.nn as nn
import numpy as np
#import torch.nn.functional as F
#from torchvision.datasets.utils import download_url
#from torch.utils.data import TensorDataset
#from torch.utils.data import DataLoader
#import torchvision.transforms as tt
#from torch.utils.data import random_split
#from torchvision.utils import make_grid
#from sklearn.preprocessing import LabelEncoder
#import seaborn as sns
#import matplotlib
#import matplotlib.pyplot as plt
#Let's defining function for doing optimization after each convolution layer
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), #Convolution layer
              nn.BatchNorm2d(out_channels), # Batch normalization, to help model generalize and learn better
              nn.ReLU(inplace=True)] # Activation function introduction of non linearity
    if pool: 
        layers.append(nn.MaxPool2d(2)) # Maxpooling if required
    return nn.Sequential(*layers)

class Base(nn.Module):
    # training
    def training_step(self, batch,age=False):
                          
        if not age: 
            images, labels = batch 
            out = self(images) # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
        else: 
            images, targets = batch 
            targets = targets.to(torch.float32)
            out = self(images)
            loss =torch.sqrt(F.mse_loss(out[:,0],targets))
        return loss
    # For validation set
    def validation_step(self, batch,age=False):
                           
        if not age:
            images, labels = batch 
            out = self( images)  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            acc = accuracy(out, labels)           # Calculate accuracy
            return {'val_loss': loss.detach(), 'val_acc': acc}
        else: 
            images, targets = batch 
            out = self(images)
            loss =torch.sqrt(F.mse_loss(out[:,0],targets)) # Calculate loss
            
              
        return {'val_loss': loss.detach()}
    # Stacking batch losses & accuracies and getting average   
    def validation_epoch_end(self, outputs,age=False):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        if not age: 
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result,age=False):
        if not age:
            print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
        else:
            print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}".format(
                epoch, result['lrs'][-1], result['train_loss'], result['val_loss']))


class ResNet9(Base):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 200)
        self.conv2 = conv_block(200, 100, pool=True)
        self.res1 = nn.Sequential(conv_block(100,100), conv_block(100,100))
        
        self.conv3 = conv_block(100, 190, pool=True)
        self.conv4 = conv_block(190, 360, pool=True)
        self.res2 = nn.Sequential(conv_block(360, 360), conv_block(360, 360))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(360, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out