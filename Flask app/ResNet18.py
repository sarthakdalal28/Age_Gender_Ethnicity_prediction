import os
import torch
import torchvision
#import tarfile
import torch.nn as nn
import numpy as np
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


class ResNet18(Base):
    def __init__(self, in_channels):
        super().__init__()
        self.conv0=conv_block(in_channels, 64)
        self.res0=nn.Sequential(conv_block(64,64), conv_block(64,64))
        self.res1 = nn.Sequential(conv_block(64,64), conv_block(64,64))
        self.conv1 = conv_block(64, 128)
        self.conv2 = conv_block(128, 256, pool=True)
        self.res2 = nn.Sequential(conv_block(256,256), conv_block(256,256))
        
        self.conv3 = conv_block(256, 450)
        self.conv4 = conv_block(450, 512,pool=True)
        self.res3 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        #self.dense_out = nn.Linear(in_features=hidden_size, out_features=1)
        
        self.conv5 = conv_block(512, 72)
        self.conv6 = conv_block(72, 250,pool=True)
        self.res4 = nn.Sequential(conv_block(250,250), conv_block(250,250))
        
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(250, 1))
        
    def forward(self, xb):
        out = self.conv0(xb)
        out = self.res0(out) + out
        out = self.res1(out) + out
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res3(out) + out
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.res4(out) + out
        out = self.classifier(out)
        return out