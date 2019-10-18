# ipython train.py data_directory
import sys
import argparse
import torch
import numpy as np
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch import nn
from torch import optim
import torch.nn.functional as F
import collections
from collections import OrderedDict



print ('Begin Image Classifier Training')
print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))

parser = argparse.ArgumentParser(description='Image classifier training')
parser.add_argument("dat", help="data directory")
parser.add_argument("-s", "--save_checkpoint", help="directory for saving checkpoint")
parser.add_argument("-a", "--arch",choices=["vgg16","vgg19"], default="vgg16", help="pre trained model")


# validation - use validation data to check the classifier
# 
def validation(model, valid_loader, criterion):
    model.to (device)
    
    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

def train_model(model, train_loader, validate_loader, optimizer):
    # train the classifier
    # first I tried this in my laptop environment, but was taking a really long time
    # so I aborted and now doing it online in GPU mode
	model.to (device)
	epochs = 8
	print_every = 50
	steps = 0
	start = time.time()

	for e in range (epochs): 
		running_loss = 0
		for ii, (inputs, labels) in enumerate (train_loader):
			steps += 1
			inputs, labels = inputs.to(device), labels.to(device)
			optimizer.zero_grad () #optimizer
		
			# Forward and backward passes
			outputs = model.forward (inputs) 
			loss = criterion (outputs, labels) # loss
			loss.backward () 
			optimizer.step () #optimize 
			running_loss += loss.item () # loss.item () returns scalar value of Loss function
		
			if steps % print_every == 0:
				model.eval () # eval mode, off dropout
				
				# Turn off gradients for validation, to speed up
				with torch.no_grad():
					valid_loss, accuracy = validation(model, valid_loader, criterion)
				
				print("Epoch: {}/{}.. ".format(e+1, epochs),
					  "Running loss : {:.4f}".format(running_loss),
					  "Training Loss: {:.4f}.. ".format(running_loss/print_every),
					  "Validation Loss: {:.4f}.. ".format(valid_loss/len(valid_loader)),
					  "Validation Accuracy: {:.4f}%".format(accuracy/len(valid_loader)*100))
				print(f"Device = {device}; Time per batch: {(time.time() - start)/print_every:.3f} seconds")
				
				running_loss = 0
				start = time.time()
