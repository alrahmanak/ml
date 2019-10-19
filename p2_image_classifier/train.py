# ipython train.py data_directory
# python train.py flowers -s checkpoint.pth -ep 1
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
import json
import time
from datetime import datetime

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

print ('Begin Image Classifier Training')
print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))

parser = argparse.ArgumentParser(description='Image classifier training')
parser.add_argument("dat", help="data directory")
parser.add_argument("-s", "--save_checkpoint", default="checkpoint.pth", help="directory for saving checkpoint")
parser.add_argument("-a", "--arch",choices=["vgg16","vgg19"], default="vgg16", help="pre trained model")
parser.add_argument("-ep", "--epochs", type=int, default=8, help="Number of epochs")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.003, help="Learning rate")
parser.add_argument("-hu", "--hidden_units", nargs="+", type=int, default=[512, 256],
                    help="Number of nodes per hidden layer")
parser.add_argument("-g", "--gpu", action="store_true", help="Use GPU for training")
args = parser.parse_args()
print(args)

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

def train_model(model, train_loader, validate_loader, optimizer, device, epochs_cnt):
    # train the classifier
    # first I tried this in my laptop environment, but was taking a really long time
    # so I aborted and now doing it online in GPU mode
    model.to (device)
    epochs = epochs_cnt
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
                    valid_loss, accuracy = validation(model, validate_loader, criterion)
                
                train_loss = running_loss/print_every
                validate_loader_size = len(validate_loader)
                validation_loss = valid_loss/validate_loader_size
                valid_accuracy = (accuracy/validate_loader_size)*100
				
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Running loss : {:.4f}".format(running_loss),
                      "Training Loss: {:.4f}.. ".format(train_loss),
                      "Validation Loss: {:.4f}.. ".format(validation_loss),
                      "Validation Accuracy: {:.4f}%".format(valid_accuracy))
                print(f"Device = {device}; Time per batch: {(time.time() - start)/print_every:.3f} seconds")
                
                running_loss = 0
                
                start = time.time()
    return train_loss, valid_loss, valid_accuracy 
                
if __name__ == "__main__":

    print ('Data Dir:', args.dat)
    print ('Checkpoint Dir:', args.save_checkpoint)
    print ('Pretrained Architecture :', args.arch)
    print ('Number of epochs :', args.epochs)
	
    train_dir = args.dat + '/train'
    valid_dir = args.dat + '/valid'
    test_dir = args.dat + '/test'
    
    train_dat_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # for validation and test, no need to generalize
    valid_dat_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_dat_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_dat = datasets.ImageFolder(train_dir, transform=train_dat_transforms)
    valid_dat = datasets.ImageFolder(valid_dir, transform=valid_dat_transforms)
    test_dat = datasets.ImageFolder(test_dir, transform=test_dat_transforms)
    
	# TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dat, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dat, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_dat, batch_size=32)
	
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        print(cat_to_name)
		
	# build your pre trained model
    model = torchvision.models.vgg16(pretrained=True)
    model.name = "vgg16"
    print(model)
	
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear (25088, 4096)),
                          ('relu1', nn.ReLU ()),
                          ('dropout1', nn.Dropout (p = 0.3)),
                          ('fc2', nn.Linear (4096, 102)),
                          ('output', nn.LogSoftmax (dim =1))
                          ]))
    
    model.classifier = classifier
	
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device Type', device)
    model.to(device);
	
    criterion = nn.NLLLoss ()
    optimizer = optim.Adam (model.classifier.parameters (), lr = args.learning_rate)
	
	
    print('Start training the model')
    train_loss, valid_loss, valid_accuracy = train_model(model, train_loader, valid_loader, optimizer, device, args.epochs)
    print("Training Loss: {:.4f}.. ".format(train_loss),
                      "Validation Loss: {:.4f}.. ".format(valid_loss),
                      "Validation Accuracy: {:.4f}%".format(valid_accuracy))
					  
    model.cpu
    model.class_to_idx = train_dat.class_to_idx

    checkpoint = {'input_size': 25088,
              'output_size': 102,
              'arch': 'vgg16',
              'learning_rate': 0.01,
              'batch_size': 32,
              'classifier' : classifier,
              'epochs': args.epochs,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}
    print('Save model')
    torch.save(checkpoint, args.save_checkpoint)
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("End date and time =", dt_string)
    
