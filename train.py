""" 
    The train.py script is a comprehensive tool for training deep learning models for image classification using PyTorch. 
    It handles various aspects of the training process, including command line argument parsing, data loading and transformation, model setup, training         loop, learning rate scheduling, and checkpoint saving.

    The script begins by importing essential libraries and modules, including argparse for handling command-line arguments, PyTorch for deep learning,         torchvision for computer vision tasks, and other related modules.

    Next, the script parses command-line arguments to allow users to specify crucial parameters such as data directory, model architecture, learning rate,     number of hidden units, number of epochs, and GPU usage.

    The script then loads the training, validation, and testing datasets using the torchvision library and applies the specified transformations. It then       creates data loaders for each dataset.

    Next, the script loads a pre-trained model (either VGG16 or DenseNet121) and replaces the classifier with a custom one according to the chosen             architecture. The model's parameters are frozen to prevent updates during training.

    The core of the script is the training loop. 
    It iterates through the specified number of epochs, training the model on the training dataset. 
    During training, it computes loss, performs backpropagation, and updates the model's parameters using an optimizer. 
    It also periodically evaluates the model's performance on the validation set, displaying metrics such as training and validation loss and accuracy.

    The script uses a learning rate scheduler to adjust the learning rate during training based on the epoch number.

    Finally, the script saves the trained model checkpoint, including the model's architecture, optimizer state, and class-to-index mapping, to a specified     file.

    Overall, the train.py script provides a comprehensive and flexible framework for training deep learning models for image classification. 
    It automates many aspects of the training process, making it accessible and convenient for users with various levels of expertise in deep learning and     PyTorch.

"""

# import the argparse library for commandline argument parsing
import argparse

# import the PyTorch library for deep learning 
import torch

# import the neural networks module from the PyTorch library 
import torch.nn as nn

# import the functional module from the PyTorch library for common activation and loss functions
import torch.nn.functional as F

# import the optimizers module from the PyTorch library for optimizer algorithms
import torch.optim as optim

# import the torchvision library for computervision tasks
import torchvision

# import the datasets module from the torchvision library for loading pre-defined datasets
import torchvision.datasets as datasets

# import the transforms module from the torchvision library for image transformations
import torchvision.transforms as transforms

# import the models module from the torchvision library for pre-trained models
import torchvision.models as models

# import the ImageFolder class from the datasets module for loading image folders
from torchvision.datasets import ImageFolder

# import the Variable class from the autograd module for automatic differentiation
from torch.autograd import Variable

# import the Image module from the PIL library for image processing
import PIL.Image

# import the OrderedDict class from the collections library for storing and manipulating data structures
from collections import OrderedDict

# import the time library for measuring time intervals
import time

# import the NumPy library for scientific computing
import numpy as np

# import the MatPlotlib library for data visualization
import matplotlib.pyplot as plt

# import the keep_awake & active_session functions from the workspace_utils module
# prevent the workspace from timing out during long-running tasks
from workspace_utils import active_session

# import the save_checkpoint_file and the rebuild_model_from_checkpoint functions 
from checkpoint_info import save_checkpoint_file, rebuild_model_from_checkpoint

# import the learning_rate_scheduler function from the scheduler module
from scheduler import learning_rate_scheduler

# import the load_datasets function from the datasets module
from datasets import load_datasets

# import the os module for interacting with the operating system
import os 



# define a function to parse command line arguments 
def parse_args():
    
    # create an ArgumentParser object ('parser') with description of the training script
    parser = argparse.ArgumentParser(description="Training")
    
    # add an argument to 'parser' for specifying the data directory
    parser.add_argument('--data_dir', action='store')
    
    # Add a command-line argument to specify the model architecture. 
    # The default architecture is 'vgg16', but users can also choose 'densenet121'.
    parser.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg16', 'densenet121'])
    
    # Add the `checkpoint` argument.
    parser.add_argument('--load_dir', dest="load_dir", action="store", default="trained_model_checkpoint.pth",
                        help="The path to the trained model checkpoint file.")
    
    # add an argument for specifying the directory to save the model checkpoint to
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="trained_model_checkpoint.pth")
    
    # add an argument for setting the learning rate 
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.01')
    
    # add an argument for setting the number of hidden units 
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    
    # add an argument for setting the number of epochs
    parser.add_argument('--num_epochs', dest='num_epochs', default='4')
    
    # add an argument for specifying gpu usage 
    parser.add_argument('--gpu', action='store', default='cuda')
    
    # add an argument for specifying cpu usage
    parser.add_argument('--cpu', action='store', default='cpu')
    
    # parse the command line arguments and return the results 
    return parser.parse_args()

def train(model, criterion, optimizer, dataloaders, num_epochs, device):
    
    logging_interval = 5   # set how many times an update on progress will be shown 
    steps = 0              # set a counter to keep track of how far we are in training
    
    for num_epoch in range(num_epochs):

        # Initialize the running loss
        running_loss = 0

        # invoke the active session function
        with active_session():

            # Iterate over the training data loader
            for inputs, labels in dataloaders[0]:

                # Increment the step counter
                steps += 1

                model.to(device)
                
                # Move the inputs and labels to the specified device (GPU or CPU)
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Reset the gradients
                optimizer.zero_grad()

                # Forward pass through the model
                outputs = model(inputs)

                # Compute the loss
                loss = criterion(outputs, labels)

                # Backpropagate the loss
                loss.backward()

                # Update the parameters
                optimizer.step()

                # Accumulate the running loss
                running_loss += loss.item()

                # If the step number is divisible by the logging interval,
                # evaluate the model on the validation set and print the results
                if steps % logging_interval == 0:

                    # Set the model to evaluation mode
                    model.eval()

                    # Disable gradient calculation
                    with torch.no_grad():

                        # Initialize validation loss and accuracy
                        val_loss = 0
                        val_accuracy = 0

                        # Iterate over the validation data loader (dataloaders[1])
                        for val_inputs, val_labels in dataloaders[1]:
                            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                            # Calculate validation loss
                            val_outputs = model(val_inputs)
                            val_loss += criterion(val_outputs, val_labels).item()

                            # Calculate validation accuracy
                            val_ps = torch.exp(val_outputs)
                            val_equality = (val_labels.data == val_ps.max(dim=1)[1])
                            val_accuracy += val_equality.type(torch.FloatTensor).mean()

                            # Calculate average validation loss and accuracy
                            validation_loss = (val_loss / len(dataloaders[1]))*100
                            validation_accuracy = (val_accuracy / len(dataloaders[1]))*100

                            # Calculate the train loss
                            train_loss = (running_loss/logging_interval)*100

                            # Print the training and validation metrics
                            print(f"Epoch: {num_epoch+1}/{num_epochs}   ",
                                  f"Train loss: {train_loss:.1f}%   ",
                                  f"Validation loss: {validation_loss:.1f}%   ",
                                  f"Validation accuracy: {validation_accuracy:.1f}%")

                            # check num_epochs and update the learning rate if the set condition is True
                            learning_rate_scheduler(optimizer, num_epoch)
                                                       
                            # save the checkpoint dictionary to the file named 'checkpoint_filename'
                            save_checkpoint_file(checkpoint_filename, model, optimizer, args, classifier, class_to_idx)

                            # Reset the running loss
                            running_loss = 0

                            # Set the model back to training mode
                            model.train()

if __name__ == '__main__':
    
    data_dir = 'flowers'                         # define the data directory
    train_dir = os.path.join(data_dir, 'train')  # define the training directory
    valid_dir = os.path.join(data_dir, 'valid')  # define the validation directory
    test_dir = os.path.join(data_dir, 'test')    # define the test directory
    
    train_transforms = transforms.Compose([           # define the train transforms
                   transforms.RandomRotation(75),     # rotate image by 30 degrees
                   transforms.RandomResizedCrop(224), # resize image 224 by 224 pixels
                   transforms.RandomHorizontalFlip(), # flip the image horizontally
                   transforms.ToTensor(),             # convert image to a PyTorch tensor
                   transforms.Normalize([0.485,       # normalize using required measurements 
                   0.456, 0.406], [0.229, 0.224,
                   0.225])])
    
    def custom_transforms():
        return transforms.Compose([transforms.Resize(200),      # resize the image to 256x256
                                   transforms.CenterCrop(224),  # crop the image to 224x224 at the center
                                   transforms.ToTensor(),       # convert the image to a PyTorch tensor
                                   transforms.Normalize([0.485, # normalize the image
                                   0.456,0.406],[0.229, 0.224, 
                                   0.225])])

    # invoke the 'custom_transforms()' function
    # store its value in the 'validation_transforms' and the 'test_transforms' variable  
    validation_transforms = custom_transforms()
    test_transforms = custom_transforms()

    

    # load the image datasets using the defined function load_datasets()
    # store the result in the variable image_datasets
    image_datasets = load_datasets()

    # create an empty list to store the data loaders
    dataloaders = []

    # iterate over the image datasets
    for i in range(len(image_datasets)):

        # create a data loader for the current dataset
        dataloader = torch.utils.data.DataLoader(image_datasets[i], batch_size=32, shuffle=True)

        # add the data loader to the list of data loaders
        dataloaders.append(dataloader)
        

    # Get a mapping from class names to class indices.
    class_to_idx = dataloaders[0].dataset.class_to_idx
    
    # Parse the command line arguments.
    args = parse_args()
    
    # After creating dataloaders, create a mapping from class names/labels to indices
    class_to_idx = dataloaders[0].dataset.class_to_idx
    
    # Load the pre-trained model.
    model = getattr(models, args.arch)(pretrained=True)
    
    # Freeze the model parameters.    
    for param in model.parameters():
        
        # Disable gradient computation for the model parameters 
        # to prevent them from being updated during training.
        param.requires_grad = False

    
    # Define the classifier
    if args.arch == "vgg16":
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 1536)),                           # First fully connected layer with 25088 inputs and 1536 outputs
            ('relu1', nn.ReLU()),                                      # ReLU activation function after the first layer
            ('dropout1', nn.Dropout(0.4)),                             # Dropout 40% of the inputs
            ('l2_reg', nn.LayerNorm(1536, elementwise_affine=False)),  # Apply L2 regularization to the first layer output
            ('fc2', nn.Linear(1536, 256)),                             # Second fully connected layer with 1536 inputs and 256 outputs
            ('relu2', nn.ReLU()),                                      # ReLU activation function after the second layer
            ('fc3', nn.Linear(256, 102)),                              # Third fully connected layer with 256 inputs and 102 outputs
            ('output', nn.LogSoftmax(dim=1))                           # Convert the outputs to probabilities
        ]))
    elif args.arch == "densenet121":
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, 450)),                             # First fully connected layer with 1024 inputs and 450 outputs
            ('dropout', nn.Dropout(p=0.55)),                           # Dropout 55% of the inputs
            ('relu', nn.ReLU()),                                       # ReLU activation function after the first layer
            ('fc2', nn.Linear(450, 102)),                              # Second fully connected layer with 450 inputs and 102 outputs
            ('output', nn.LogSoftmax(dim=1))                           # Convert the outputs to probabilities
        ]))

    # Update the classifier in the model
    model.classifier = classifier

    # Define the loss criterion and optimizer
    criterion = nn.NLLLoss()
    
    # Filter out the parameters that do not require gradients
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))

    # Get the number of epochs from the command line arguments
    num_epochs = int(args.num_epochs)

    # Get the class index from the training dataset
    class_index = image_datasets[0].class_to_idx

    # check if cuda (GPU) is available
    if torch.cuda.is_available():
        # use the gpu specified in the command line arguments
        device = args.gpu 
    else:
        # use the cpu specified in the command line arguments
        device = args.cpu  

    # Set the checkpoint filename variable to save the directory.
    checkpoint_filename = args.save_dir    

    # Train the model
    train(model, criterion, optimizer, dataloaders, num_epochs, device)

    # Update the class index to the model
    model.class_to_idx = class_index

    # Get the checkpoint filename from the command line arguments
    checkpoint_filename = args.save_dir

    # Save the checkpoint file
    save_checkpoint_file(checkpoint_filename, model, optimizer, args, classifier, class_to_idx)


