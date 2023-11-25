"""

The three defined functions in the checkpoint_info.py script serve the following purposes:
save_checkpoint_file(): saves a model checkpoint to a file. 
The checkpoint includes the model state dictionary, optimizer state dictionary, number of epochs, classifier, hidden units, learning rate, model architecture, and model itself.

rebuild_model_from_checkpoint(): loads a model checkpoint from a file and rebuilds the model. 
The function takes the path to the checkpoint file as input and returns the rebuilt model as output.

load_cat_names(): loads category names from a JSON file and returns a list of strings representing the category names.
These three functions work together to support the saving, loading, and rebuilding of model checkpoints.

The save_checkpoint_file() function can be used to save a model checkpoint at any point during training, which can be useful for resuming training from a previous checkpoint or for evaluating the model at different stages of training.

The rebuild_model_from_checkpoint() function can be used to load a saved model checkpoint and rebuild the model, which can be useful for deploying the model to production or for continuing training on a different machine. 

The load_cat_names() function can be used to load the category names from the dataset, which can be useful for visualizing the model's predictions or for interpreting the model's weights.

Overall, these three functions provide a comprehensive set of tools for managing model checkpoints in PyTorch.
"""

# import the argparse library for commandline argument parsing
import argparse
# import the PyTorch library for deep learning
import torch
# import the json module for reading and writing java script object notation data
import json
# import the copy module for deep copying objects 
import copy
# import the os module for interacting with the operating system
import os
# import the datasets module from the torchvision library 
import torchvision.datasets as datasets
# import the transforms module from the torchvision library 
import torchvision.transforms as transforms

def save_checkpoint_file (checkpoint_filename, model, optimizer, args, classifier, class_to_idx): 
      
    checkpoint = {                               # create a dictionry called 'checkpoint'
    'class_to_idx'    : class_to_idx,            # the 'class_to_idx' attribute of the model
    'state_dict'      : model.state_dict(),      # the state dictionary of the model 
    'optimizer'       : optimizer.state_dict(),  # the optimizer state of the model
    'num_epochs'      : 3,                       # the number of epochs used to train the model 
    'classifier'      : classifier,              # the model classifier
    'hidden_units'    : args.hidden_units,       # the number of hidden units in the model's classifier
    'learning_rate'   : args.learning_rate,      # the learning rate used to train the model
    'model'           : model,                   # the model itself
    'arch'            : args.arch                # the architecture of the model
    }

    torch.save(checkpoint, checkpoint_filename)      # save the checkpoint to the specified file 
  
def rebuild_model_from_checkpoint(checkpoint_filename):

    """
      load a checkpoint from a file_path_variable and rebuild the model when using GPU

      Args: 
          file_path_variable (str): the path to the checkpoint file

      Returns: the model

      """
    # load the chekpoint from the file_path_variable
    checkpoint = torch.load(checkpoint_filename)

    # get the model architecture from the checkpoint
    arch = checkpoint['arch']

    # load the model from the checkpoint
    model = checkpoint['model']

    # load the learning rate from the checkpoint
    learning_rate = checkpoint['learning_rate']

    # load the number of epochs from the checkpoint
    num_epochs = checkpoint['num_epochs']

    # load the optimizer from the checkpoint
    optimizer = checkpoint['optimizer']

    # replace the model classifier with the classifier from the checkpoint
    model.classifier = checkpoint["classifier"]

    # load the model state dict from the checkpoint
    model.load_state_dict(checkpoint["state_dict"])

    model.class_to_idx = checkpoint['class_to_idx']
    
    return model



def load_cat_names(filename):
    """Loads category names from a JSON file.

    Args:
    filename: A string representing the path to the JSON file.
    Returns:
    A list of strings representing the category names.
    """
    # open the JSON file in read mode.
    # loads the category names from the JSON file.
    cat_to_name = json.load(open(filename, 'r'))
    return cat_to_name




