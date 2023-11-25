"""
The predict.py script is a Python program that can be used to train and use a deep learning model for image classification. 
It utilizes the PyTorch library and various other Python modules and functions to accomplish its tasks.

The script begins by importing necessary libraries and modules, including PyTorch, NumPy, Matplotlib, and others. 
It then defines functions for parsing command-line arguments, loading category names, pre-processing images, and predicting the class of an image using a pre-trained model.

The main() function orchestrates the entire process. 
It parses command-line arguments, rebuilds a model from a checkpoint, loads category names, and performs image classification using the specified image. 
It prints out the chosen image path, predicted labels, class indices, and top probabilities.

Overall, the predict.py script serves as a versatile tool for image classification tasks. 
It allows users to train and deploy deep learning models for recognizing objects in images with ease.
"""


import math
import torch                                                # import PyTorch library
import torch.nn as nn                                       # import the 'nn' module from PyTorch
import torch.optim as optim                                 # import the 'optim' module from PyTorch
import torch.nn.functional as F                             # import the 'F' function from the 'nn.functional' module 
import torchvision                                          # import PyTorch Vision library
import torchvision.datasets as datasets                     # import the 'datasets' module from PyTorch Vision
import torchvision.transforms as transforms                 # import the 'transforms' module from PyTorch Vision
import torchvision.models as models                         # import the 'models' module from the PyTorch Vision
import numpy as np                                          # import the NumPy library
import matplotlib.pyplot as plt                             # import the Matplotlib library
import os                                                   # import the operating system library 
import json
from PIL import Image                                       # import the image class from the python imaging library
from collections import OrderedDict                         # import the OrderedDict class                    
from checkpoint_info import rebuild_model_from_checkpoint   # import the load_checkpoint function from the checkpoint_info.py file
from checkpoint_info import load_cat_names                  # import the load_cat_to_name function from the checkpoint_info.py file
import argparse

def parse_args():
    """Parses the command line arguments.

    Returns:
    A namespace object containing the parsed arguments.
    """
    # Create an argument parser.
    parser = argparse.ArgumentParser()

    # Add the `checkpoint` argument.
    parser.add_argument('--load_dir', dest='load_dir', action="store", default="trained_model_checkpoint.pth", help="The path to the trained model checkpoint file.")

    # Add the `top_k` argument.
    parser.add_argument('--top_k', dest='top_k', default='5', help="The number of top classes to return.")

    # Add the `image_path` argument.
    parser.add_argument('--image_path', dest='image_path', default='flowers/test/11/image_03147.jpg', help="The path to the image file to predict.")

    # Add the `category_to_name` argument.
    parser.add_argument('--category_to_name', dest='category_to_name', default='cat_to_name.json', help="The path to the JSON file containing the category to name mapping.")

    # Add the `gpu` argument.
    parser.add_argument('--gpu', action='store', default='cuda', help="Whether to use the GPU for prediction.")
    
    # Add the `cpu` argument
    parser.add_argument('--cpu', action='store', default='cpu', help="Whether to use the cpu for prediction.")

    # Parse the command line arguments.
    return parser.parse_args()


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

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # open the image using PIL
    img = Image.open(image_path)
    
    # define transformations for image preprocessing 
    preprocess =  transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # apply the defined transformations
    img_tensor = preprocess(img)
    
    return img_tensor

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Process the image
    image = process_image(image_path)

    # Add a batch dimension to the image
    image = image.unsqueeze(0)

    # Move the image to the specified device (GPU or CPU)
    image = image.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation
    with torch.no_grad():

        # Calculate the probability distribution over the classes
        ps = torch.exp(model(image))

        # Get the top k most likely classes
        top_ps, top_classes = ps.topk(topk, dim=1)
        
        # Parse the command line arguments.
        args = parse_args()

        # Load the category to name mapping.
        category_to_name = load_cat_names(args.category_to_name)

        # Convert the class indices to class names
        idx_to_flower = {v: category_to_name[k] for k, v in model.class_to_idx.items()}

        # Modify the following line to return class indices instead of flower names
        predicted_classes = [idx.item() for idx in top_classes[0]]

        # Convert the class indices to class names.
        predicted_flowers = [idx_to_flower[i] for i in top_classes.tolist()[0]]

        # Return the top k most likely classes and their probabilities
        return top_ps.tolist()[0], predicted_classes, predicted_flowers



# Define the main function
def main(): 
    
    # Parse command line arguments
    args = parse_args()
    
    # check if cuda (GPU) is available
    if torch.cuda.is_available():
        
        # use the gpu specified in the command line arguments
        device = args.gpu 
        
    else:
        
        # use the cpu specified in the command line arguments
        device = args.cpu  
    
    # Rebuild the model from a checkpoint
    model = rebuild_model_from_checkpoint(args.load_dir)
    
    # Load category names
    category_to_name = load_cat_names(args.category_to_name)
    
    # Get the path to the input image
    image = args.image_path
    
    # Perform image classification and get top predictions
    top_p, classes, labels = predict(image, model, device, int(args.top_k))
    
    # Print a blank line
    print(' ')
    # Print the chosen image file path
    print('chosen_file: ' + image)
    # Print a blank line
    print(' ')
    # Print the predicted labels
    print(labels)
    # Print a blank line
    print(' ')
    # Print the corresponding class indices
    print(classes)
    # Print a blank line
    print(' ')
    # Print the top probabilities
    print(top_p)
    # Print a blank line
    
    # Loop through the top predictions and print labels and probabilities
    i = 0 
    while i < len(labels):
        print(' ')
        print("{} -----> top probability: {}".format(labels[i], top_p[i]))
        i += 1

# Check if the script is executed as the main program
if __name__ == "__main__":
    # Call the main function
    main()

