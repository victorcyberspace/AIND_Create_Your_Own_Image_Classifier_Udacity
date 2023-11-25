"""
The load_datasets function loads, transforms, and stores image datasets for training, validation, and testing using the PyTorch ImageFolder class, centralizing the data loading process in the script and making it easier to maintain and modify.
"""

# import the datasets module from the torchvision library for loading pre-defined datasets
import torchvision.datasets as datasets


# import the Image module from the PIL library for image processing
import PIL.Image

# import the ImageFolder class from the datasets module for loading image folders
from torchvision.datasets import ImageFolder

# import the os module for interacting with the operating system
import os 

# import the transforms module from the torchvision library
import torchvision.transforms as transforms

# import the datasets module from the torchvision library
import torchvision.datasets as datasets  
 
data_dir = 'flowers'  # define the data directory
train_dir = os.path.join(data_dir, 'train')  # define the training directory
valid_dir = os.path.join(data_dir, 'valid')  # define the validation directory
test_dir = os.path.join(data_dir, 'test')    # define the test directory
    
train_transforms = transforms.Compose([               # define the train transforms
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

def load_datasets():
        """loads the train, validation, and test datasets with ImageFolder"""

        image_datasets = []

        # Iterate over the dataset types
        for dataset_type in ["train", "validation", "test"]:

            # Create an ImageFolder object for the current dataset
            if dataset_type == "train":
                dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
            elif dataset_type == "validation":
                dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
            elif dataset_type == "test":
                dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

            # Add the datasets to the image_datasets list
            image_datasets.append(dataset)

        return image_datasets