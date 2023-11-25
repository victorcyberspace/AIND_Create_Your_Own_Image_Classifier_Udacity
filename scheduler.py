"""
The purpose of the function learning_rate_scheduler() is to adjust the learning rate of an optimizer during training. 
It does this by adding a small value to the learning rate after a certain number of epochs.

The function takes two arguments: optimizer: The optimizer to use, num_epoch: The current epoch.
The function first gets the current learning rate from the optimizer's parameter groups. 

It then checks if the current epoch is greater than 3. If it is, the function adds 0.005 to the learning rate. Finally, the function sets the learning rate in the optimizer's parameter groups to the new learning rate.
"""


def learning_rate_scheduler(optimizer, num_epoch):
    """
    A learning rate scheduler that adds 0.005 to the learning rate after 3 epochs.

    Args:
        optimizer: The optimizer to use.
        epoch: The current epoch.
    """
    lr = optimizer.param_groups[0]["lr"]
    if num_epoch > 3:
        lr = lr + 0.005
    optimizer.param_groups[0]["lr"] = lr

