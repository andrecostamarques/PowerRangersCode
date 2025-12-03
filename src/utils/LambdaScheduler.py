#!/usr/bin/env python3
"""
Parameter Scheduler Module.

This module provides a class for scheduling a numerical parameter (lambda) value, 
typically a learning rate or regularization weight, based on monitoring a loss value. 
It implements a strategy similar to 'ReduceLROnPlateau' but uses a relative
threshold for defining improvement.

Classes:
    LambdaScheduler: A scheduler that adapts a parameter based on monitored loss.
"""

class LambdaScheduler:
    """
    A class that controls the rate of change of the Lambda in the training loop using the 
    Lambda Scheduler. 

    """
    def __init__(self, init, factor, patience, treshold):
        """
        Initialize the Scheduler

        Args:
            init (float): The initial Lambda value.
            Factor (int): The rate of change when scheduled.
            Patience (int): The limit of non-change in epochs.
            Treshold (float): The limit minimum of change for activation.
        """
        self.lbd = init
        self.factor = factor
        self.patience = patience
        self.treshold = treshold
        self.limit = np.inf
        self.count = 0
        
    def adapt_lambda(self, new_loss): 
        """
        Apply the new Lambda based on the Scheduler.

        If the treshold is passed while in the patient count, it multiplies the current Lambda by the factor.
        Limit is based on the best loss, with treshold.

        Args:
            new_loss (float): The new lambda after the change.
        """
        if new_loss < self.limit:
            self.count = 0
            self.limit = new_loss * (1 - self.treshold)
        else:
            if self.count >= self.patience:
                self.lbd *= self.factor
                self.limit = new_loss * (1 - self.treshold)
                self.count = 0
            else:
                self.count += 1