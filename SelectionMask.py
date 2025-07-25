import torch
import torch.nn.functional as F

class SelectionMask:
    """
    A class to generate and manipulate a random selection mask tensor.

    Attributes:
        tensor (torch.Tensor): The generated selection mask tensor.
    """

    def __init__(self, shape):
        """
        Initializes the SelectionMask object by creating a random mask tensor.

        Args:
            shape (tuple): A 4D tuple representing the desired shape of the mask 
                           (batch_size, channels, height, width).
        """
        self.tensor = self.get_mask_from_shape(shape)
        
    def get_mask_from_shape(self, shape) -> torch.Tensor:
        """
        Generates a random tensor of the given shape.

        Args:
            shape (tuple): A 4D shape tuple (batch_size, channels, height, width).

        Returns:
            torch.Tensor: A tensor with random values in the range [0, 1].

        Raises:
            ValueError: If the shape is not 4-dimensional.
        """
        if len(shape) != 4:
            raise ValueError("Size must be equal to 4 (batch, channels, height, width)")
        
        return torch.rand(size=shape)
    
    def binarized(self) -> torch.Tensor:
        """
        Converts the mask into a binary tensor using a threshold of 0.5.

        Returns:
            torch.Tensor: A binary tensor (0 or 1), where values >= 0.5 become 1.
        """
        return (torch.abs(self.tensor) >= 0.5).int()
    
    def interpolated(self, size, mode="nearest") -> torch.Tensor:
        """
        Resizes the mask tensor to a new spatial size using interpolation.

        Args:
            size (tuple): The desired output size (height, width).
            mode (str): Interpolation mode to use (default is "nearest").

        Returns:
            torch.Tensor: The resized tensor.
        """
        return F.interpolate(self.tensor, size=size, mode=mode)
    
    def apply(self, image) -> torch.Tensor:
        """
        Applies the binary mask to an input image by element-wise multiplication.

        Args:
            image (torch.Tensor): Input image tensor with shape matching the mask
                                (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The masked image.

        Raises:
            ValueError: If the image shape doesn't match the mask tensor shape.
        """
        if image.shape == self.tensor.shape:
            return image * self.binarized()
        else:
            raise ValueError("image and mask must have equal shapes")
