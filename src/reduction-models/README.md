# Reduction Models

In this repository, we plan to house all the source code for testing, developing, and producing our **Reduction Models**.

First, what are Reduction Models?

## What is the plan?

We have decided to pivot from the SBRC paper we were writing. Since then, we have decided to focus fully on reducing models using a mask.

To that end, we have developed several ideas and plans. To understand them better, we need to define two core concepts.

### 1. Reduction Mask

The **Reduction Mask** is what we call the method of using a trained static mask to actively reduce the dimensionality of the data.

For example, a $256 \times 256$ image would be reduced to a string of 320 pixels containing the most important data. We will use only this **320-pixel string** for processing.

By using the **String Data** as the sole source of information, we aim to validate if we can effectively reduce dimensionality while maintaining accuracy. We will experiment with the following approaches:

**1.1 Simple MLP Approach**

- Using a Multi-Layer Perceptron (MLP) as the classifier for the string of information.
    
- We need to validate if this is a viable option for high-demand datasets.
    

**1.2 Traditional Machine Learning**

- Using traditional ML algorithms to classify the string data.
    
- Testing methods like KNN, Random Forest, SVM, and others.
    

**1.3 1D Convolution**

- Applying 1D convolution techniques to analyze and extract features from the data string.
    

This plethora of methods serves to experiment with and validate the central hypothesis:

> **Can we use a static mask to reduce data dimensionality and, consequently, reduce the complexity of the classifier?**

---

### 2. Classification Mask

The second method involves using the static mask to reduce the **complexity** of the data rather than its dimensionality. The reasoning behind this experiment is:

**Can we use the static mask to reduce data complexity so that a simpler CNN can classify what previously required a much larger model?**

To test this, we will train a mask alongside a State-of-the-Art (SOTA) model recommended for the specific dataset, and then perform the following:

**2.1 Test with the same model and analyze the statistical difference**

- Compare the results of the same model with and without the mask applied.
    

**2.2 Test with a smaller model of the same architecture**

- For example, moving from a ResNet-50 to a ResNet-20.
    

**2.3 Test with a simpler architecture altogether**

- Training the original data with a ResNet and testing the masked data with a LeNet.
    

We will measure the statistical differences across all tests to determine if reducing data complexity allows for better performance in simpler models.

---

## Methodology

To implement these methods, we have created the `/src/reduction-models` directory for all code and utilities.

The idea is to create a utility class to aggregate our custom functions, specifically:

- **A custom wrapper for the Dataloader:** Returns the dataset with the mask applied.
    
- **A custom wrapper for the Dataloader:** Returns the dataset as a byte string of the selected pixels.
    

> Both wrappers will receive the `.pth` file as an argument, and the transformations will be applied dynamically as the Dataloader runs.

Finally, we will create several Jupyter notebooks to conduct a thorough statistical analysis of the results. This will allow us to determine the best path forward for the project.

## Conclusion

By distinguishing between **Reduction Masks** (dimensionality focused) and **Classification Masks** (complexity focused), this project seeks to find the optimal balance between data efficiency and model performance. Our goal is to prove that intelligent data filtering via static masks can significantly lower the computational overhead required for high-accuracy classification.