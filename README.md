
# Machine Learning Playground

This repository demonstrates the practical application of machine learning in Julia using the Flux ecosystem. While I used machine learning potentials during my computational physics thesis and have a strong understanding of the underlying mathematics, this project focuses on implementing, training, and evaluating machine learning models directly in Julia, rather than using the FORTRAN implementation of RuNNer.

The aim is to demonstrate familiarity with the machine learning workflow while gaining hands-on experience with Julia's ML tooling.

# Objectives

* Demonstrate the standard workflow for training machine learning models.
* Become proficient with Flux.jl and the surrounding Julia ML ecosystem.
* Develop experience with data preparation, model training, and evaluation.
* Build intuition for optimisation, loss functions, and model performance.
* Apply these techniques to both standard datasets and scientific data.

# Projects

## Linear Regression

Implement a basic regression model to predict continuous values from small datasets.
This was performed on the Boston Housing data for linear regression, as well as the iris dataset for sigmoid regression/binary predictor. 


## Image Classification

Train a neural network to recognise handwritten digits using the MNIST dataset, a classic machine learning benchmark.

## Scientific Regression

Train a regression model to characterise a nonlinear neon dimer potential currently represented as a lookup table.

# Repository Structure

Each project builds upon the previous one, with commits documenting both implementation and the progression of ideas throughout the repository.

# Technologies

* Julia
* Flux.jl
* MLDatasets.jl
* Plots.jl
* StaticArrays.jl
