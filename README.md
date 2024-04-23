## Introduction
This is the hypothesized improved implementation of the eRisk 2024 CLEF Lab Task 2 baseline system, which can be seen cited below:

Wang, Y. T., Huang, H. H., Chen, H. H., & Chen, H. (2018, September). A Neural
Network Approach to Early Risk Detection of Depression and Anorexia on Social
Media Text. In CLEF (Working Notes) (pp. 1-8).

## Requirement

- python 3
- pytorch > 0.1
- torchtext > 0.1
- numpy
- other libraries listed at top of both .py files

## Usage

Make sure all files in github are available, and all path strings in code are changed to suit your own code.

run -> python3 main.py

Model will take n amount of epochs (decided by user) and train until epochs are reached

The model will then test on each of the testing labels (same labels as used in the last part of the project, so that consistency is used)

Model prints T (true label), P (predicted label), and the line it tested on

In this model, 2 is a positive sentiment, meaning the user likely does not have anorexia. 0 is a negative sentiment, meaning the user likely has anorexia. 1 is a neutral sentiment, meaning there is no clear conclusion that can be reached.