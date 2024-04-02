## Introduction
This is the implementation of the eRisk 2024 CLEF Lab Task 2 baseline system, which can be seen cited below:

Wang, Y. T., Huang, H. H., Chen, H. H., & Chen, H. (2018, September). A Neural
Network Approach to Early Risk Detection of Depression and Anorexia on Social
Media Text. In CLEF (Working Notes) (pp. 1-8).

## Requirement

- python 3
- pytorch > 0.1
- torchtext > 0.1
- numpy

## Usage

run -> python3 main.py -h
This opens the help menu to find different commands.

To train, run -> python3 main.py
This will run batches of 100 to find loss and accuracy

To make test set, run -> python3 main.py -test -save-dir="SAVE_DIR"
This will create the test set needed to run sentence classification

For prediciton, run -> python3 main.py -predict="PREDICTION TEXT , SPACE BEFORE PUNCTUATION ." -snapshot="NAMEOFSNAPSHOT"
This will predict the label of the text (pos, neg, neutral)

Example output of prediction:

tensor([[    2, 15442,    11,     2, 14359,  5938,     2,   647]])

[Text]  The rainbows in the mist dazzle the eyes .
[Label] positive

