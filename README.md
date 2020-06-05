# FoDS-Final_project
Final project. Topic: "An Empirical Exploration of Recurrent Network Architectures"
This project was made by Skoltech students as a final project for the course Theoretical Foundations of Data Science. 
Authors: Gostilovich Sergei, Valeriya Strizhkova

# Description:
This project consists of Pipeline (TestRNNArch and Trial), Tasks (ArithmeticTask, XMLTask, PTBTask, AbstractTasks), RNN_cells (EncoderCell, DecoderCell, and so on) and PBT_module (for Penn-Tree Bank task).
The target is evaluating performance of different RNN architectures by estimating its performance on three different tasks:  ArithmeticTask, XMLTask, PTBTask.
All classes based on PyTorch (https://pytorch.org/)

You need to install PyTorch and torch NLP (pip install pytorch-nlp (https://pypi.org/project/pytorch-nlp/))
 
In main.py there is the example of testing three RNN architectures (GRU, LSTM, RNN)
Collect all files in one directory and run main.py. All should work.

# Task description 
To evaluate the effectiveness of different RNN architectures, each architecture was tested on the following three tasks: Arithmetic, XML modeling, and Penn Tree-Bank (PTB). The practical implementation of each task is a python class, which has 3 main methods: initialization-which initializes tasks based on the input parameters “ task_param ”, do_train – which trains the model for a given number of epochs, do_estimate - which evaluates the trained model and outputs accuracy.

# PipeLine
Direct testing of RNN architectures was performed using the pipeline module, which includes two python classes: Trial and TestRNNArch. The General structure for estimating of an input RNN architecture for an input set of hyperparameters is shown in figure 1. The RNN architecture, which is a python class, and a set of hyperparameters (“test_dict ”) are fed to the input of an instance of the TestRNNArch class.

The estimating procedure is as follows. 
TestRNNArch is initialized with the following parameters. They are “test_param” which required by the TestRNNArch class, “trial_param”, that are initialization parameters required by the python class Trial and “task_param”, that are initialization parameters required by python classes of tasks (ArithmeticTask, XMLTask, PTBTask). After initialization, the TestRNNArch generates a list of hyperparameter sets that will be passed to the Trail class to train and estimate the RNN architecture on the selected hyperparameter set and the selected task.
Then the TestRNNArch class sequentially goes through the parameters of the list of hyperparameter sets for each task and calculates the results that contains the best results of completing each of the task among the entire list of hyperparameter sets. In this way, TestRNNArch returns an estimates of the best performance of each of the tasks.
The class  Trial accepts input parameters, interacts with task classes, passes the result of task completion to the TestRNNArch class, and logs the learning and evaluation process.


# Estimate
For estimating performance of a RNN architecture the follows simple criterion was used:

Estimate = N_matching / N_total

The criterion simple counts the number of matchings characters (words) considering the order and normalizes it by dividing by the total number of characters (words)

# More information about the idea of experiments: http://proceedings.mlr.press/v37/jozefowicz15.pdf
