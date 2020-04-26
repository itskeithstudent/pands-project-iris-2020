# pands-project-iris-2020
An analysis of the famous Iris Dataset for Programming and Scripting 52167 taught at GMIT.

This repository describes the dataset using a few .py and .ipynb files which output their results in the form of images and summary text files (in the case of the .py python files) and in the form of step by step execution using jupyter notebooks.

This project consists of three parts:
  A set of analysis files in jupyter notebook that step through the analysis cell by cell and include descriptions of the analysis as   well as describing my thought process based on the outputs.
  An exploratory data analysis python file this takes work done in the jupyter notebook and formalizes by putting it into functions, running this file produces a set of visualisations, some show more basic plot's using matplotlib and other more refined one's that were made with seaborn.
  A decision tree model python file, this uses the sklearn package to generate a model, test and validate it.
  
### Contained within this repository
* Readme.md - What you are reading right now.
* Jupyter Analysis\iris_prelim_analysis.ipynb - Jupyter notebook exploratory data analysis.
* Jupyter Analysis\iris_decision_tree.ipynb - Jupyter notebook for decision tree classifier.
* file_names.py - Python file with variables storing string names for where to save files.
* hist_subplot.py - Python file containing function for generating a histogram subplot.
* iris_exploratory_data_analysis.py - Python file where once called perfroms an exploratory data anlysis on the iris dataset.
* iris_modelling.py - Python file where once called generates a Decision Tree Classifier model on the iris dataset.

### Technologies Used
  [Python](https://www.python.org/)
 
  [Jupyter Notebook](https://jupyter.org/)


## How to run the analysis

### Jupyter Notebooks
Jupyter notebooks for the analysis can be found here: "Jupyter Analysis/iris_prelim_analysis.ipynb" 
This is a .ipynb filetype so should be opened within jupyter notebook to properly follow cell by cell through the analysis.
This Jupyter notebook goes through some initial exploratory data analysis, by opening the file and summarising the data within.
This file is intended to show my thought process for this project, in that I am documenting it as I am thinking what to do with the data and why. I also try to explain what I'm understanding from the data after certain steps so that there is a logical sequence and flow.

Within the jupyter notebook file I'd reccomend to run all cells on opening (but of course there is no issue executing line by line), by reading downwards you will go step by step through the analysis and the thought process behind what is being executed (why I'm plotting, why using a particular package, etc.)


### Exploratory Data Analysis
For executing the remaining analysis you will need to run "iris_exploratory_data_analysis.py", (required libraries for this will be included in a requirements.txt file) this will then generate a text file and series of images saved from the generated plots in the "Preliminary Analysis" folder.
This python file imports a simple but large function for generating a subplot of histograms from "hist_subplot.py".
String names for the images generated by the analysis are imported from "filenames.py"


## Resources and References Used
For retrieving the Iris Dataset I used a csv which you can find as part of this repository - original csv was found here: https://www.kaggle.com/arshid/iris-flower-dataset/data 



````
Please note: you can get the iris dataset using the sklearn package for python,
but for the purposes of this project I want to include file reading and potentially 
manipulating and outputting the data into one or more other .csv file's.
(...of course this may change as the project progresses, but for the moment this sounds like a good idea to me)
````
