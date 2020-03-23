# pands-project-iris-2020
An analysis of the famous Iris Dataset for Programming and Scripting 52167 taught at GMIT
This repository is to act as a way of describing the Iris dataset by the analysis performed in the .py files contained and the conclusions made.

## How to run the analysis
Currently all of the analysis can be found in "Jupyter Analysis/iris_prelim_analysis.ipynb" in this repository, this is a jupyter notebook file so should be opened within jupyter notebook to properly follow the analysis.
This Jupyter notebook goes through some initial exploratory data analysis, by opening the file and summarising the data within.
This file is intended to show my thought processfor this project, in that I am documenting it as I am thinking what to do with the data and why. I also try to explain what I'm understanding from the data after certain steps so that there is a logical sequence and flow.

Within the jupyter notebook file I'd reccomend to run all cells on opening (but of course there is no issue executing line by line), by reading downwards you will go step by step through the analysis and the thought process behind what is being executed (why I'm plotting, why using a particular package, etc.)

For executing the remaining analysis you will need to run "prelim_analysis.py", (required libraries for this will be included in a requirements.txt file) this will then generate a text file and series of images saved from the generated plots.

## Structure for analysis
The analysis will be most likely carried out over a few steps.
We will first need to retrieve the dataset in question, for this project I got the dataset from Kaggle (which you can find here: https://www.kaggle.com/arshid/iris-flower-dataset/data). You can also find the IRIS.csv file which contains all the data in this repository in the iris-flower-dataset directory.

The analysis performed in the .py files will write observations on the dataset to a summary text file and save the generated plots to a analysis output folder. These files will execute based on key observations from the jupyter notebook analysis.

````
Please note: you can get the iris dataset using the sklearn package for python,
but for the purposes of this project I want to include file reading and potentially 
manipulating and outputting the data into one or more other .csv file's.
(...of course this may change as the project progresses, but for the moment this sounds like a good idea to me)
````
