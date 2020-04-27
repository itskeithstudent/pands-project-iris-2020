import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree #specify the model
from sklearn.model_selection import train_test_split #to split data in train and validate
from matplotlib import pyplot as plt
import file_names

# generateModel function returns a DecisionTreeClassifier object
# takes arguments for:
#   maximum number of leaf nodes used by model
#   X/feature data, what data determines our response
#   y/response data, the data for our responses
def generateModel(leaf_nodes, feature_data, response_data):

    #define iris_model object from DecisionTreeClassifier, using random_state=0 as it makes the result consistent
    #max_leaf_nodes defines the number of end nodes I want, use too many can result in overfitting data
    model = DecisionTreeClassifier(random_state=0, max_leaf_nodes=leaf_nodes)

    #Fit the model, really straightforward, just supply your X and y as arguments
    model.fit(feature_data, response_data)
    #return the model
    return model

# writeToSummary function to write summary data of DecisionTreeClassifier model to a text file
def writeToSummary(write_to_file, responses, features, predicted_values, validation_values, score):
    #with will close if encounter's any errors
    with open(write_to_file, 'w') as summaryFile:
        summaryFile.write("This is a summary text file of a decision tree classification model ran against the Iris Data Set.\n")
        summaryFile.write("As a first step I will pick my column of interest which will be called y, this is the Species column as I want to determine the species using the other columns.\n")
        summaryFile.write("After having identified my y, I need to pick my X, X are my features (can also be called variables) for this model.")
        summaryFile.write(f"From the exporatory data analysis I performed earlier, I know we have 3 different species: {responses}\n")
        summaryFile.write(f"This then brings me back to X, my features of interest which to start with are:\n{features}\n\n")
        summaryFile.write("After defining these I define my iris_model object which uses sklearn.tree.DecisionTreeRegressor\n")
        summaryFile.write("With the DecisionTreeRegressor I supply the arguments random_state=0 - this ensures that we get a consistent value back, not using random_state results in getting different results everytime\n")
        summaryFile.write("The other argument that get's supplied is max_leaf_nodes=7, which sets the max number of splits in the tree we have (I played around with this in my jupyter notebook file:'Jupyter Analysis\iris_prelim_analysis.ipynb')\n\n")
        summaryFile.write("I then split the iris dataset into training and validating for the X and y, this ensures that we aren't overfitting the data.\n")
        summaryFile.write("The model get's fitted using the training data for X and y.\n")
        summaryFile.write("Then using the fitted model, we predict the flower species by supplying our validated X data, this will let us know whether the model we've built so far is accurate or not.\n")
        summaryFile.write(f"Our predicted values are:\n{list(predicted_values)}\n\n")
        summaryFile.write(f"Our actual values that we're trying to predict are:\n{list(validation_values)}\n\n")
        summaryFile.write("Finally we get the score for or model, all this is really doing is checking with our validation data to see whether we can guess correctly we do this by supplying arguments for the validated X data and the values we predict for these, y.\n")
        summaryFile.write(f"Our returned model score is: {score}\n")
        summaryFile.write(f"Or another way of putting it, our model can correctly predict the species for {round((score*100),2)}% of our feature validation data.\n\n")
        summaryFile.write("The model score looks pretty good\nWhat this tells us is that across the validation data we are off for one flower, which is a pretty good guess for our purposes as it was wrong only in one case.")
        summaryFile.write("This is a spoiler for Decision Tree image, the model only uses two of the four provided features to get this score: Petal_Length and Petal_Width")

# generateModelPlot function to generate a plot of a DecisionTreeClassifier Model
def generateModelPlot(model, features, responses, save_to_file):
    #create a plot for the decision tree
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=125)
    #using plot_tree sklearn library, generate the tree plot
    plot_tree(model,
                feature_names = features,
                class_names= responses,
                filled = True)
    #save plot to png in Decision Tree Analysis Output Folder
    plt.savefig(save_to_file)

#iris_df is the dataframe storing the full iris dataset
iris_df = pd.read_csv('iris-flower-dataset\IRIS.csv')

#y is the iris species and is my response i.e. what I want to predict
y = iris_df['species']
species_names = y.unique()

#Features, these are my columns of interest that I think will factor into determining flower species
feature_names = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']

#X is a dataframe of just the columns from feature_names
X = iris_df[feature_names]

# Supplying a numeric value to the random_state argument guarantees we get the same split every time this script is ran.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# leaf_nodes is set to 4, from testing in Jupyter Notebook
# 4 was the optimal number of leaf nodes as it gave the most accurate result at the lowest number of leaf nodes
leaf_nodes = 4

# iris_model gets returned a DecisionTreeModel object
iris_model = generateModel(leaf_nodes, train_X, train_y)

# get what our model predicts what y should be when we pass in val_X the validation feature data
val_predictions = iris_model.predict(val_X)

# get the score for our model e.g. how accurate it is
model_score = iris_model.score(val_X, val_y)

# folder and file name to save summary to
file_folder_name = 'Decision Tree Analysis Output\summary.txt'

# write text summary
writeToSummary(file_folder_name, species_names, feature_names, val_predictions, val_y, model_score)

# generate a plot for the model so far
generateModelPlot(model=iris_model, features=feature_names, responses=species_names, save_to_file=file_names.optimalNodesDTCPngName)
