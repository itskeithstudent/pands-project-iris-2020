import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree #specify the model
from sklearn.model_selection import train_test_split #to split data in train and validate
from matplotlib import pyplot as plt
import file_names

#iris_df is the dataframe storing the full iris dataset
iris_df = pd.read_csv('iris-flower-dataset\IRIS.csv')
y = iris_df['species']
species_names = y.unique()

#Features, these are my columns of interest that I think will factor into determining flower species
feature_names = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']

#X is a dataframe of just the columns from feature_names
X = iris_df[feature_names]
print("x and y set")

#define iris_model object from DecisionTreeClassifier, using random_state=0 as it makes the result consistent
iris_model = DecisionTreeClassifier(random_state=0, max_leaf_nodes=4)
print("Model defined")

# Supplying a numeric value to the random_state argument guarantees we get the same split every time this script is ran.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
print("train and val")

# Fit model
iris_model.fit(train_X, train_y)
print("model fitted")

# get predicted category on validation data
val_predictions = iris_model.predict(val_X)
print("model predictions")

print(val_predictions)
#get the score
print(iris_model.score(val_X, val_y))
model_score = iris_model.score(val_X, val_y)

with open('Decision Tree Analysis Output\summary.txt', 'w') as summaryFile:
    summaryFile.write("This is a summary text file of a decision tree classification model ran against the Iris Data Set.\n")
    summaryFile.write("As a first step I will pick my column of interest which will be called y, this is the Species column as I want to determine the species using the other columns.\n")
    summaryFile.write("After having identified my y, I need to pick my X, X are my features (can also be called variables) for this model.")
    summaryFile.write(f"From the exporatory data analysis I performed earlier, I know we have 3 different species: {species_names}")
    summaryFile.write(f"This then brings me back to X, my features of interest which to start with are:\n{feature_names}\n")
    summaryFile.write("After defining these I define my iris_model object which uses sklearn.tree.DecisionTreeRegressor\n")
    summaryFile.write("With the DecisionTreeRegressor I supply the arguments random_state=0 - this ensures that we get a consistent value back, not using random_state results in getting different results everytime\n")
    summaryFile.write("The other argument that get's supplied is max_leaf_nodes=7, which sets the max number of splits in the tree we have (I played around with this in my jupyter notebook file:'Jupyter Analysis\iris_prelim_analysis.ipynb')\n")
    summaryFile.write("I then split the iris dataset into training and validating for the X and y, this ensures that we aren't overfitting the data.\n")
    summaryFile.write("The model get's fitted using the training data for X and y.\n")
    summaryFile.write("Then using the fitted model, we predict the flower species by supplying our validated X data, this will let us know whether the model we've built so far is accurate or not.\n")
    summaryFile.write(f"Our predicted values are:\n{list(val_predictions)}\n")
    summaryFile.write(f"Our actual values that we're trying to predict are:\n{list(val_y)}\n")
    summaryFile.write("Finally we get the score for or model, all this is really doing is checking with our validation data to see whether we can guess correctly we do this by supplying arguments for the validated X data and the values we predict for these, y.\n")
    summaryFile.write(f"Our returned model score is: {model_score}\n")
    summaryFile.write("The model score looks pretty good\nWhat this tells us is that across the validation data we are off for one flower, which is a pretty good guess for our purposes as it was wrong only in one case.")
print("open df")

#create a plot for the decision tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=125)
#using plot_tree sklearn library, generate the tree plot
plot_tree(iris_model,
               feature_names = feature_names,
               class_names=species_names,
               filled = True)
#save plot to png in Decision Tree Analysis Output Folder
plt.savefig(file_names.optimalNodesDTCPngName)