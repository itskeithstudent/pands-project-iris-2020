import pandas as pd
from sklearn.tree import DecisionTreeRegressor #specify the model
from sklearn.model_selection import train_test_split #to split data in train and validate
from sklearn.metrics import mean_absolute_error

#iris_df is the dataframe storing the full iris dataset
iris_df = pd.read_csv('iris-flower-dataset\IRIS.csv')
y = iris_df['species']

#Features, these are my columns of interest that I think will factor into determining flower species
feature_names = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']

#X is a dataframe of just the columns from feature_names
X = iris_df[feature_names]
print("x and y set")
#for getting my y we can't usepi categorical values like a named species, but we can convert this to a number equivalent
cat_conversion = {'Iris-setosa': 0, "Iris-versicolor":1, "Iris-virginica":2}

y.replace(cat_conversion, inplace=True)
print("y replaced")
#define iris_model object from DecisionTreeRegressor, using random_state=0 as it makes the result consistent
iris_model = DecisionTreeRegressor(random_state=0)

# Define model
iris_model = DecisionTreeRegressor(random_state=0, max_leaf_nodes=7)
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
#get the mean absolute error like before
print(mean_absolute_error(val_y, val_predictions))
mae = mean_absolute_error(val_y, val_predictions)

with open('Decision Tree Analysis Output\summary.txt', 'w') as summaryFile:
    summaryFile.write("This is a summary text file of a decision tree classification model ran against the Iris Data Set.\n")
    summaryFile.write("As a first step I will pick my column of interest which will be called y, this is the Species column as I want to determine the species using the other columns.\n")
    summaryFile.write("After having identified my y, I need to pick my X, X are my features (can also be called variables) for this model.")
    summaryFile.write("From the exporatory data analysis I performed earlier, I know we have 3 different species, which presents our first problem, the species are not numeric they are strings.")
    summaryFile.write(f"So these will be replaced as follows:\n{cat_conversion}\n")
    summaryFile.write(f"This then brings me back to X, my features of interest which to start with are:\n{feature_names}\n")
    summaryFile.write("After defining these I define my iris_model object which uses sklearn.tree.DecisionTreeRegressor\n")
    summaryFile.write("With the DecisionTreeRegressor I supply the arguments random_state=0 - this ensures that we get a consistent value back, not using random_state results in getting different results everytime\n")
    summaryFile.write("The other argument that get's supplied is max_leaf_nodes=7, which sets the max number of splits in the tree we have (I played around with this in my jupyter notebook file:'Jupyter Analysis\iris_prelim_analysis.ipynb')\n")
    summaryFile.write("I then split the iris dataset into training and validating for the X and y, this ensures that we aren't overfitting the data.\n")
    summaryFile.write("The model get's fitted using the training data for X and y.\n")
    summaryFile.write("Then using the fitted model, we predict the flower species by supplying our validated X data, this will let us know whether the model we've built so far is accurate or not.\n")
    summaryFile.write(f"Our predicted values are:\n{list(val_predictions)}\n")
    summaryFile.write(f"Our actual values that we're trying to predict are:\n{list(val_y)}\n")
    summaryFile.write("Finally we get the mean absolute error using the mean_absolute_error and supplying arguments for the validated y data and the values we predicted.\n")
    summaryFile.write(f"Our returned MAE is: {mae}\n")
    summaryFile.write("The MAE is very small, but we should also remember that our values 0,1,2 for y are also pretty small\nWhat this tells us is that across the validation data we are off by an average of 0.026, which is a pretty good guess for our purposes as it was wrong on only one flower.")
print("open df")

#Using graphviz to visualise the decision tree
from sklearn.tree import export_graphviz #this is used for exporting the decision tree model to dot data format which is of type str
import pydotplus  #pydotplus interprets the graphviz and we can then use this to save the graph
#from graphviz import Source

#
dot_data = export_graphviz(iris_model, out_file=None,  
                filled=True, rounded=False,
                feature_names=feature_names,
                label='all')

graph = pydotplus.graph_from_dot_data(dot_data)  

# Create PNG
graph.write_png("Decision Tree Analysis Output\iris.png")