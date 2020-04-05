import pandas as pd
from sklearn.tree import DecisionTreeRegressor #specify the model
from sklearn.model_selection import train_test_split #to split data in train and validate
from sklearn.metrics import mean_absolute_error

#iris_df is the dataframe storing the full iris dataset
iris_df = pd.read_csv('iris-flower-dataset\IRIS.csv')
print("open df")
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

# Supplying a numeric value to the random_state argument guarantees we get the same split every time this script is ran.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
print("train and val")
# Define model
iris_model = DecisionTreeRegressor(random_state=0, max_leaf_nodes=7)
print("Model defined")
# Fit model
iris_model.fit(train_X, train_y)
print("model fitted")

# get predicted category on validation data
val_predictions = iris_model.predict(val_X)
print("model predictions")

print(val_predictions)
#get the mean absolute error like before
print(mean_absolute_error(val_y, val_predictions))

#This is a weird 'hack' to get the correct path environment when using conda for graphviz

import os
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

#Using graphviz to visualise the decision tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()

export_graphviz(iris_model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png("png"))