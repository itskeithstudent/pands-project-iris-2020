#import the iris dataset file is .csv so I'll first need to open this
#using pandas to open the file and want to store it as a dataframe to do some initial explorartory analysis
import pandas as pd
import matplotlib.pyplot as plt

#iris_df is the dataframe storing the full iris dataset
iris_df = pd.read_csv('iris-flower-dataset\IRIS.csv')

#print information about the dataset to the command line
print("\nThis is a summary of all the numeric columns of the Iris Flower Dataset:")
print(iris_df.describe())

print("\n This is the top 5 rows of the dataset, this shows us one of the flower species and all of the column names:")
print(iris_df.head())

print("\n This is the tail: ")
print(iris_df.tail())

print(f"\nThe total number of rows is {len(iris_df)}")

uniqueSpecies = iris_df['species'].unique()
print("\nThe individual Iris flower Species are:")
for species in uniqueSpecies:
    print(species)

#save plots to Preliminary Analysis Output folder
plt.plot( iris_df['petal_width'], iris_df['petal_length'], 'g.', label="petal_width vs petal_length") #declare the plot and define it's x and y axis
plt.title("Petal Width vs. Petal Length across all flower species") #title for plot
plt.xlabel("petal_width") #x axis label
plt.ylabel("petal_length") #y axis label
plt.legend() #handily the legend uses both the dot and linestlye
plt.savefig('Preliminary Analysis Output\Petal Width v Length.png')
plt.close()

#plot sepal width vs sepal length
plt.figure()
plt.plot( iris_df['sepal_width'], iris_df['sepal_length'], 'r.', label="sepal_width vs sepal_length")
plt.title("Sepal Width vs. Sepal Length across all flower species")
plt.xlabel("sepal_width")
plt.ylabel("sepal_length")
plt.legend()
plt.savefig('Preliminary Analysis Output\Sepal Width v Length.png')
plt.close()

#plot petal length vs sepal length
plt.figure()
plt.plot( iris_df['petal_length'], iris_df['sepal_length'], 'g.', label="petal_length vs sepal_length")
plt.title("Petal Length vs. Sepal Length across all flower species") 
plt.xlabel("petal_length") 
plt.ylabel("sepal_length") 
plt.legend() 
plt.savefig('Preliminary Analysis Output\Petal and Sepal Length.png')
plt.close()

#plot petal width vs sepal width
plt.plot( iris_df['petal_width'], iris_df['sepal_width'], 'b.', label="petal_width vs sepal_width")
plt.title("Petal Width vs. Sepal Width across all flower species")
plt.xlabel("petal_width") 
plt.ylabel("sepal_width") 
plt.savefig('Preliminary Analysis Output\Petal v Sepal Width.png')
plt.close()
