#import the iris dataset file is .csv so I'll first need to open this
#using pandas to open the file and want to store it as a dataframe to do some initial explorartory analysis
import pandas as pd
import matplotlib.pyplot as plt

#iris_df is the dataframe storing the full iris dataset
iris_df = pd.read_csv('iris-flower-dataset\IRIS.csv')

#after doing a bit of digging we can use .to_string on the describe pandas function to give us a str representation of our dataframe object
iris_df_description = iris_df.describe().to_string() #took inspiration from stackOverflow: https://stackoverflow.com/questions/51829923/write-a-pandas-dataframe-to-a-txt-file

#open summary.txt file or create if doesn't exist and write to it
with open('Preliminary Analysis Output\summary.txt', 'w') as summaryFile:
    #start writing to file, summary info from dataframe summary functions like describe(), head(), unique() etc.
    summaryFile.write('This is a summary of all the numeric columns of the Iris Flower Dataset:\n\n') #double new line for readability
    summaryFile.write(f"{iris_df_description} \n\n")
    summaryFile.write('This is the top 5 rows of the dataset, this shows us one of the flower species and all of the column names:\n\n')
    summaryFile.write(f"{iris_df.head().to_string()} \n\n") #write's head or top 5 rows of dataframe
    summaryFile.write("This is the tail: \n")
    summaryFile.write(f"{iris_df.tail().to_string()} \n\n") #write tail/bottom 5 rows of dataframe
    summaryFile.write(f"The total number of rows is {len(iris_df)}\n\n")
    uniqueSpecies = iris_df['species'].unique() #store all the unique species in a list
    for species in uniqueSpecies:
        summaryFile.write(f"{species}\n") #write each item of list to text file on new line
    summaryFile.write("\nPlease find all generated .png's of plots in this same folder")

#leaving in old print statements for now, so still getting messages in console
print("\nThis is a summary of all the numeric columns of the Iris Flower Dataset:")
print(iris_df_description)

print("This is the top 5 rows of the dataset, this shows us one of the flower species and all of the column names:")
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
