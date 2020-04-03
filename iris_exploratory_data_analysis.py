#import the iris dataset file is .csv so I'll first need to open this
#using pandas to open the file and want to store it as a dataframe to do some initial explorartory analysis
import pandas as pd
import matplotlib.pyplot as plt
import hist_subplot #adjacent .py file, histogram subplot's took up many lines so separating it out to this file
import file_names #adjacent .py file containing variables holding string val's for different filenames
import seaborn as sns #using seaborn to create nicer visuals

def Write_Eda_Summary(iris_df):
    print("Exploratory Data Analysis Summary Being Written...")
    #open summary.txt file or create if doesn't exist and write to it
    with open('Preliminary Analysis Output\summary.txt', 'w') as summaryFile:
        #start writing to file, summary info from dataframe summary functions like describe(), head(), unique() etc.
        #after doing a bit of digging we can use .to_string on the describe pandas function to give us a str representation of our dataframe object
        iris_df_description = iris_df.describe().to_string() #took inspiration from stackOverflow: https://stackoverflow.com/questions/51829923/write-a-pandas-dataframe-to-a-txt-file
        summaryFile.write('This is a summary of all the numeric columns of the Iris Flower Dataset, from this we can infer the range of values we will be working with and the amount of variation:\n\n') #double new line for readability
        summaryFile.write(f"{iris_df_description} \n\n")
        summaryFile.write('This is the top 5 rows of the dataset, this shows us one of the flower species and all of the column names:\n\n')
        summaryFile.write(f"{iris_df.head().to_string()} \n\n") #write's head or top 5 rows of dataframe
        summaryFile.write("This is the tail, showing similar to the head but the opposite end of the datagrame: \n")
        summaryFile.write(f"{iris_df.tail().to_string()} \n\n") #write tail/bottom 5 rows of dataframe
        summaryFile.write(f"The total number of rows is {len(iris_df)}\n\n")
        summaryFile.write(f"The different species of flower in the dataset are:\n")
        uniqueSpecies = iris_df['species'].unique() #store all the unique species in a list
        for species in uniqueSpecies:
            summaryFile.write(f"{species}\n") #write each item of list to text file on new line

        summaryFile.write(f"\nDistribution of these species is:\n{iris_df.groupby('species').size().to_string()}\n") #Species distribution
        summaryFile.write(f"\nCorrelation of each of the parameters versus one another:\n")
        summaryFile.write(f"{iris_df.corr().to_string()}\n")

        summaryFile.write("\nPlease find all generated .png's of plots in this same folder")
        summaryFile.write("\n\nSome Observations:")
        summaryFile.write("\nFrom the 'Correlation Heatmap.png' we can see the correlation of our different parameters, \
        which shows that there is likely a strong relationship between petal_length and petal_width, also for sepal_length \
        and petal_length (though not as strongly correlated) and finally for petal_width and sepal_width, this also tells \
        us that the remaining parameters do not have an impact on one another")
        summaryFile.write("\nThis is further supported when looking at the 'Petal Width v Length.png', \
        'Petal v Sepal Length.png' and 'Petal v Sepal Width.png', which all show a strong linear relationship but also hint at there being groupings or clusters of data, most likely based on species")


def pyplot_plots(iris_df):
    print("Exploratory Data Analysis Pyplots Being Generated...")
    #save plots to Preliminary Analysis Output folder
    plt.plot( iris_df['petal_width'], iris_df['petal_length'], 'g.', label="petal_width vs petal_length") #declare the plot and define it's x and y axis
    plt.title("Petal Width vs. Petal Length across all flower species") #title for plot
    plt.xlabel("petal_width") #x axis label
    plt.ylabel("petal_length") #y axis label
    plt.legend() #handily the legend uses both the dot and linestlye
    plt.savefig(file_names.petalWidthVLengthPngName)
    plt.close()

    #plot sepal width vs sepal length
    plt.figure()
    plt.plot( iris_df['sepal_width'], iris_df['sepal_length'], 'r.', label="sepal_width vs sepal_length")
    plt.title("Sepal Width vs. Sepal Length across all flower species")
    plt.xlabel("sepal_width")
    plt.ylabel("sepal_length")
    plt.legend()
    plt.savefig(file_names.sepalWidthVLengthPngName)
    plt.close()

    #plot petal length vs sepal length
    plt.figure()
    plt.plot( iris_df['petal_length'], iris_df['sepal_length'], 'g.', label="petal_length vs sepal_length")
    plt.title("Petal Length vs. Sepal Length across all flower species")
    plt.xlabel("petal_length")
    plt.ylabel("sepal_length")
    plt.legend()
    plt.savefig(file_names.petalVSepalLengthPngName)
    plt.close()

    #plot petal width vs sepal width
    plt.plot( iris_df['petal_width'], iris_df['sepal_width'], 'b.', label="petal_width vs sepal_width")
    plt.title("Petal Width vs. Sepal Width across all flower species")
    plt.xlabel("petal_width")
    plt.ylabel("sepal_width")
    plt.savefig(file_names.petalVSepalWidthPngName)
    plt.close()

def seaborn_plots(iris_df, iris_setosa_df, iris_versicolor_df, iris_virginica_df):
    print("Fancy Seaborn Plots Being Hand-crafted...")
    #Start seaborn plot's
    myPairPlot = sns.pairplot(iris_df, hue="species")#this idea was got from seaborn official doc.:https://seaborn.pydata.org/examples/scatterplot_matrix.html
    myPairPlot.fig.suptitle("Pair Plot of entire iris dataset coloured by species",y = 1, x=0.45)
    plt.savefig(file_names.seabornPairplotPngName)
    plt.close()

    #Distribution plot's basically a fancier version of hist from matplotlib
    #for this plot I want one large one on top of all species together, then 3 beneath coloured by species
    gridsize = (2, 3)
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=3, rowspan=1)
    ax1.set_title("All Iris Species Distribution Plots")
    ax2 = plt.subplot2grid(gridsize, (1, 0))
    ax2.set_title("Iris Setosa")
    ax3 = plt.subplot2grid(gridsize, (1, 1))
    ax3.set_title("Iris Versicolor")
    ax4 = plt.subplot2grid(gridsize, (1, 2))
    ax4.set_title("Iris Virginica")
    sns.distplot(iris_df['petal_length'], ax=ax1, bins=10)#played around with a few different bin sizes, 10 seems pretty good
    sns.distplot(iris_setosa_df['petal_length'], ax=ax2, bins=10, color='red')
    sns.distplot(iris_versicolor_df['petal_length'], ax=ax3, bins=10, color='green')
    sns.distplot(iris_virginica_df['petal_length'], ax=ax4, bins=10, color='blue')
    plt.subplots_adjust(hspace=.4)
    plt.tight_layout()
    plt.savefig(file_names.petalLengthDistPngName)
    plt.close()

    gridsize = (2, 3)
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=3, rowspan=1)
    ax1.set_title("All Iris Species Distribution Plots")
    ax2 = plt.subplot2grid(gridsize, (1, 0))
    ax2.set_title("Iris Setosa")
    ax3 = plt.subplot2grid(gridsize, (1, 1))
    ax3.set_title("Iris Versicolor")
    ax4 = plt.subplot2grid(gridsize, (1, 2))
    ax4.set_title("Iris Virginica")
    sns.distplot(iris_df['petal_width'], ax=ax1, bins=10)
    sns.distplot(iris_setosa_df['petal_width'], ax=ax2, bins=10, color='red')
    sns.distplot(iris_versicolor_df['petal_width'], ax=ax3, bins=10, color='green')
    sns.distplot(iris_virginica_df['petal_width'], ax=ax4, bins=10, color='blue')
    plt.subplots_adjust(hspace=.4)
    plt.tight_layout()
    plt.savefig(file_names.petalWidthDistPngName)
    plt.close()

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(14, 10))
    sns.violinplot(x='species', y='petal_length', hue='species', data=iris_df, ax=axes[0][0])
    sns.violinplot(x='species', y='petal_width', hue='species', data=iris_df, ax=axes[0][1])
    sns.violinplot(x='species', y='sepal_length', hue='species', data=iris_df, ax=axes[1][0])
    sns.violinplot(x='species', y='sepal_width', hue='species', data=iris_df, ax=axes[1][1])
    fig.suptitle("Violin plot's showing distributions of each parameter by Species", y=0.99)
    plt.tight_layout()
    plt.savefig(file_names.allParamsViolinPlotPngName)
    plt.close()

    plt.figure(figsize=(12,8))
    sns.heatmap(iris_df.corr(),annot=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(file_names.corrHeatmapPngName)
    plt.close()

#initialise seaborn, this will apply to all future plot's
sns.set()
#iris_df is the dataframe storing the full iris dataset
iris_df = pd.read_csv('iris-flower-dataset\IRIS.csv')
#create dataframes by species for filtering convenience
iris_setosa_df = iris_df[iris_df['species']=='Iris-setosa']
iris_versicolor_df = iris_df[iris_df['species']=='Iris-versicolor']
iris_virginica_df = iris_df[iris_df['species']=='Iris-virginica']

Write_Eda_Summary(iris_df)
pyplot_plots(iris_df)
myMultiHistFig = hist_subplot.hist_subplot_iris(iris_df) #call funtion from hist_subplot.py file for generating hist subplot
myMultiHistFig.savefig(file_names.histPngName) #save hist subplot to .png
seaborn_plots(iris_df,iris_setosa_df,iris_versicolor_df,iris_virginica_df)