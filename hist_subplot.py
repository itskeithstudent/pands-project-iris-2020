#This set of histograms ended up with so many lines it seemed sensible to put it in a separate .py file
import pandas as pd
import matplotlib.pyplot as plt

# hist_subplot_iris function returns a fig object that can be shown/saved (fig.show(), fig.save()) or further altered
def hist_subplot_iris(iris_df):
    print("Histograms Being Generated...")
    #get passed iris_df as argument from prelim_analysis.py and split it here
    iris_setosa_df = iris_df[iris_df['species']=='Iris-setosa']
    iris_versicolor_df = iris_df[iris_df['species']=='Iris-versicolor']
    iris_virginica_df = iris_df[iris_df['species']=='Iris-virginica']
    fig, axes = plt.subplots(ncols=4, nrows=5, figsize=(25, 25)) #do a subplot of a series of histograms
    #below I'll do a series of rows of histograms, where I will show the petal and sepal length and with across the different species
    #first row show all species
    axes[0][0].hist(iris_df['petal_length'])#using row,col indexes on ax
    axes[0][0].set_title('All species petal length')
    axes[0][0].set_ylabel('No. Petals')
    axes[0][0].set_xlabel('Petal Length')

    #histogram for petal_width, no filter
    axes[0][1].hist(iris_df['petal_width'])
    axes[0][1].set_title('All species petal width')
    axes[0][1].set_ylabel('No. Petals')
    axes[0][1].set_xlabel('Petal Width')

    #histogram for sepal_length, no filter
    axes[0][2].hist(iris_df['sepal_length'])
    axes[0][2].set_title('All species sepal length')
    axes[0][2].set_ylabel('No. Petals')
    axes[0][2].set_xlabel('Sepal Length')

    #histogram for sepal_width, no filter
    axes[0][3].hist(iris_df['sepal_width'])
    axes[0][3].set_title('All species sepal width')
    axes[0][3].set_ylabel('No. Petals')
    axes[0][3].set_xlabel('Sepal Width')

    #For the next row of my set of subplots, I'm going to show a overlayed histogram
    axes[1][0].hist(iris_setosa_df['petal_length'], color='red')
    axes[1][0].set_title('Iris-Setosa petal length')
    axes[1][0].legend(['Iris Setosa'])
    axes[1][0].set_ylabel('No. Petals')
    axes[1][0].set_xlabel('Petal Length')

    #ommitting comments here due to rinse and repeat, with slight changes
    axes[1][1].hist(iris_setosa_df['petal_width'], color='red')
    axes[1][1].set_title('Iris-Setosa petal width')
    axes[1][1].legend(['Iris Setosa'])
    axes[1][1].set_ylabel('No. Petals')
    axes[1][1].set_xlabel('Petal Width')

    axes[1][2].hist(iris_setosa_df['sepal_length'], color='red')
    axes[1][2].set_title('Iris-Setosa sepal length')
    axes[1][2].legend(['Iris Setosa'])
    axes[1][2].set_ylabel('No. Petals')
    axes[1][2].set_xlabel('Sepal Length')

    axes[1][3].hist(iris_setosa_df['sepal_width'], color='red')
    axes[1][3].set_title('Iris-Setosa sepal width')
    axes[1][3].legend(['Iris Setosa'])
    axes[1][3].set_ylabel('No. Petals')
    axes[1][3].set_xlabel('Sepal Width')

    axes[1][0].hist(iris_virginica_df['petal_length'], color='orange')
    axes[1][0].set_title('Iris-Virginica petal length')
    axes[1][0].legend(['Iris Virginica'])
    axes[1][0].set_ylabel('No. Petals')
    axes[1][0].set_xlabel('Petal Length')

    axes[1][1].hist(iris_virginica_df['petal_width'], color='orange')
    axes[1][1].set_title('Iris-Virginica petal width')
    axes[1][1].legend(['Iris Virginica'])
    axes[1][1].set_ylabel('No. Petals')
    axes[1][1].set_xlabel('Petal Width')

    axes[1][2].hist(iris_virginica_df['sepal_length'], color='orange')
    axes[1][2].set_title('Iris-Virginica sepal length')
    axes[1][2].legend(['Iris Virginica'])
    axes[1][2].set_ylabel('No. Petals')
    axes[1][2].set_xlabel('Sepal Length')

    axes[1][3].hist(iris_virginica_df['sepal_width'], color='orange')
    axes[1][3].set_title('Iris-Virginica sepal width')
    axes[1][3].legend(['Iris Virginica'])
    axes[1][3].set_ylabel('No. Petals')
    axes[1][3].set_xlabel('Sepal Width')

    axes[1][0].hist(iris_versicolor_df['petal_length'], color='green')
    axes[1][0].set_title('Iris-Versicolor petal length')
    axes[1][0].legend(['Iris Versicolor'])
    axes[1][0].set_ylabel('No. Petals')
    axes[1][0].set_xlabel('Petal Length')

    axes[1][1].hist(iris_versicolor_df['petal_width'], color='green')
    axes[1][1].set_title('Iris-Versicolor petal width')
    axes[1][1].legend(['Iris Versicolor'])
    axes[1][1].set_ylabel('No. Petals')
    axes[1][1].set_xlabel('Petal Width')

    axes[1][2].hist(iris_versicolor_df['sepal_length'], color='green')
    axes[1][2].set_title('Iris-Versicolor sepal length')
    axes[1][2].legend(['Iris Versicolor'])
    axes[1][2].set_ylabel('No. Petals')
    axes[1][2].set_xlabel('Sepal Length')

    axes[1][3].hist(iris_versicolor_df['sepal_width'], color='green')
    axes[1][3].set_title('Iris-Versicolor sepal width')
    axes[1][3].legend(['Iris Versicolor'])
    axes[1][3].set_ylabel('No. Petals')
    axes[1][3].set_xlabel('Sepal Width')

    #Here we start to add our different species in isolation and add our Iris_setosa_df row to the subplot
    axes[2][0].hist(iris_setosa_df['petal_length'], color='red')
    axes[2][0].set_title('Iris-Setosa petal length')
    axes[2][0].legend(['Iris Setosa'])
    axes[2][0].set_ylabel('No. Petals')
    axes[2][0].set_xlabel('Petal Length')

    axes[2][1].hist(iris_setosa_df['petal_width'], color='red')
    axes[2][1].set_title('Iris-Setosa petal width')
    axes[2][1].legend(['Iris Setosa'])
    axes[2][1].set_ylabel('No. Petals')
    axes[2][1].set_xlabel('Petal Width')

    axes[2][2].hist(iris_setosa_df['sepal_length'], color='red')
    axes[2][2].set_title('Iris-Setosa sepal length')
    axes[2][2].legend(['Iris Setosa'])
    axes[2][2].set_ylabel('No. Petals')
    axes[2][2].set_xlabel('Sepal Length')

    axes[2][3].hist(iris_setosa_df['sepal_width'], color='red')
    axes[2][3].set_title('Iris-Setosa sepal width')
    axes[2][3].legend(['Iris Setosa'])
    axes[2][3].set_ylabel('No. Petals')
    axes[2][3].set_xlabel('Sepal Width')

    axes[3][0].hist(iris_virginica_df['petal_length'], color='orange')
    axes[3][0].set_title('Iris-Virginica petal length')
    axes[3][0].legend(['Iris Virginica'])
    axes[3][0].set_ylabel('No. Petals')
    axes[3][0].set_xlabel('Petal Length')

    axes[3][1].hist(iris_virginica_df['petal_width'], color='orange')
    axes[3][1].set_title('Iris-Virginica petal width')
    axes[3][1].legend(['Iris Virginica'])
    axes[3][1].set_ylabel('No. Petals')
    axes[3][1].set_xlabel('Petal Width')

    axes[3][2].hist(iris_virginica_df['sepal_length'], color='orange')
    axes[3][2].set_title('Iris-Virginica sepal length')
    axes[3][2].legend(['Iris Virginica'])
    axes[3][2].set_ylabel('No. Petals')
    axes[3][2].set_xlabel('Sepal Length')

    axes[3][3].hist(iris_virginica_df['sepal_width'], color='orange')
    axes[3][3].set_title('Iris-Virginica sepal width')
    axes[3][3].legend(['Iris Virginica'])
    axes[3][3].set_ylabel('No. Petals')
    axes[3][3].set_xlabel('Sepal Width')

    #Finally we add our Iris_verisoclor_df row to the subplot
    axes[4][0].hist(iris_versicolor_df['petal_length'], color='green')
    axes[4][0].set_title('Iris-Versicolor petal length')
    axes[4][0].legend(['Iris Versicolor'])
    axes[4][0].set_ylabel('No. Petals')
    axes[4][0].set_xlabel('Petal Length')

    axes[4][1].hist(iris_versicolor_df['petal_width'], color='green')
    axes[4][1].set_title('Iris-Versicolor petal width')
    axes[4][1].legend(['Iris Versicolor'])
    axes[4][1].set_ylabel('No. Petals')
    axes[4][1].set_xlabel('Petal Width')

    axes[4][2].hist(iris_versicolor_df['sepal_length'], color='green')
    axes[4][2].set_title('Iris-Versicolor sepal length')
    axes[4][2].legend(['Iris Versicolor'])
    axes[4][2].set_ylabel('No. Petals')
    axes[4][2].set_xlabel('Sepal Length')

    axes[4][3].hist(iris_versicolor_df['sepal_width'], color='green')
    axes[4][3].set_title('Iris-Versicolor sepal width')
    axes[4][3].legend(['Iris Versicolor']) #weirdly for the legend need's to get argument as list item
    axes[4][3].set_ylabel('No. Petals')
    axes[4][3].set_xlabel('Sepal Width')
    fig.tight_layout() #and now we show the completed subplot
    #weirdly with the above there is incosistency with placement, so I should probably set top right here for all
    return fig #returns to what called it, so perform saving of subplot in main .py file not here