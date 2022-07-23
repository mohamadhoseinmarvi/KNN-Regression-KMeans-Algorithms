import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

if __name__ == "__main__":
        
    df = pd.read_csv('data/breast-cancer-wisconsin.data')

    df.replace('?', -99999, inplace=True) 
    df.drop(['id'], 1, inplace=True) 

    x = np.array(df.drop(['class'], 1)) 
    y = np.array(df['class'])  
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train) 

    accuracy = clf.score(x_test, y_test)
    print('accuracy: {}'.format(accuracy))
    example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1]])
    example_measures = example_measures.reshape(len(example_measures), -1)

    tipos_de_cancer = {4 : "maligo", 2 : "benigno"}

    prediction = clf.predict(example_measures)
    print('Tipo de cancer é: {}'.format(tipos_de_cancer[prediction.item(0)]))

