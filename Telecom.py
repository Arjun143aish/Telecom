import os
import numpy as np
import pandas as pd

os.chdir("C:\\Users\\user\\Documents\\Python\\Deep Learning\\ANN\\Complete-Deep-Learning-master\\ANN")

FullRaw = pd.read_csv("Churn_Modelling.csv")

FullRaw.isnull().sum()

FullRaw.drop(['RowNumber','CustomerId', 'Surname'], axis =1, inplace =True)

Category_Vars = (FullRaw.dtypes == 'object')
dummydf = pd.get_dummies(FullRaw.loc[:,Category_Vars],drop_first =True)

FullRaw2 = pd.concat([FullRaw.loc[:,~Category_Vars],dummydf], axis =1)

from sklearn.model_selection import train_test_split

Train,Test = train_test_split(FullRaw2,test_size =0.3, random_state =123)

Train_X = Train.drop(['Exited'], axis =1)
Train_Y = Train['Exited'].copy()
Test_X = Test.drop(['Exited'],axis =1)
Test_Y = Test['Exited'].copy()

from sklearn.preprocessing import StandardScaler

Train_Scaling = StandardScaler().fit(Train_X)
Train_X_Std = Train_Scaling.transform(Train_X)
Test_X_Std = Train_Scaling.transform(Test_X)

Train_X_Std = pd.DataFrame(Train_X_Std, columns = Train_X.columns)
Test_X_Std = pd.DataFrame(Test_X_Std, columns = Test_X.columns)

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout

Classifier = Sequential()
Classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation = 'relu',input_dim = 11))
Classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation = 'relu',input_dim = 11))
Classifier.add(Dense(units = 1,kernel_initializer ='glorot_uniform', activation ='sigmoid'))

Classifier.compile(optimizer = 'Adamax',loss = 'binary_crossentropy',metrics = ['accuracy'])

M1 = Classifier.fit(Train_X_Std,Train_Y,batch_size = 10,epochs = 100, validation_split =0.3)

Test_pred= Classifier.predict(Test_X_Std)
Test['Test_pred'] = Test_pred
Test['Test_Class'] = np.where(Test['Test_pred'] > 0.5,1,0)

from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score

Con_Mat = confusion_matrix(Test['Test_Class'],Test_Y)
sum(np.diag(Con_Mat))/Test_Y.shape[0]*100

from sklearn.model_selection import GridSearchCV
from keras.layers import Activation
from keras.wrappers.scikit_learn import KerasClassifier

def create_model(layers, activation):
    model = Sequential()
    for i,nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes,input_dim = Train_X_Std.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units =1,kernel_initializer ='glorot_uniform', activation ='sigmoid'))
    model.compile(optimizer ='Adamax',loss = 'binary_crossentropy',metrics = ['accuracy'])
    
    return(model)

model = KerasClassifier(build_fn =create_model, verbose = 0)  
                
my_layers = [[20],[20,40]]
my_activation = ['relu','sigmoid']
my_Param_grid = {'layers': my_layers, 'activation': my_activation, 'epochs' :[30,40,50],
                 'batch_size': [80,90,100], 'validation_split': [0.3]}

Grid = GridSearchCV(estimator = model,param_grid = my_Param_grid, scoring = 'accuracy',
                    cv = 5).fit(Train_X_Std,Train_Y)

Grid.best_score_

Grid_Df = pd.DataFrame.from_dict(Grid.cv_results_)


import pickle

pickle.dump(Classifier,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

