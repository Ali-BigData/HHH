# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from scipy import  stats 
from sklearn.ensemble import GradientBoostingRegressor


Dataset = pd.read_excel("Concrete_Data.xls")

dataset = Dataset.rename(columns={'Cement (kg per  m^3 of  mixture)':'Cement','Water  (kg per  m^3 of  mixture)':'Water','Coarse Aggregate  (kg per  m^3 of  mixture)':'Coarse Aggregate','Fly Ash (kg per  m^3 of  mixture)': 'Fly Ash','Superplasticizer (kg per  m^3 of  mixture)':'Superplasticizer',
                                 'Fine Aggregate (kg per  m^3 of  mixture)':'Fine Aggregate','Concrete compressive strength(MPa, megapascals) ':'Concrete Strenght',
                                 'Blast Furnace Slag (kg per  m^3 of  mixture)':'Blast Furnace Slag','Age (day)':'Age'},inplace = False)


# Feature Engineering
Y = dataset['Concrete Strenght']
X = dataset.drop (columns=['Concrete Strenght'])

# Scalling data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X)
X = scaler.transform(X)
# convet it to dataframe.
X = pd.DataFrame(columns= dataset.columns[:8], data=X)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
Model = RandomForestRegressor()

param_grid= [{'n_estimators':[3,10,20,30,40], 'max_features':[1,2,3,4,5,6,7,8]}]
# GridSearchModel to obtain the best parameter

GridSearchModel = GridSearchCV(Model,param_grid=param_grid,cv= 5,return_train_score=True)

# fit (train) the model
GridSearchModel.fit(X_train, y_train)

y_pred = GridSearchModel.predict(X_test)

mse = mean_squared_error(y_test, y_pred )
rmse = np.sqrt(mse)

print('Best Score is :', GridSearchModel.best_score_)

print("Best cross-validation score: {:.2f}\n".format(GridSearchModel.best_score_))

print("Best parameters: {}".format(GridSearchModel.best_params_))
print("Best estimators: {}".format(GridSearchModel.best_estimator_))

print("Test accuracy: {}".format(GridSearchModel.score(X_test, y_test)))
print("Train accuracy: {}".format(GridSearchModel.score(X_train, y_train)))


# Saving model to disk
pickle.dump(GridSearchModel, open('model.bin','wb'))

# Loading model to compare the results
model_1 = pickle.load(open('model.pkl','rb'))
print(model_1.predict([[12, 29, 6,44,12,37,8,5]])) 

   
