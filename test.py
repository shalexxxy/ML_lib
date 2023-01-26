import pandas as pd
from ML import *
from funcs import *
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor, _criterion

x = pd.DataFrame(load_boston().data)
y = pd.DataFrame(load_boston().target)
#data = pd.read_csv('Covid Data.csv').dropna()
#x = data[['SEX', 'TOBACCO', 'PREGNANT',	'DIABETES',	'COPD',	'ASTHMA',	'INMSUPR',	'HIPERTENSION',	'OTHER_DISEASE',	'CARDIOVASCULAR',	'OBESITY',	'RENAL_CHRONIC'
#]]
#y = data['CLASIFFICATION_FINAL']
#
x = x.to_numpy()

y = y.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)


#
# x[:,0] = x[:,0]/np.linalg.norm(x[:,0])
# x[:,1] = x[:,1]/np.linalg.norm(x[:,1])
# x[:,2] = x[:,2]/np.linalg.norm(x[:,2])
# x[:,3] = x[:,3]/np.linalg.norm(x[:,3])
# x[:,4] = x[:,4]/np.linalg.norm(x[:,4])

#
#model = RandomClassificationForest()
model = RegressionTree()
model.fit(x_train,y_train)
#model_def.fit(x_train,y_train)
print('learning completed')
print(model.lists[0])


#pred = model.predict(x_test)
#pred2 = model_def.predict(x_test)
#print(mean_absolute_percentage_error(y_test, pred))
#print('My model MAE: ', accuracy_score(y_test,np.array(pred)))
#print('sklearn model MAE: ', mean_absolute_error(y_test,np.array(pred2)))