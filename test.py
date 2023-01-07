import pandas as pd
from ML import *
from funcs import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('Covid Data.csv')
print(data)
x = data[['SEX', 'TOBACCO', 'PREGNANT',	'DIABETES',	'COPD',	'ASTHMA',	'INMSUPR',	'HIPERTENSION',	'OTHER_DISEASE',	'CARDIOVASCULAR',	'OBESITY',	'RENAL_CHRONIC'
]]
y = data['CLASIFFICATION_FINAL']
#
x = x.to_numpy()
y = y.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33)


#
# x[:,0] = x[:,0]/np.linalg.norm(x[:,0])
# x[:,1] = x[:,1]/np.linalg.norm(x[:,1])
# x[:,2] = x[:,2]/np.linalg.norm(x[:,2])
# x[:,3] = x[:,3]/np.linalg.norm(x[:,3])
# x[:,4] = x[:,4]/np.linalg.norm(x[:,4])
print(set(y))

#
model = ClassificationTree()
model.fit(x_train,y_train, max_depth=10)
print(model.root.mass)
pred = model.predict(x_test)
print(accuracy_score(y_test, pred))
