import pandas as pd
from ML import *
from funcs import *


data = pd.read_csv('Covid Data.csv')
print(data)
x = data[['SEX', 'TOBACCO', 'PREGNANT',	'DIABETES',	'COPD',	'ASTHMA',	'INMSUPR',	'HIPERTENSION',	'OTHER_DISEASE',	'CARDIOVASCULAR',	'OBESITY',	'RENAL_CHRONIC'
]]
y = data['CLASIFFICATION_FINAL']
#
x = x.to_numpy()
y = y.to_numpy()
#
# x[:,0] = x[:,0]/np.linalg.norm(x[:,0])
# x[:,1] = x[:,1]/np.linalg.norm(x[:,1])
# x[:,2] = x[:,2]/np.linalg.norm(x[:,2])
# x[:,3] = x[:,3]/np.linalg.norm(x[:,3])
# x[:,4] = x[:,4]/np.linalg.norm(x[:,4])

#
model = ClassificationTree()
model.fit(x,y)
print(model.root.mass)
print(model.predict(x))