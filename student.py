import pandas as pd

data = pd.read_csv('student-mat.csv',sep=';')
data = data[["sex","age","guardian","traveltime","studytime","failures",
             "internet","health","absences","G1","G2","G3"]]
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#sex
label_sex = LabelEncoder()
X[:,0] = label_sex.fit_transform(X[:,0])
#guardian
label_guardian = LabelEncoder()
X[:,2] = label_guardian.fit_transform(X[:,2])
#internet
label_internet = LabelEncoder()
X[:,6] = label_internet.fit_transform(X[:,6])

#One Hot Encoder
oneHot_sex = OneHotEncoder(categorical_features=[0,2,6])
X = oneHot_sex.fit_transform(X).toarray()

#Test Train Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

# Fitting the SVR Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,y_train)

# Predicting a new result
y_pred_1 = regressor.predict(X_test)
score = regressor.score(X_test,y_test)

#Saving regressor
from sklearn.externals import joblib

joblib.dump(regressor,'svm_model.pkl')

#Loading the regressor
regressor_new = joblib.load('svm_model.pkl')