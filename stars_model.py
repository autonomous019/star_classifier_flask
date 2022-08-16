#Imported Stuff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report
import pickle

'''
ML work flow:
1. Collect the data
2. Visualize the data
3. Clean the data
4. Train the model
5. Evaluate
6. Hyperparameter tuning
7. Choose the best model and prediction
see for more info, https://towardsdatascience.com/regression-analysis-for-beginners-using-tree-based-methods-2b65bd193a7#bb44


'''


#Reading Dataset
df = pd.read_csv('data/stars csv.csv')
print(df.head())


print(df.shape)

#Checking for null values
print(df.isnull().sum())

print(df.describe())

print(df.corr())
sns.pairplot(df)
plt.show()

#Correlations of each feature in dataset
corrmat = df.corr()
top_features = corrmat.index
plt.figure(figsize = (20,20))

g = sns.heatmap(df[top_features].corr(), annot = True, cmap = "Blues")
plt.show()

#plt.figure()
#df.hist(figsize=(20,20))
#plt.show()

#Setting independant and target variables
X = df.drop(['Star type', 'Star color', 'Spectral Class'],axis=1) #dropped categorical fields, easiest to convert to int in csv file
y = df['Star type']

#Splitting Data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42,shuffle=True)

print("X data head")
print(X.head())
print ("Y data head")
print(y.head())



data = pd.DataFrame(np.arange(12).reshape((4,3)), columns=['a', 'b', 'c'])
label = pd.DataFrame(np.random.randint(2, size=4))
print("data")
print(data) #X, the data minus what we are trying to solve for, the independent variable is X
print("label") #values of Y, dependent variable, expressed in maths as y=f(x)
print(label)
#dtrain = xgb.DMatrix(data, label=label)
#


xgbc = XGBClassifier(max_depth=3, #how many levels of tree to grow, higher the num greater chance of overfitting
                     subsample = 0.5, #fraction of observations to be randomly sampled for each tree
                     n_estimators=200,
                     learning_rate=0.2, #alias of eta hyperparam, the step size shrinkage used in update to prevent overfit should be .01-.2
                     min_child_weight=1, #min sum of weights of all obs rquired in a child, high val can lead to underfitting, tune using CV
                     reg_alpha=0, #L! loss func regularization term on weights. higher more conservative the model
                     reg_lambda=1 #L2 loss func regularization term on weights, higher more conservative the model
                     )
#more info on hyperparams https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook
print(xgbc)


xgbc.fit(X_train, y_train)
y_predict = xgbc.predict(X_test)
y_train_predict = xgbc.predict(X_train)

# - cross validataion
scores = cross_val_score(xgbc, X_train, y_train, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgbc, X_train, y_train, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

ypred = xgbc.predict(X_test)
cm = confusion_matrix(y_test,ypred)
print(cm)

print('train accuracy', accuracy_score(y_train, y_train_predict))
print('test accuracy', accuracy_score(y_test, ypred))


#Feature importances and visualising it see https://www.kaggle.com/general/237792
print(xgbc.feature_importances_)

feat_importances = pd.Series(xgbc.feature_importances_, index = X.columns)
feat_importances.nlargest(5).plot(kind = 'barh')
plt.show()

xgbc.save_model("model.json")

#Pickling and dumping
file = open('xgbcl_model.pkl', 'wb')
pickle.dump(xgbc, file)





