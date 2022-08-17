#Imported Stuff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report
from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from sklearn.model_selection import cross_val_score
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

plt.interactive(True) #interactive mode to display plots
#plt.savefig('myplot1.png') plot to a file instead

#Reading Dataset
df = pd.read_csv('data/stars csv.csv')
print(df.head())


print(df.shape)

#Checking for null values
print(df.isnull().sum())

print(df.describe())


#what is the distribution of features to be used for X
data_num = df.drop(columns=['Star type', 'Star color', 'Spectral Class'])

'''
fig, axes = plt.subplots(len(data_num.columns)//3, 3, figsize=(15, 6))
i = 0
for triaxis in axes:
    for axes in triaxis:
        data_num.hist(column = data_num.columns[i], ax=axes)
        i = i+1
#plt.show()
'''

#what is the distro of the target variable Y, Star Type with respect to Spectral Class
sns.displot(
  data=df,
  x="Star type",
  hue="Spectral Class",
  kind="hist",
  height=6,
  aspect=1.4,
  bins=15
)
#plt.show()

print(df.corr())
sns.pairplot(df)
#plt.show()

#Correlations of each feature in dataset, 1 is the best value for correlated data
corrmat = df.corr()
top_features = corrmat.index
plt.figure(figsize = (20,20))

g = sns.heatmap(df[top_features].corr(), annot = True, cmap = "Blues")
#plt.show()

plt.figure()
df.hist(figsize=(20,20))
#plt.show()

#Setting independant and target variables, cleaning by dropping categorical and Target (Y) variable (feature)
X = df.drop(['Star type', 'Star color', 'Spectral Class'],axis=1) #dropped categorical fields, easiest to convert to int in csv file
y = df['Star type']

#Splitting Data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42,shuffle=True)

print("X data head")
print(X.head())
print ("Y data head")
print(y.head())


#experimental demonstration of x (data) and y (label)
data = pd.DataFrame(np.arange(12).reshape((4,3)), columns=['a', 'b', 'c'])
label = pd.DataFrame(np.random.randint(2, size=4))
print("data")
print(data) #X, the data minus what we are trying to solve for, the independent variable is X
print("label") #values of Y, dependent variable, expressed in maths as y=f(x)
print(label)
#dtrain = xgb.DMatrix(data, label=label)
#


############################### hyperparameter optimization################################
'''  #UNCOMMENT THIS TO DO HYPEROPT

num_estimator = [100, 150, 200, 250]

space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
         #'gamma': hp.uniform('gamma', 1, 9),
         'reg_alpha': hp.quniform('reg_alpha', 30, 180, 1),
         'reg_lambda': hp.quniform('reg_lambda', 30, 180, 1),
         #'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
         'subsample': hp.uniform('subsample', 0.5, 1),
         'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
         'n_estimators': hp.choice("n_estimators", num_estimator),
          'learning_rate': hp.quniform("learning_rate", 0.05, 0.2, 0.01)
         }


def hyperparameter_tuning(space):
    model = xgb.XGBRegressor(n_estimators=space['n_estimators'],
                             max_depth=int(space['max_depth']),
                             #gamma=space['gamma'],
                             subsample=space['subsample'],
                             reg_alpha=int(space['reg_alpha']),
                             reg_lambda=int(space['reg_lambda']),
                             min_child_weight=space['min_child_weight'],
                             #colsample_bytree=space['colsample_bytree'],
                             learning_rate=space['learning_rate']
                             )

    score_cv = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error").mean()
    return {'loss': -score_cv, 'status': STATUS_OK, 'model': model}


trials = Trials()
best = fmin(fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

print("Hyperopt results")
print(best)

#best was output as {
'learning_rate': 0.07, 
'max_depth': 18.0, 
'min_child_weight': 8.0, 
'n_estimators': 3, 
'reg_alpha': 30.0, 
'reg_lambda': 60.0, 
'subsample': 0.9896752647307849
}


# XGB parameters
xgb_reg_params = {
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
    'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'subsample':        hp.uniform('subsample', 0.8, 1),
    'n_estimators':     100,
}
xgb_fit_params = {
    'eval_metric': 'rmse',
    'early_stopping_rounds': 10,
    'verbose': False
}
xgb_para = dict()
xgb_para['reg_params'] = xgb_reg_params
xgb_para['fit_params'] = xgb_fit_params
xgb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))
for more information see: https://medium.com/towards-data-science/an-example-of-hyperparameter-optimization-on-xgboost-lightgbm-and-catboost-using-hyperopt-12bc41a271e

these values gave a badd result so have to check baseline arguments as valid or not
########## end hyperparamater optimization routine#################
'''

xgbc = XGBClassifier(max_depth=3, #how many levels of tree to grow, higher the num greater chance of overfitting
                     subsample = 0.5, #fraction of observations to be randomly sampled for each tree
                     n_estimators=200,
                     eval_metric=['merror','mlogloss'],
                     learning_rate=0.1, #alias of eta hyperparam, the step size shrinkage used in update to prevent overfit should be .01-.2
                     min_child_weight=1, #min sum of weights of all obs rquired in a child, high val can lead to underfitting, tune using CV
                     reg_alpha=0, #L! loss func regularization term on weights. higher more conservative the model
                     reg_lambda=1 #L2 loss func regularization term on weights, higher more conservative the model
                     )
#more info on hyperparams https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook
print(xgbc)


xgbc.fit(X_train, y_train)
y_predict = xgbc.predict(X_test)
y_train_predict = xgbc.predict(X_train)



##################Evalutation of results########################
# - cross validataion
scores = cross_val_score(xgbc, X_train, y_train, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())


'''
K-Fold Cross Validation
As there is never enough data to train your model, removing a part of it for validation poses a problem of underfitting. By reducing the training data, we risk losing important patterns/ trends in data set, which in turn increases error induced by bias. So, what we require is a method that provides ample data for training the model and also leaves ample data for validation. K Fold cross validation does exactly that.

In K Fold cross validation, the data is divided into k subsets. Now the holdout method is repeated k times, such that each time, one of the k subsets is used as the test set/ validation set and the other k-1 subsets are put together to form a training set. The error estimation is averaged over all k trials to get total effectiveness of our model. As can be seen, every data point gets to be in a validation set exactly once, and gets to be in a training set k-1 times. This significantly reduces bias as we are using most of the data for fitting, and also significantly reduces variance as most of the data is also being used in validation set. Interchanging the training and test sets also adds to the effectiveness of this method. As a general rule and empirical evidence, K = 5 or 10 is generally preferred, but nothing’s fixed and it can take any value.

sklearn provides the functionality for this cross check
'''
kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgbc, X_train, y_train, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

'''
Confusion Matrix 
more information: https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62

a performance measurement for machine learning classification problem where output can be two or more classes. It is a table with 4 different combinations of predicted and actual values.
True Positive, False Positive, True Negative, True Positive
It is extremely useful for measuring Recall, Precision, Specificity, Accuracy, and most importantly AUC-ROC curves.

Recall
    is calculated as Recall = TP / TP + FN
    The above equation can be explained by saying, from all the positive classes, how many we predicted correctly. Recall should be high as possible.

Precision
    is calculated as Precision = TP / TP + FP
    The above equation can be explained by saying, from all the classes we have predicted as positive, how many are actually positive. Precision should be high as possible.

Accuracy
    From all the classes (positive and negative), how many of them we have predicted correctly. In this case, it will be 4/7. Accuracy should be high as possible.

F-measure 
    is calculated as 2*Recall*Precision / Recall+Precision
    It is difficult to compare two models with low precision and high recall or vice versa. So to make them comparable, we use F-Score. F-score helps to measure Recall and Precision at the same time. It uses Harmonic Mean in place of Arithmetic Mean by punishing the extreme values more.
'''
ypred = xgbc.predict(X_test)
cm = confusion_matrix(y_test,ypred)
print(cm)

'''
Accuracy Score is not the best way to measure the fit of the model
'''
print('train accuracy', accuracy_score(y_train, y_train_predict))
print('test accuracy', accuracy_score(y_test, ypred))

'''
Additional evaluations on the fit of the model: log loss and classification error

logloss: Log loss, short for logarithmic loss is a loss function for classification that quantifies the price paid for the inaccuracy of predictions in classification problems. Log loss penalizes false classifications by taking into account the probability of classification.
Remember that there is another important metric heavily used to evaluate the performance of a classification algorithm - ROC-AUC score
Log-loss is indicative of how close the prediction probability is to the corresponding actual/true value (0 or 1 in case of binary classification). The more the predicted probability diverges from the actual value, the higher is the log-loss value.

eval_metric see https://xgboost.readthedocs.io/en/latest/python/examples/sklearn_evals_result.html
'''
xgbc.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=True)
# Load evals result by calling the evals_result() function
evals_result = xgbc.evals_result()

'''
print('Access logloss metric directly from validation_0:')
print(evals_result['validation_0']['mlogloss'])

print('')
print('Access metrics through a loop:')
for e_name, e_mtrs in evals_result.items():
    print('- {}'.format(e_name))
    for e_mtr_name, e_mtr_vals in e_mtrs.items():
        print('   - {}'.format(e_mtr_name))
        print('      - {}'.format(e_mtr_vals))

print('')
'''
print('Access complete dict:')
print(evals_result)

results = evals_result
epochs = len(results["validation_0"]["merror"])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots(figsize=(12, 12))
ax.plot(x_axis, results["validation_0"]["mlogloss"], label="Train")
ax.plot(x_axis, results["validation_1"]["mlogloss"], label="Test")
ax.legend()
pyplot.ylabel("Log Loss")
pyplot.title("XGBoost Log Loss")
pyplot.show()

fig, ax = pyplot.subplots(figsize=(12, 12))
ax.plot(x_axis, results["validation_0"]["merror"], label="Train")
ax.plot(x_axis, results["validation_1"]["merror"], label="Test")
ax.legend()
pyplot.ylabel("Classification Error")
pyplot.title("XGBoost Classification Error")
pyplot.show()




#########################Feature Importance####################################

'''
#Feature importances and visualising it see https://www.kaggle.com/general/237792
Be careful when interpreting your features importance in XGBoost, since the ‘feature importance’ results might be misleading!
# importance_types = [‘weight’, ‘gain’, ‘cover’, ‘total_gain’, ‘total_cover’]
f = ‘gain’

get_score(fmap='', importance_type='weight')
Get feature importance of each feature. For tree model Importance type can be defined as:

‘weight’: the number of times a feature is used to split the data across all trees.

‘gain’: the average gain across all splits the feature is used in.

‘cover’: the average coverage across all splits the feature is used in.

‘total_gain’: the total gain across all splits the feature is used in.

‘total_cover’: the total coverage across all splits the feature is used in.

'''
print(xgbc.feature_importances_)

feat_importances = pd.Series(xgbc.feature_importances_, index = X.columns)
feat_importances.nlargest(5).plot(kind = 'barh')




#wrapping up
plt.show()
xgbc.save_model("model.json") #save model in json format

#Pickling and dumping, saving model in pkl format
file = open('xgbcl_model.pkl', 'wb')
pickle.dump(xgbc, file)





