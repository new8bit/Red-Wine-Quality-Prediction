import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import data_preprocess as dp
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import joblib


x_train, y_train = dp.load_training_data()
x_test, y_test = dp.load_test_data()

pca_model = PCA(n_components=8).fit(x_train) 
trainpca = pca_model.transform(x_train)  
testpca = pca_model.transform(x_test)


x_train, y_train = dp.load_training_data()
x_test, y_test = dp.load_test_data()


bayes_model = GaussianNB()



param_grid = {
    'var_smoothing': [0, 1e-9, 1e-8, 1e-7, 1e-10]  # 调整 Laplace 平滑参数
}
kfold = KFold(n_splits=5, random_state=42, shuffle=True)
grid_search = GridSearchCV(bayes_model, param_grid, cv=kfold, verbose=3, n_jobs=-1, scoring='accuracy')


grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_

print("Best parameters:", grid_search.best_params_)

acc = []
d = {}
clf = best_model

clf.fit(x_train, y_train)
pred = clf.predict(x_test)
acc.append(accuracy_score(pred, y_test))
d = {'Modelling Algo': 'Best model', 'Accuracy': acc}
print(d)

joblib.dump(best_model, 'models/bayes.joblib')