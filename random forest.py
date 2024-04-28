import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import joblib


# 1. 读取训练集和测试集
train_df = pd.read_csv(r'data/train.csv')
test_df = pd.read_csv(r'data/test.csv')
#train_df = pd.read_csv(r'data/clustered_data.csv')
#test_df = pd.read_csv(r'data/clustered_test.csv')

# 2. 特征抽取

pca_model = PCA(n_components=8).fit(train_df.drop(columns=['quality'])) #// 用训练集训练PCA模型
trainpca = pca_model.transform(train_df.drop(columns=['quality']))  #// 将规则应用到训练集
testpca = pca_model.transform(test_df.drop(columns=['quality']))  #// 将规则应用到测试集


# 3. 划分数据集
y_train = train_df['quality']
y_test = test_df['quality']
x_train = train_df.drop(columns=['quality'])
x_test = test_df.drop(columns=['quality'])
model = RandomForestClassifier()

# 5. 定义要调节的参数范围
param_grid = {
    'n_estimators': [50, 100, 200],             # 树的数量
    'max_depth': [10, 50, 100],            # 最大深度
    'min_samples_split': [2, 5, 10],            # 内部节点分裂所需的最小样本数
    'min_samples_leaf': [1, 2, 4],              # 叶子节点所需的最小样本数
    'max_features': ['sqrt', 'log2'],   # 每次分裂时考虑的特征数量
    'bootstrap': [True, False]                  # 是否使用自助采样
}

# 6. 使用网格搜索进行参数调优
grid_search = GridSearchCV(model, param_grid, cv=2, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(x_train, y_train)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 打印最佳参数
print("Best parameters:", grid_search.best_params_)

acc = []
d = {}
clf = best_model

clf.fit(x_train, y_train)
pred = clf.predict(x_test)
acc.append(accuracy_score(pred, y_test))
d = {'Modelling Algo': 'Best model', 'Accuracy': acc}
print(d)

kfold = KFold(n_splits=5, random_state=42, shuffle=True) # 5折交叉验证
scores = cross_val_score(clf, x_train, y_train, cv=kfold)  # 使用交叉验证评估模型
print(f'random forest: Mean Accuracy = {scores.mean()}, Standard Deviation = {scores.std()}')

joblib.dump(best_model, 'Random Forest.joblib')