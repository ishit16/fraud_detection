import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored as cl
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score 

df = pd.read_csv('creditcard.csv')
df.drop('Time', axis = 1, inplace = True)

cases = len(df)
nonfraudcount = len(df[df.Class == 0])
fraudcount = len(df[df.Class == 1])


## Since the ratio of fraudcount to nonfraudcount is highly imbalanced
## The data needs to be carefully modeled

nonfraudcases = df[df.Class == 0]
fraudcases = df[df.Class == 1]

sc = StandardScaler()
amount = df['Amount'].values
df['Amount'] = sc.fit_transform(amount.reshape(-1,1))

## Data Split

x = df.drop('Class', axis = 1).values
y = df['Class'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


## Implementing Models

# K-Nearest Neighbors

n = 5

knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)

# Decision Tree

tree_model = DecisionTreeClassifier(max_depth=4, criterion='entropy')
tree_model.fit(x_train, y_train)
tree_pred = tree_model.predict(x_test)

# Logistic Regression

lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)

# SVM

svm = SVC()
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)

# Random Forest Tree

rf = RandomForestClassifier(max_depth = 4)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)

# XGBoost

xgb = XGBClassifier(max_depth = 4)
xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)

acc_k = accuracy_score(y_test, knn_pred)
print(f"Accuracy of KNN = {acc_k}")
acc_dt = accuracy_score(y_test, tree_pred)
print(f"Accuracy Score of Decision Tree = {acc_dt}")
acc_lr = accuracy_score(y_test, lr_pred)
print(f"Accuracy Score of Logistic Regression = {acc_lr}")
acc_svc = accuracy_score(y_test, svm_pred)
print(f"Accuracy Score of SVM = {acc_svc}")
acc_rf = accuracy_score(y_test, rf_pred)
print(f"Accuracy Score of Random Forest = {acc_rf}")
acc_xgb = accuracy_score(y_test, xgb_pred)
print(f"Accuracy Score of XGBoost = {acc_xgb}")

# 3. Confusion Matrix

# defining the plot function

def plot_confusion_matrix(cm, classes, title, normalize = False, cmap = plt.cm.Blues):
    title = 'Confusion Matrix of {}'.format(title)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix for the models

tree_matrix = confusion_matrix(y_test, tree_yhat, labels = [0, 1]) # Decision Tree
knn_matrix = confusion_matrix(y_test, knn_yhat, labels = [0, 1]) # K-Nearest Neighbors
lr_matrix = confusion_matrix(y_test, lr_yhat, labels = [0, 1]) # Logistic Regression
svm_matrix = confusion_matrix(y_test, svm_yhat, labels = [0, 1]) # Support Vector Machine
rf_matrix = confusion_matrix(y_test, rf_yhat, labels = [0, 1]) # Random Forest Tree
xgb_matrix = confusion_matrix(y_test, xgb_yhat, labels = [0, 1]) # XGBoost

# Plot the confusion matrix

plt.rcParams['figure.figsize'] = (6, 6)

# 1. Decision tree

tree_cm_plot = plot_confusion_matrix(tree_matrix, 
                                classes = ['Non-Default(0)','Default(1)'], 
                                normalize = False, title = 'Decision Tree')
plt.savefig('tree_cm_plot.png')
plt.show()

# 2. K-Nearest Neighbors

knn_cm_plot = plot_confusion_matrix(knn_matrix, 
                                classes = ['Non-Default(0)','Default(1)'], 
                                normalize = False, title = 'KNN')
plt.savefig('knn_cm_plot.png')
plt.show()

# 3. Logistic regression

lr_cm_plot = plot_confusion_matrix(lr_matrix, 
                                classes = ['Non-Default(0)','Default(1)'], 
                                normalize = False, title = 'Logistic Regression')
plt.savefig('lr_cm_plot.png')
plt.show()

# 4. Support Vector Machine

svm_cm_plot = plot_confusion_matrix(svm_matrix, 
                                classes = ['Non-Default(0)','Default(1)'], 
                                normalize = False, title = 'SVM')
plt.savefig('svm_cm_plot.png')
plt.show()

# 5. Random forest tree

rf_cm_plot = plot_confusion_matrix(rf_matrix, 
                                classes = ['Non-Default(0)','Default(1)'], 
                                normalize = False, title = 'Random Forest Tree')
plt.savefig('rf_cm_plot.png')
plt.show()

# 6. XGBoost

xgb_cm_plot = plot_confusion_matrix(xgb_matrix, 
                                classes = ['Non-Default(0)','Default(1)'], 
                                normalize = False, title = 'XGBoost')
plt.savefig('xgb_cm_plot.png')
plt.show()