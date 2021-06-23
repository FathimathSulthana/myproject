import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
import numpy as np
import matplotlib.pyplot as plt

rf_clf = RandomForestClassifier()
svm_clf = SVC()

#df_2.csv
print('df_2.csv')
df = pd.read_csv('df_2.csv')

X = df.drop(['index','Result'], axis=1)
y = df['Result']

scoring = {'acc': 'accuracy', 'prec_macro': 'precision_macro', 'rec_micro': 'recall_macro'}

print('rf')
scores = cross_validate(rf_clf,X,y,cv=5,scoring=scoring)
print('Accuracy: ',scores['test_acc'].mean())
print('Precision: ',scores['test_prec_macro'].mean())
print('Recall: ',scores['test_rec_micro'].mean())

x_rf = np.array([scores['test_acc'].mean(), scores['test_prec_macro'].mean(), scores['test_rec_micro'].mean()])

print('svm')
scores = cross_validate(svm_clf,X,y,cv=5,scoring=scoring)
print('Accuracy: ',scores['test_acc'].mean())
print('Precision: ',scores['test_prec_macro'].mean())
print('Recall: ',scores['test_rec_micro'].mean())

x_svm = np.array([scores['test_acc'].mean(), scores['test_prec_macro'].mean(), scores['test_rec_micro'].mean()])

label = ['accuracy', 'precision', 'recall']

w = 0.25
fig = plt.subplots(figsize =(12, 8))

br1 = np.arange(len(x_rf))
br2 = [x + w for x in br1]
 
plt.bar(br1, x_rf, color ='r', width = w, edgecolor ='grey', label ='rf')
plt.bar(br2, x_svm, color ='g', width = w, edgecolor ='grey', label ='svm')

plt.xticks([r+w/2 for r in range(len(x_rf))], label)
plt.ylabel('values')
plt.xlabel('evaluation method')
plt.title('comparison btw rf and svm for df_2')
 
plt.legend()
plt.savefig('graph_df2.png')
#plt.show()

#df_1.csv
print('df_1.csv')
df = pd.read_csv('df_1.csv')

X = df.drop(['Urls','Result'], axis=1)
y = df['Result']

scaler = StandardScaler()  
X = scaler.fit_transform(X)

scoring = {'acc': 'accuracy', 'prec_macro': 'precision_macro', 'rec_micro': 'recall_macro'}

print('rf')
scores = cross_validate(rf_clf,X,y,cv=5,scoring=scoring)
print('Accuracy: ',scores['test_acc'].mean())
print('Precision: ',scores['test_prec_macro'].mean())
print('Recall: ',scores['test_rec_micro'].mean())

x_rf = np.array([scores['test_acc'].mean(), scores['test_prec_macro'].mean(), scores['test_rec_micro'].mean()])

print('svm')
scores = cross_validate(svm_clf,X,y,cv=5,scoring=scoring)
print('Accuracy: ',scores['test_acc'].mean())
print('Precision: ',scores['test_prec_macro'].mean())
print('Recall: ',scores['test_rec_micro'].mean())

x_svm = np.array([scores['test_acc'].mean(), scores['test_prec_macro'].mean(), scores['test_rec_micro'].mean()])

label = ['accuracy', 'precision', 'recall']

w = 0.25
fig = plt.subplots(figsize =(12, 8))

br1 = np.arange(len(x_rf))
br2 = [x + w for x in br1]
 
plt.bar(br1, x_rf, color ='r', width = w, edgecolor ='grey', label ='rf')
plt.bar(br2, x_svm, color ='g', width = w, edgecolor ='grey', label ='svm')

plt.xticks([r+w/2 for r in range(len(x_rf))], label)
plt.ylabel('values')
plt.xlabel('evaluation method')
plt.title('comparison btw rf and svm for df_1')
 
plt.legend()
plt.savefig('graph_df1.png')
#plt.show()