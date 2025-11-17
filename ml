1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("suv_data.csv")
dataset.head()
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values
print (x)
print (y)
print(dataset[dataset.isnull().any(axis=1)])
bool_series=pd.isnull(dataset["Gender"])
dataset[bool_series]

bool_series=pd.notnull(dataset["Gender"])
dataset[bool_series]
dataset[10:25]
new_data=dataset.dropna(axis=0,how='any')
new_data
dataset.replace(to_replace=np.nan, value=-99)
dataset["Gender"].fillna("No Gender")
print("Old data frame length:", len(dataset))
print("New data frame length:", len(new_data))
print("Number of rows with at least 1 NA value:", len(dataset)-len(new_data))
new_df1 = dataset.ffill()
print(new_df1)
new_df3=dataset.dropna(how='all')
new_df3




2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('Titanic-Dataset.csv')
print(data)
x = data.drop('Survived', axis = 1)
y = data['Survived']
print(x)
print(y)
x.drop(['Name', 'Ticket', 'Cabin'],axis = 1, inplace = True)
print(x)
x['Age'] = x['Age'].fillna(x['Age'].mean())
print(x)
x['Embarked'] = x['Embarked'].fillna(x['Embarked'].mode()[0])
print(x)
x = pd.get_dummies(x, columns = ['Sex', 'Embarked'],prefix = ['Sex', 'Embarked'],drop_first
= True)
print(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print(x_train)
print(y_train)
from sklearn.preprocessing import StandardScaler
std_x = StandardScaler()
x_train = std_x.fit_transform(x_train)
x_test = std_x.transform(x_test)
print(x_train)



3

from pandas import read_csv
from numpy import set_printoptions
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from matplotlib import pyplot

path = 'diabetes.csv'
dataframe = read_csv(path)

print("--- Data ki pehli 5 rows ---")
print(dataframe.head())

y = dataframe['Outcome']
x = dataframe.drop('Outcome', axis=1)

feature_names = x.columns.tolist()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)

fs = SelectKBest(score_func=f_classif, k='all')

fs.fit(x_train, y_train)

x_train_fs = fs.transform(x_train)
x_test_fs = fs.transform(x_test)

print("\n--- Feature Scores (Kaun kitna zaroori hai) ---")
for i in range(len(fs.scores_)):
        print('Feature %d (%s): %f' % (i, feature_names[i], fs.scores_[i]))

print("\n--- Graph Display ---")
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.xticks(range(len(fs.scores_)), feature_names, rotation=45)
pyplot.title("Feature Importance Scores")
pyplot.show()



4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('loandata.csv')

print("--- Original Data Head ---")
print(df.head())

if 'Loan_ID' in df.columns:
    df = df.drop('Loan_ID', axis=1)

num_cols = ['LoanAmount', 'Loan_Amount_Term', 'ApplicantIncome', 'CoapplicantIncome']
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Education', 'Property_Area']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

le = LabelEncoder()

df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

print("\n--- Processed Data (Ready for ML) ---")
print(df.head())

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

num_features = 10
k_val = min(num_features, len(X.columns))

best_features = SelectKBest(score_func=chi2, k=k_val)
fit = best_features.fit(X, y)

df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)

feature_scores = pd.concat([df_columns, df_scores], axis=1)
feature_scores.columns = ['Feature', 'Score']

print(f"\n--- Top {k_val} Features ---")
top_features = feature_scores.nlargest(k_val, 'Score')
print(top_features)

plt.figure(figsize=(12, 6))
plt.bar(top_features['Feature'], top_features['Score'], color='blue')
plt.xlabel("Features", fontsize=14)
plt.ylabel("Chi2 Score", fontsize=14)
plt.title(f"Top {k_val} Features in Loan Data (Chi-Squared Test)", fontsize=16)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn import metrics
from  sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris 

iris = load_iris()
iris = sns.load_dataset('iris') 
iris.head()

print(iris)

x = iris.iloc[:,:-1]
y = iris.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=1)
treemodel = DecisionTreeClassifier()
treemodel.fit(x_train,y_train)
y_pred = treemodel.predict(x_test)
plt.figure(figsize=(20,30))
tree.plot_tree(treemodel,filled=True)
# plt.show()
# print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))





6
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd

dataset = pd.read_csv('User_Data.csv')
x= dataset.iloc[:,[2,3]].values
print(x)
y = dataset.iloc[:,4].values
print(y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train , y_train)
y_pred= classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Decision Tree Model:", accuracy)
