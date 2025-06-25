from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import pickle

df=pd.read_csv("HR_comma_sep.csv")


le_dept = LabelEncoder()
le_salary = LabelEncoder()
df['Department'] = le_dept.fit_transform(df['Department'])
df['salary'] = le_salary.fit_transform(df['salary'])
feature_names = [col for col in df.columns if col != 'left']
X = df[feature_names]
y=df['left']

selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)

selected_columns = X.columns[selector.get_support()]
print(selected_columns.tolist())


X_1=df[selected_columns]

X_train,X_test,y_train,y_test=train_test_split(X_1,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_tarin_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

LR=LogisticRegression(max_iter=2000, class_weight='balanced')
LR.fit(X_tarin_scaled,y_train)

y_pred=LR.predict(X_test_scaled)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
acc=accuracy_score(y_test,y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")


with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": LR,
        "scaler": scaler,
        "selector": selector,
        "selected_columns": selected_columns,
        "le_dept": le_dept,
        "le_salary": le_salary
    }, f)