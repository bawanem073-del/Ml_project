

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

data = pd.read_csv("C:/Users/LOQ/OneDrive/Desktop/student_scores.csv")

# data filter
data["result"] = np.where(data["exam_score"] >= 40, 1, 0)

X = data[["study_hours", "class_attendance", "sleep_hours"]]
y = data["result"] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

#model creation
model = LogisticRegression()
model.fit(X_train, y_train)

predicted_result = model.predict(X_test) 

#accuracy check
Acs = accuracy_score(y_test, predicted_result)
ps = precision_score(y_test, predicted_result)
rec = recall_score(y_test, predicted_result)
fs = f1_score(y_test, predicted_result)

cm = confusion_matrix(y_test, predicted_result)

print("Accuracy Score : ",Acs)
print("Precision Score: ", ps)
print("Recall Score : ", rec)
print("F1 Score : ", fs)
print("Confusion Matrix :")
print(cm)



# model Testing
data = [5, 75, 2]

result = model.predict([data])[0]

if result == 1:
    print("Student likely to Pass")
else:
    print("Student likely to Faill")    