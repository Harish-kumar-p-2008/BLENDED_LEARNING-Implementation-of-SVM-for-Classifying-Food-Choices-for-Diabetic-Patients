# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Bring in the necessary libraries such as NumPy, Pandas, Scikit-learn, and Matplotlib.
2. Load the Dataset: Load the dataset containing food items and their nutritional information relevant to diabetic patients.
3. Data Preprocessing: Handle missing values, encode categorical variables if present, and perform feature scaling to normalize the data.
4. Define Features and Target: Separate the dataset into features (X) such as calories, carbohydrates, sugar, fiber, etc., and the target variable (y) indicating whether the food is suitable or not suitable for diabetic patients.
5. Split Data: Divide the dataset into training and testing sets using a train-test split method.
6. Build Support Vector Machine Model: Initialize the Support Vector Machine (SVM) classifier with an appropriate kernel (such as linear or RBF).
7. Train the Model: Fit the SVM model using the training dataset to learn the pattern of food choices suitable for diabetic patients.
8. Evaluate Performance: Evaluate the model using metrics such as accuracy score, confusion matrix, and classification report.
9. Display Model Results: Display the performance results and important model outputs.
10. Make Predictions & Compare: Use the trained SVM model to predict food suitability for diabetic patients and compare the predicted results with the actual values from the test dataset.

## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: Harish Kumar P
RegisterNumber: 25006070

import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('food_items_binary.csv')

print(data.head())
print(data.columns)

features=['Calories','Total Fat','Saturated Fat','Sugars','Dietary Fiber','Protein']
target='class'

X=data[features]
y=data[target]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

svm=SVC()

param_grid={
    'C':[0.1,1,10,100],
    'kernel':['linear','rbf'],
    'gamma':['scale','auto']
}

grid_search = GridSearchCV(svm,param_grid,cv=5,scoring='accuracy')
grid_search.fit(X_train,y_train)

best_model=grid_search.best_estimator_
print("Name: Harish Kumar P")
print("Register Number: 25006070")
print("Best Parameters:",grid_search.best_params_)

y_pred=best_model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print("Name: Harish Kumar P")
print("Register Number: 25006070")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
*/
```

## Output:
![alt text](<Screenshot 2026-03-12 153656.png>)
![alt text](<Screenshot 2026-03-12 153705.png>)
![alt text](<Screenshot 2026-03-12 153711.png>)
![alt text](<Screenshot 2026-03-12 153717.png>)
![alt text](<Screenshot 2026-03-12 153723.png>)

## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
