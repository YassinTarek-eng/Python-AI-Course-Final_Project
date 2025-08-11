import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.metrics import accuracy_score, precision_score , mean_absolute_error , mean_squared_error , f1_score , recall_score , confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np 

# Reading Data:

data = pd.read_csv("D:/heart_cleveland_upload.csv")

print(data.head(10))
print(f"The data shape : {data.shape}")            # Prints the file shape which is 100 rows x 13 columns
print("-"*150)
print(f"The data columns : {data.columns}")        # Prints the names of the 13 columns we have.
print("-"*150)
print(f"The data info : {data.info()}") 
print("-"*150)
print(f"The data info : {data.describe()}") 

# data = data.replace({"yes": 1, "no": 0})  # Here we replace each yes with 1 & each no with 0 as scikit models doesn't deal with strings.
X = data.drop(["condition"], axis = 1)                       # axis = 1 ---> to drop columns not rows.
y = data["condition"]

# Data Visualizing:
columns = ['sex']

# # Create pie chart for each column

for col in columns:
    counts = data[col].value_counts()
    
    plt.figure(figsize=(5, 5))
    plt.bar(counts.index, counts.values, color='skyblue', edgecolor='black')
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)  # Rotate labels if needed
    plt.tight_layout()       # Adjust layout to prevent label cut-off
    plt.show()


# Data Training & Testing (70/15/15):

X_train , X_temp , y_train , y_temp = train_test_split(X , y , test_size = 0.3 , random_state = 42)
X_val , X_test , y_val , y_test = train_test_split(X_temp , y_temp , test_size= 0.5 , random_state = 42)

# Model (1) :

Model1 = LogisticRegression(max_iter=1000) 
Model1.fit(X_train , y_train)

y_Pred_m1 = Model1.predict(X_test)
y_Pred_Val_m1 = Model1.predict(X_val)

mse_m1 = mean_squared_error(y_test , y_Pred_m1)
mae_m1 = mean_absolute_error(y_test , y_Pred_m1)
rmse_m1 = np.sqrt(mse_m1) 

mse_v_m1 = mean_squared_error(y_val , y_Pred_Val_m1)
mae_v_m1 = mean_absolute_error(y_val , y_Pred_Val_m1)
rmse_v_m1 = np.sqrt(mse_v_m1)

accuracy_m1 = accuracy_score(y_test, y_Pred_m1) * 100
accuracy_val_m1 = accuracy_score(y_val, y_Pred_Val_m1) * 100

# Precision, Recall, F1, Confusion Matrix (with zero_division fix)

precision_m1 = precision_score(y_val, y_Pred_Val_m1, zero_division=0)
recall_m1 = recall_score(y_val, y_Pred_Val_m1, zero_division=0)
f1_m1 = f1_score(y_val, y_Pred_Val_m1, zero_division=0)
conf_mat_m1 = confusion_matrix(y_val, y_Pred_Val_m1)

# Model(1) Results :

print("======== Model(1) Evaluation ========")
print("Validation Accuracy  : {:.2f}%".format(accuracy_val_m1))
print("Test Accuracy        : {:.2f}%".format(accuracy_m1))
print("Mean Absolute Error : ",mae_m1)
print("Mean Squared Error : ",mse_m1)
print("Root Absolute squared Error : ",rmse_m1)
print("Mean Absolute Error(val) : ",mae_v_m1)
print("Mean Squared Error(val) : ",mse_v_m1)
print("Root Absolute squared Error(val) : ",rmse_v_m1)
print("\nConfusion Matrix (Validation):\n", conf_mat_m1)
print("\nPrecision            : {:.2f}".format(precision_m1))
print("Recall               : {:.2f}".format(recall_m1))
print("F1 Score             : {:.2f}".format(f1_m1))

# Model (2) :

Model2 = DecisionTreeClassifier()
Model2.fit(X_train , y_train)

print("Decision Tree :")
plot_tree(Model2)                                                                       # Show the decision tree graph.
plt.show()

y_Pred_m2 = Model2.predict(X_test)
y_Pred_Val_m2 = Model2.predict(X_val)

mse_m2 = mean_squared_error(y_test , y_Pred_m2)
mae_m2 = mean_absolute_error(y_test , y_Pred_m2)
rmse_m2 = np.sqrt(mse_m2) 

mse_v_m2 = mean_squared_error(y_val , y_Pred_Val_m2)
mae_v_m2 = mean_absolute_error(y_val , y_Pred_Val_m2)
rmse_v_m2 = np.sqrt(mse_v_m2)

accuracy_m2 = accuracy_score(y_test, y_Pred_m2) * 100
accuracy_val_m2 = accuracy_score(y_val, y_Pred_Val_m2) * 100

# Precision, Recall, F1, Confusion Matrix (with zero_division fix)

precision_m2 = precision_score(y_val, y_Pred_Val_m2, zero_division=0)
recall_m2 = recall_score(y_val, y_Pred_Val_m2, zero_division=0)
f1_m2 = f1_score(y_val, y_Pred_Val_m2, zero_division=0)
conf_mat_m2 = confusion_matrix(y_val, y_Pred_Val_m2)

# Model(2) Results :

print("======== Model(2) Evaluation ========")
print("Validation Accuracy  : {:.2f}%".format(accuracy_val_m2))
print("Test Accuracy        : {:.2f}%".format(accuracy_m2))
print("Mean Absolute Error : ",mae_m2)
print("Mean Squared Error : ",mse_m2)
print("Root Absolute squared Error : ",rmse_m2)
print("Mean Absolute Error(val) : ",mae_v_m2)
print("Mean Squared Error(val) : ",mse_v_m2)
print("Root Absolute squared Error(val) : ",rmse_v_m2)
print("\nConfusion Matrix (Validation):\n", conf_mat_m2)
print("\nPrecision            : {:.2f}".format(precision_m2))
print("Recall               : {:.2f}".format(recall_m2))
print("F1 Score             : {:.2f}".format(f1_m2))