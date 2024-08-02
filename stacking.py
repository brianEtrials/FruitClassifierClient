import pandas as pd
import numpy as np
from holoviews.ipython import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,average_precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


pd.set_option('display.max_columns', None)
df = pd.read_csv("/Users/preetpatel/Downloads/collected_image_data (1).csv", low_memory=False)
data = pd.read_csv("/Users/preetpatel/Downloads/collected_image_data (1).csv", low_memory=False)




#enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
#df_ohe = enc.fit_transform(df[["Type"]])
labeler = LabelEncoder()
df["Type"] = labeler.fit_transform(df["Type"])

#df = pd.DataFrame(arr_ohe, index = df.index)



#df = pd.concat([df, df_ohe], axis = 1).drop(columns=['Type'])
#df_enc = df.join(df_ohe)



#X = df.drop(["Machine failure"], axis=1)
#Y = df["Machine failure"]
print(df.head())
print(df["Type"].unique())

scaler = StandardScaler()
df = scaler.fit_transform(df)
#print(df)

scaler = MinMaxScaler()
X = scaler.fit_transform(df)

df = pd.DataFrame(X, columns= ['Type','Area', 'Perimeter', 'Major Axis Length', 'Minor Axis Length',
       'Eccentricity', 'ROI Mean (R)', 'ROI Mean (G)', 'ROI Mean (B)',
       'ROI Std Dev (R)', 'ROI Std Dev (G)', 'ROI Std Dev (B)'])
print(df.head())

log = LogisticRegression(max_iter=1000)
ann = MLPClassifier(max_iter=3000)

baseClassifiers = [('lr', log), ('ann', ann)]

finalEstimator = LogisticRegression(max_iter=1000)

stackingClassifier = StackingClassifier(estimators= baseClassifiers, final_estimator= finalEstimator)



kFold = KFold(n_splits=10, shuffle=True, random_state=42)

X = df.drop(["Type"], axis=1)
print(X.head())
print(X.tail())

labeler = LabelEncoder()
df["Type"] = labeler.fit_transform(df["Type"])

Y = df[["Type"]]
print(Y.head())

accuracies = []
for foldIndex, (trainIndex, testIndex) in enumerate(kFold.split(X)):

    XTrain, XTest = X.iloc[trainIndex], X.iloc[testIndex]
    YTrain, YTest = Y.iloc[trainIndex], Y.iloc[testIndex]
    print("Here")
    stackingClassifier.fit(XTrain, YTrain.values.ravel())
    yPred = stackingClassifier.predict(XTest)

    accuracy = accuracy_score(YTest, yPred)
    accuracies.append(accuracy)
    print(f"Index: {foldIndex} Accuracy: {accuracy}")
    #f1 = f1_score(YTest, yPred)
    #avg_precision = average_precision_score(YTest, yPred)
    #recall = recall_score(YTest, yPred, average=None)

print(np.mean(accuracies))




