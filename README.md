<H3>ENTER YOUR NAME : VISALAN H </H3>
<H3>ENTER YOUR REGISTER NO: 212223240183</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv('Churn_Modelling.csv')
print(df)
X = df.iloc[:, :-1].values
print(X)
y = df.iloc[:, -1].values
print(y)
print(df.isnull().sum())
print(df.describe())
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(data))
print(df1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(X_train)
print(len(X_train))
print(X_test)
print(len(X_test))
```

## OUTPUT:
DATASET PREVIEW:

![image](https://github.com/user-attachments/assets/7cbf7c88-a06d-40b2-981a-c793b33bc474)

FEATURE MATRIX - X VALUES:

![image](https://github.com/user-attachments/assets/78a9654f-0de8-452b-89d3-0875c3462d74)

TARGET VECTOR - Y VALUES:

![image](https://github.com/user-attachments/assets/3f9955a1-068a-49b2-b9a2-132a60c0b4ff)

NULL VALUES CHECK:

![image](https://github.com/user-attachments/assets/40249d18-3014-4cbc-85ae-4a886cdb74d2)

DATASET STATISTICAL SUMMARY:

![image](https://github.com/user-attachments/assets/601df596-a38f-4fed-9cc2-6bb993a4d7ad)

NORMALIZED DATASET:

![image](https://github.com/user-attachments/assets/1b84f544-a74a-4d84-a6ac-f68f18233ea6)

TRAINING DATA AND TESTING DATA:

![image](https://github.com/user-attachments/assets/03bc2adb-9329-4a9b-9905-fe9ed86e14ff)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


