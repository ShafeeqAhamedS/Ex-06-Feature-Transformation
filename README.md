# Ex-06-Feature-Transformation

## AIM

To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM

### STEP 1:
Read the given Data

### STEP 2:
Clean the Data Set using Data Cleaning Process

### STEP 3:
Apply Feature Transformation techniques to all the features of the data set

### STEP 4:
Print the transformed features

## PROGRAM:

```
# NAME: Shafeeq Ahamed.S
# REG NO: 212221230092
```
### Importing Libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
```
### Reading CSV File
```
df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semester 3/19AI403 _Intro to DS/Exp_6/Data_to_Transform.csv")
df
```
### Basic Process
```
df.head()

df.info()

df.describe()

df.tail()

df.shape

df.columns

df.isnull().sum()

df.duplicated()
```
### Before Transformation
```
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()
```
### Log Transformation
```
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()


df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()
```
### Reciprocal Transformation
```
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
### Square Root Transformation
```
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
### Power Transformation
```
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()


from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")

df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```
### Quantile Transformation
```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')

df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```

## OUTPUT:
### Reading CSV File
![df](./df)
### Basic Process

#### Head
![head](./head)
#### Info
![info](./info)
#### Describe
![des](./describe)
#### Tail
![tail](./tail)
#### Shape
![shape](./shape)
#### Columns
![cols](./cols)
#### Null Values
![null](./null)
#### Duplicate Values
![duplicate](./duplicate)

### Before Transformation

#### Highly Positive Skew
![HPS](./HPS)
#### Highly Negative Skew
![HNS](./HNS)
#### Moderate Positive Skew
![MPS](./MPS)
#### Moderate Negative Skew
![MNS](./MNS)

### Log Transformation

#### Highly Positive Skew
![HPS](./log_transformation_HPS)
#### Moderate Positive Skew
![MPS](./log_transformation_MPS)

### Reciprocal Transformation

#### Highly Positive Skew
![HPS](./Reciprocal_transformation_HPS)

### Square Root Transformation

#### Highly Positive Skew
![HPS](./sqrt_transformation_HPS)

### Quantile Transformation

#### Moderate Negative Skew
![MNS](./Quantile_transformation_MNS)


## RESULT:
Thus feature transformation is done for the given dataset.
