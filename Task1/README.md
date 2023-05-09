# Linear Regression by Umar Anzar

Linear Regression is a type of supervised learning algorithm used in machine learning and statistics to predict a continuous target variable based on one or more predictor variables. It is a linear approach to modeling the relationship between a dependent variable and one or more independent variables.

## Import Libraries

- NumPy: A Python library used for numerical computing. It provides support for multidimensional arrays and matrices, along with functions to perform mathematical operations on them.

- Pandas: A library used for data manipulation and analysis. It provides data structures for efficient storage and manipulation of tabular data.

- Matplotlib: A plotting library for creating static, interactive, and animated visualizations in Python.

- Seaborn: A data visualization library based on Matplotlib.

- Scikit-learn: A machine learning library for Python. It provides a range of algorithms for classification, regression, clustering, and dimensionality reduction, along with tools for model selection and evaluation.


```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# allows for displaying plots inline within the notebook
%matplotlib inline 
```

## Import Dataset


```python
dataset = pd.read_csv('student_scores.csv')
dataset.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.5</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.1</td>
      <td>47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.2</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.5</td>
      <td>75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.5</td>
      <td>30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.5</td>
      <td>20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9.2</td>
      <td>88</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.5</td>
      <td>60</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.3</td>
      <td>81</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.7</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>



## Plot 2d Graph
By observing the scatter plot graph, it can be inferred that there is a strong relationship between scores and hours, which seems to be directly proportional. As the number of hours increase, the scores also tend to increase.


```python
sns.set(rc={'figure.figsize':(5,4)})
sns.set_style("whitegrid")

sns.scatterplot(x='Hours',y='Scores',data=dataset)
plt.show()
```


    
![png](output\output_6_0.png)
    


### Best Fit Line 
Used the regplot function to gain an understanding of the best-fit line on this graph


```python
sns.regplot(x='Hours',y='Scores',data=dataset, line_kws={"color": "red"})
plt.show()
```


    
![png](output_8_0.png)
    


## Preparing X and Y


```python
x = dataset.iloc[:,:1].values
y = dataset.iloc[:,-1].values
```

## Predicting Dataset

Data is split into two sets: training and testing data. The model is trained on the training data and then used to predict the target value of the test data. The error between the true and predicted target value is then calculated to evaluate the performance of the model.

### Train/Test Split


```python
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state = 0)
```


```python
model = LinearRegression()
model.fit(train_x,train_y)
m = model.coef_
c = model.intercept_
Y_axis = m*x + c
```


```python
sns.scatterplot(x='Hours',y='Scores',data=dataset)
sns.lineplot(x=x.flatten(),y=Y_axis.flatten(), color='red')
plt.show()
```


    
![png](output_15_0.png)
    


### Prediction


```python
y_pred = model.predict(test_x)
y_pred
```




    array([16.88414476, 33.73226078, 75.357018  , 26.79480124, 60.49103328])



### Evaluation
- mean_squared_error: A function from scikit-learn.metrics used to compute the mean squared error between the predicted and actual values.

- mean_absolute_error: A function from scikit-learn.metrics used to compute the mean absolute error between the predicted and actual values.

- r2_score: A function from scikit-learn.metrics used to compute the R-squared (coefficient of determination) regression score function.


```python
print('mean_squared_error', mean_squared_error(test_y, y_pred), '\n',
      'mean_absolute_error', mean_absolute_error(test_y, y_pred), '\n',
      'r2_score', r2_score(test_y, y_pred))
```

    mean_squared_error 21.598769307217413 
     mean_absolute_error 4.18385989900298 
     r2_score 0.9454906892105355
    

## Cross-Validation 

Cross-validation is a technique used to evaluate the performance of a machine learning model on unseen data. It is done by splitting the dataset into multiple folds, then training the model on each fold and evaluating it on the remaining folds. This process is repeated multiple times, with each fold used as a test set once. The final performance of the model is then averaged over all the folds.

In this case, the dataset is split at a random state different on each iteration. This means that each time the cross-validation process is run, a different set of training and test folds will be created. This helps to ensure that the performance of the model is not simply due to chance.

### Error Function
This function returns a data frame row containing the error metrics of model evaluation.


```python
def error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'mean_squared_error':mse, 'mean_absolute_error':mae, 'r2_score':r2}
```


```python
errorDf = pd.DataFrame(columns=['random_state', 'mean_squared_error', 'mean_absolute_error', 'r2_score'])

model = LinearRegression()
for rdmState in range(1,200,5):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state = rdmState)
    model.fit(train_x,train_y)
    pred_y = model.predict(test_x)
    result = error(test_y, pred_y)
    result['random_state'] = rdmState
    errorDf = errorDf.append(result, ignore_index=True)

errorDf.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>random_state</th>
      <th>mean_squared_error</th>
      <th>mean_absolute_error</th>
      <th>r2_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>68.880921</td>
      <td>7.882398</td>
      <td>0.842103</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.0</td>
      <td>18.765475</td>
      <td>4.230413</td>
      <td>0.972394</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11.0</td>
      <td>78.660909</td>
      <td>8.237073</td>
      <td>0.881990</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>50.029620</td>
      <td>6.682278</td>
      <td>0.835299</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21.0</td>
      <td>30.680774</td>
      <td>5.332780</td>
      <td>0.884031</td>
    </tr>
  </tbody>
</table>
</div>



### Plotting Histogram


```python
# Create subplots with 1 row and 3 columns
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.5)

# Plot histograms for each column on a separate subplot
sns.histplot(data=errorDf, x='mean_squared_error', ax=axes[0])
sns.histplot(data=errorDf, x='mean_absolute_error', ax=axes[1])
sns.histplot(data=errorDf, x='r2_score', ax=axes[2])
plt.show()
```


    
![png](output_25_0.png)
    


## Result
Overall, the model is performing well. The MSE and MAE are relatively small, and the R2 score is high. This means that the model is able to accurately predict the true values.

- The mean squared error is higher than the mean absolute error. This is because the squared error is more sensitive to outliers than the absolute error.
- The R2 score is close to 1. This means that the model is able to explain a large amount of the variance in the true values.


```python
errorDf.drop('random_state', axis=1).agg(['mean','median']).transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>median</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean_squared_error</th>
      <td>37.598356</td>
      <td>30.740782</td>
    </tr>
    <tr>
      <th>mean_absolute_error</th>
      <td>5.546015</td>
      <td>5.288847</td>
    </tr>
    <tr>
      <th>r2_score</th>
      <td>0.901040</td>
      <td>0.926632</td>
    </tr>
  </tbody>
</table>
</div>


