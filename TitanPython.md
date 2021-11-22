```python
import numpy as np
import pandas as pd
```


```python
titanicmd= pd.read_csv('titanic_MD.csv')
titanic= pd.read_csv('titanic.csv')
```


```python
print('titanicmd dataset shape:', titanicmd.shape)
titanicmd.head(15)
```

    titanicmd dataset shape: (183, 14)
    




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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_sd</th>
      <th>SibSp_sd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>NaN</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>1.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>1.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>PP 9549</td>
      <td>16.7000</td>
      <td>G6</td>
      <td>S</td>
      <td>1.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>Bonnell, Miss. Elizabeth</td>
      <td>female</td>
      <td>58.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>113783</td>
      <td>26.5500</td>
      <td>C103</td>
      <td>S</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>22</td>
      <td>1</td>
      <td>2</td>
      <td>Beesley, Mr. Lawrence</td>
      <td>male</td>
      <td>34.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>248698</td>
      <td>13.0000</td>
      <td>D56</td>
      <td>S</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>24</td>
      <td>1</td>
      <td>1</td>
      <td>Sloper, Mr. William Thompson</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>113788</td>
      <td>35.5000</td>
      <td>A6</td>
      <td>S</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>28</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr. Charles Alexander</td>
      <td>NaN</td>
      <td>19.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
      <td>3.0</td>
      <td>1.753355</td>
    </tr>
    <tr>
      <th>8</th>
      <td>53</td>
      <td>1</td>
      <td>1</td>
      <td>Harper, Mrs. Henry Sleeper (Myna Haxtun)</td>
      <td>female</td>
      <td>49.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>PC 17572</td>
      <td>76.7292</td>
      <td>D33</td>
      <td>C</td>
      <td>1.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>55</td>
      <td>0</td>
      <td>1</td>
      <td>Ostby, Mr. Engelhart Cornelius</td>
      <td>male</td>
      <td>65.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>113509</td>
      <td>61.9792</td>
      <td>B30</td>
      <td>C</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>63</td>
      <td>0</td>
      <td>1</td>
      <td>Harris, Mr. Henry Birkhardt</td>
      <td>male</td>
      <td>45.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>36973</td>
      <td>83.4750</td>
      <td>C83</td>
      <td>S</td>
      <td>1.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>67</td>
      <td>1</td>
      <td>2</td>
      <td>Nye, Mrs. (Elizabeth Ramell)</td>
      <td>female</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>C.A. 29395</td>
      <td>10.5000</td>
      <td>F33</td>
      <td>S</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>76</td>
      <td>0</td>
      <td>3</td>
      <td>Moen, Mr. Sigurd Hansen</td>
      <td>male</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>348123</td>
      <td>NaN</td>
      <td>F G73</td>
      <td>S</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>89</td>
      <td>1</td>
      <td>1</td>
      <td>Fortune, Miss. Mabel Helen</td>
      <td>female</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
      <td>3.0</td>
      <td>1.753355</td>
    </tr>
    <tr>
      <th>14</th>
      <td>93</td>
      <td>0</td>
      <td>1</td>
      <td>Chaffee, Mr. Herbert Fuller</td>
      <td>NaN</td>
      <td>46.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>W.E.P. 5734</td>
      <td>61.1750</td>
      <td>E31</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.unique(titanicmd['Sex'])
```




    array(['?', 'female', 'male'], dtype=object)




```python
titanicmd.isna().sum() #or df.isnull().sum()
```




    PassengerId     0
    Survived        0
    Pclass          0
    Name            0
    Sex             0
    Age            25
    SibSp           3
    Parch          12
    Ticket          0
    Fare            8
    Cabin           0
    Embarked       12
    Age_sd          3
    SibSp_sd        3
    dtype: int64




```python
cols = []
val = []
for col in titanicmd.select_dtypes(include='object').columns:
    cols.append(col)
    val.append(titanicmd[col].str.contains(r'\?').sum())
pd.DataFrame({
    'cols':cols,
    'val':val
})
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
      <th>cols</th>
      <th>val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Name</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sex</td>
      <td>51</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ticket</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cabin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Embarked</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanicmd.replace(r'\?', np.nan, regex = True, inplace = True)
```


```python
titanicmd.isna().sum()
```




    PassengerId     0
    Survived        0
    Pclass          0
    Name            0
    Sex            51
    Age            25
    SibSp           3
    Parch          12
    Ticket          0
    Fare            8
    Cabin           0
    Embarked       12
    dtype: int64




```python
titanicmd.dropna().shape
```




    (100, 12)




```python
from sklearn.impute import SimpleImputer
```


```python
### ImputacionSexo
```


```python
imp_modeS = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
```


```python
imp_modeS.fit_transform(titanicmd[['Sex']])
```




    array([['male'],
           ['female'],
           ['male'],
           ['female'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['female'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['female'],
           ['female'],
           ['male'],
           ['male'],
           ['female'],
           ['female'],
           ['female'],
           ['female'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['female'],
           ['female'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['female'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['female'],
           ['female'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['female'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['female'],
           ['female'],
           ['male'],
           ['female'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['female'],
           ['female'],
           ['male'],
           ['female'],
           ['female'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['male'],
           ['female'],
           ['female'],
           ['female'],
           ['female'],
           ['female'],
           ['male'],
           ['male'],
           ['male'],
           ['female'],
           ['male'],
           ['female'],
           ['male'],
           ['male']], dtype=object)




```python
##AgeImputation
```


```python
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
```


```python
imp_mean.fit_transform(titanicmd[['Age']])
```




    array([[38.        ],
           [35.        ],
           [54.        ],
           [35.69253165],
           [58.        ],
           [34.        ],
           [35.69253165],
           [19.        ],
           [49.        ],
           [65.        ],
           [45.        ],
           [29.        ],
           [25.        ],
           [35.69253165],
           [46.        ],
           [71.        ],
           [23.        ],
           [21.        ],
           [47.        ],
           [24.        ],
           [35.69253165],
           [54.        ],
           [19.        ],
           [37.        ],
           [24.        ],
           [36.5       ],
           [22.        ],
           [61.        ],
           [56.        ],
           [50.        ],
           [35.69253165],
           [ 3.        ],
           [44.        ],
           [58.        ],
           [ 2.        ],
           [40.        ],
           [31.        ],
           [32.        ],
           [38.        ],
           [35.69253165],
           [44.        ],
           [37.        ],
           [35.69253165],
           [62.        ],
           [30.        ],
           [52.        ],
           [40.        ],
           [35.69253165],
           [35.        ],
           [37.        ],
           [63.        ],
           [19.        ],
           [36.        ],
           [ 2.        ],
           [50.        ],
           [ 0.92      ],
           [17.        ],
           [30.        ],
           [24.        ],
           [18.        ],
           [31.        ],
           [40.        ],
           [35.69253165],
           [36.        ],
           [16.        ],
           [35.69253165],
           [38.        ],
           [29.        ],
           [41.        ],
           [45.        ],
           [35.69253165],
           [24.        ],
           [24.        ],
           [22.        ],
           [60.        ],
           [24.        ],
           [25.        ],
           [27.        ],
           [36.        ],
           [23.        ],
           [24.        ],
           [33.        ],
           [35.69253165],
           [28.        ],
           [50.        ],
           [14.        ],
           [64.        ],
           [ 4.        ],
           [35.69253165],
           [30.        ],
           [49.        ],
           [65.        ],
           [48.        ],
           [47.        ],
           [23.        ],
           [25.        ],
           [35.        ],
           [58.        ],
           [55.        ],
           [54.        ],
           [25.        ],
           [16.        ],
           [18.        ],
           [35.69253165],
           [47.        ],
           [34.        ],
           [30.        ],
           [35.69253165],
           [45.        ],
           [22.        ],
           [35.69253165],
           [50.        ],
           [17.        ],
           [48.        ],
           [39.        ],
           [53.        ],
           [36.        ],
           [39.        ],
           [39.        ],
           [36.        ],
           [18.        ],
           [35.69253165],
           [52.        ],
           [49.        ],
           [40.        ],
           [ 4.        ],
           [35.69253165],
           [61.        ],
           [21.        ],
           [80.        ],
           [32.        ],
           [24.        ],
           [48.        ],
           [56.        ],
           [58.        ],
           [47.        ],
           [31.        ],
           [36.        ],
           [27.        ],
           [15.        ],
           [31.        ],
           [49.        ],
           [35.69253165],
           [18.        ],
           [35.        ],
           [42.        ],
           [24.        ],
           [48.        ],
           [19.        ],
           [35.69253165],
           [27.        ],
           [27.        ],
           [29.        ],
           [35.        ],
           [36.        ],
           [21.        ],
           [70.        ],
           [19.        ],
           [ 6.        ],
           [33.        ],
           [36.        ],
           [51.        ],
           [35.69253165],
           [43.        ],
           [17.        ],
           [29.        ],
           [35.69253165],
           [49.        ],
           [11.        ],
           [39.        ],
           [33.        ],
           [52.        ],
           [27.        ],
           [39.        ],
           [16.        ],
           [35.69253165],
           [48.        ],
           [31.        ],
           [47.        ],
           [35.69253165],
           [56.        ],
           [19.        ],
           [35.69253165]])




```python
imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
```


```python
imp_mode.fit_transform(titanicmd[['Age']])
```




    array([[38.  ],
           [35.  ],
           [54.  ],
           [24.  ],
           [58.  ],
           [34.  ],
           [24.  ],
           [19.  ],
           [49.  ],
           [65.  ],
           [45.  ],
           [29.  ],
           [25.  ],
           [24.  ],
           [46.  ],
           [71.  ],
           [23.  ],
           [21.  ],
           [47.  ],
           [24.  ],
           [24.  ],
           [54.  ],
           [19.  ],
           [37.  ],
           [24.  ],
           [36.5 ],
           [22.  ],
           [61.  ],
           [56.  ],
           [50.  ],
           [24.  ],
           [ 3.  ],
           [44.  ],
           [58.  ],
           [ 2.  ],
           [40.  ],
           [31.  ],
           [32.  ],
           [38.  ],
           [24.  ],
           [44.  ],
           [37.  ],
           [24.  ],
           [62.  ],
           [30.  ],
           [52.  ],
           [40.  ],
           [24.  ],
           [35.  ],
           [37.  ],
           [63.  ],
           [19.  ],
           [36.  ],
           [ 2.  ],
           [50.  ],
           [ 0.92],
           [17.  ],
           [30.  ],
           [24.  ],
           [18.  ],
           [31.  ],
           [40.  ],
           [24.  ],
           [36.  ],
           [16.  ],
           [24.  ],
           [38.  ],
           [29.  ],
           [41.  ],
           [45.  ],
           [24.  ],
           [24.  ],
           [24.  ],
           [22.  ],
           [60.  ],
           [24.  ],
           [25.  ],
           [27.  ],
           [36.  ],
           [23.  ],
           [24.  ],
           [33.  ],
           [24.  ],
           [28.  ],
           [50.  ],
           [14.  ],
           [64.  ],
           [ 4.  ],
           [24.  ],
           [30.  ],
           [49.  ],
           [65.  ],
           [48.  ],
           [47.  ],
           [23.  ],
           [25.  ],
           [35.  ],
           [58.  ],
           [55.  ],
           [54.  ],
           [25.  ],
           [16.  ],
           [18.  ],
           [24.  ],
           [47.  ],
           [34.  ],
           [30.  ],
           [24.  ],
           [45.  ],
           [22.  ],
           [24.  ],
           [50.  ],
           [17.  ],
           [48.  ],
           [39.  ],
           [53.  ],
           [36.  ],
           [39.  ],
           [39.  ],
           [36.  ],
           [18.  ],
           [24.  ],
           [52.  ],
           [49.  ],
           [40.  ],
           [ 4.  ],
           [24.  ],
           [61.  ],
           [21.  ],
           [80.  ],
           [32.  ],
           [24.  ],
           [48.  ],
           [56.  ],
           [58.  ],
           [47.  ],
           [31.  ],
           [36.  ],
           [27.  ],
           [15.  ],
           [31.  ],
           [49.  ],
           [24.  ],
           [18.  ],
           [35.  ],
           [42.  ],
           [24.  ],
           [48.  ],
           [19.  ],
           [24.  ],
           [27.  ],
           [27.  ],
           [29.  ],
           [35.  ],
           [36.  ],
           [21.  ],
           [70.  ],
           [19.  ],
           [ 6.  ],
           [33.  ],
           [36.  ],
           [51.  ],
           [24.  ],
           [43.  ],
           [17.  ],
           [29.  ],
           [24.  ],
           [49.  ],
           [11.  ],
           [39.  ],
           [33.  ],
           [52.  ],
           [27.  ],
           [39.  ],
           [16.  ],
           [24.  ],
           [48.  ],
           [31.  ],
           [47.  ],
           [24.  ],
           [56.  ],
           [19.  ],
           [24.  ]])




```python
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
```


```python
imp_median.fit_transform(titanicmd[['Age']])
```




    array([[38.  ],
           [35.  ],
           [54.  ],
           [35.5 ],
           [58.  ],
           [34.  ],
           [35.5 ],
           [19.  ],
           [49.  ],
           [65.  ],
           [45.  ],
           [29.  ],
           [25.  ],
           [35.5 ],
           [46.  ],
           [71.  ],
           [23.  ],
           [21.  ],
           [47.  ],
           [24.  ],
           [35.5 ],
           [54.  ],
           [19.  ],
           [37.  ],
           [24.  ],
           [36.5 ],
           [22.  ],
           [61.  ],
           [56.  ],
           [50.  ],
           [35.5 ],
           [ 3.  ],
           [44.  ],
           [58.  ],
           [ 2.  ],
           [40.  ],
           [31.  ],
           [32.  ],
           [38.  ],
           [35.5 ],
           [44.  ],
           [37.  ],
           [35.5 ],
           [62.  ],
           [30.  ],
           [52.  ],
           [40.  ],
           [35.5 ],
           [35.  ],
           [37.  ],
           [63.  ],
           [19.  ],
           [36.  ],
           [ 2.  ],
           [50.  ],
           [ 0.92],
           [17.  ],
           [30.  ],
           [24.  ],
           [18.  ],
           [31.  ],
           [40.  ],
           [35.5 ],
           [36.  ],
           [16.  ],
           [35.5 ],
           [38.  ],
           [29.  ],
           [41.  ],
           [45.  ],
           [35.5 ],
           [24.  ],
           [24.  ],
           [22.  ],
           [60.  ],
           [24.  ],
           [25.  ],
           [27.  ],
           [36.  ],
           [23.  ],
           [24.  ],
           [33.  ],
           [35.5 ],
           [28.  ],
           [50.  ],
           [14.  ],
           [64.  ],
           [ 4.  ],
           [35.5 ],
           [30.  ],
           [49.  ],
           [65.  ],
           [48.  ],
           [47.  ],
           [23.  ],
           [25.  ],
           [35.  ],
           [58.  ],
           [55.  ],
           [54.  ],
           [25.  ],
           [16.  ],
           [18.  ],
           [35.5 ],
           [47.  ],
           [34.  ],
           [30.  ],
           [35.5 ],
           [45.  ],
           [22.  ],
           [35.5 ],
           [50.  ],
           [17.  ],
           [48.  ],
           [39.  ],
           [53.  ],
           [36.  ],
           [39.  ],
           [39.  ],
           [36.  ],
           [18.  ],
           [35.5 ],
           [52.  ],
           [49.  ],
           [40.  ],
           [ 4.  ],
           [35.5 ],
           [61.  ],
           [21.  ],
           [80.  ],
           [32.  ],
           [24.  ],
           [48.  ],
           [56.  ],
           [58.  ],
           [47.  ],
           [31.  ],
           [36.  ],
           [27.  ],
           [15.  ],
           [31.  ],
           [49.  ],
           [35.5 ],
           [18.  ],
           [35.  ],
           [42.  ],
           [24.  ],
           [48.  ],
           [19.  ],
           [35.5 ],
           [27.  ],
           [27.  ],
           [29.  ],
           [35.  ],
           [36.  ],
           [21.  ],
           [70.  ],
           [19.  ],
           [ 6.  ],
           [33.  ],
           [36.  ],
           [51.  ],
           [35.5 ],
           [43.  ],
           [17.  ],
           [29.  ],
           [35.5 ],
           [49.  ],
           [11.  ],
           [39.  ],
           [33.  ],
           [52.  ],
           [27.  ],
           [39.  ],
           [16.  ],
           [35.5 ],
           [48.  ],
           [31.  ],
           [47.  ],
           [35.5 ],
           [56.  ],
           [19.  ],
           [35.5 ]])




```python
## Imputation SibSp
```


```python
imp_meanSp = SimpleImputer(missing_values=np.nan, strategy='mean')
```


```python
imp_meanSp.fit_transform(titanicmd[['SibSp']])
```




    array([[1.        ],
           [1.        ],
           [0.        ],
           [1.        ],
           [0.46111111],
           [0.        ],
           [0.        ],
           [3.        ],
           [1.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [3.        ],
           [0.46111111],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [2.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [1.        ],
           [1.        ],
           [2.        ],
           [1.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [1.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [1.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [2.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [3.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [1.        ],
           [1.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [1.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [1.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [1.        ],
           [1.        ],
           [2.        ],
           [0.        ],
           [1.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [1.        ],
           [1.        ],
           [0.        ],
           [2.        ],
           [0.46111111],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [1.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [2.        ],
           [1.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [1.        ],
           [1.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [1.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ]])




```python
imp_modeSp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
```


```python
imp_modeSp.fit_transform(titanicmd[['SibSp']])
```




    array([[1.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [3.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [3.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [2.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [1.],
           [1.],
           [2.],
           [1.],
           [1.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [1.],
           [0.],
           [1.],
           [0.],
           [1.],
           [1.],
           [0.],
           [0.],
           [2.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [1.],
           [3.],
           [0.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [1.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [1.],
           [1.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [1.],
           [0.],
           [0.],
           [1.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [1.],
           [1.],
           [2.],
           [0.],
           [1.],
           [1.],
           [0.],
           [0.],
           [1.],
           [1.],
           [1.],
           [0.],
           [2.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [1.],
           [2.],
           [1.],
           [1.],
           [0.],
           [0.],
           [1.],
           [1.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [1.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.]])




```python
imp_medianSp = SimpleImputer(missing_values=np.nan, strategy='median')
```


```python
imp_medianSp.fit_transform(titanicmd[['SibSp']])
```




    array([[1.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [3.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [3.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [2.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [1.],
           [1.],
           [2.],
           [1.],
           [1.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [1.],
           [0.],
           [1.],
           [0.],
           [1.],
           [1.],
           [0.],
           [0.],
           [2.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [1.],
           [3.],
           [0.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [1.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [1.],
           [1.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [1.],
           [0.],
           [0.],
           [1.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [1.],
           [1.],
           [2.],
           [0.],
           [1.],
           [1.],
           [0.],
           [0.],
           [1.],
           [1.],
           [1.],
           [0.],
           [2.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [1.],
           [2.],
           [1.],
           [1.],
           [0.],
           [0.],
           [1.],
           [1.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [1.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.]])




```python
##imputation Parch
```


```python
imp_meanP = SimpleImputer(missing_values=np.nan, strategy='mean')
```


```python
imp_meanP.fit_transform(titanicmd[['Parch']])
```




    array([[0.       ],
           [0.       ],
           [0.       ],
           [0.4619883],
           [0.       ],
           [0.       ],
           [0.       ],
           [2.       ],
           [0.4619883],
           [1.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [2.       ],
           [0.       ],
           [0.       ],
           [1.       ],
           [1.       ],
           [0.       ],
           [1.       ],
           [0.       ],
           [1.       ],
           [2.       ],
           [0.4619883],
           [0.       ],
           [2.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [1.       ],
           [1.       ],
           [0.       ],
           [0.       ],
           [1.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.4619883],
           [1.       ],
           [0.       ],
           [0.       ],
           [1.       ],
           [0.       ],
           [1.       ],
           [0.       ],
           [1.       ],
           [0.       ],
           [0.4619883],
           [0.       ],
           [2.       ],
           [1.       ],
           [2.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [2.       ],
           [2.       ],
           [1.       ],
           [0.       ],
           [0.       ],
           [1.       ],
           [0.       ],
           [1.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [1.       ],
           [0.4619883],
           [0.       ],
           [1.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [2.       ],
           [2.       ],
           [0.       ],
           [2.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [2.       ],
           [4.       ],
           [2.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [2.       ],
           [0.       ],
           [0.4619883],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [1.       ],
           [0.       ],
           [2.       ],
           [0.4619883],
           [0.       ],
           [2.       ],
           [0.       ],
           [1.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [1.       ],
           [0.       ],
           [2.       ],
           [1.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [1.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.4619883],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [2.       ],
           [0.       ],
           [0.       ],
           [1.       ],
           [0.       ],
           [1.       ],
           [0.       ],
           [1.       ],
           [0.       ],
           [0.       ],
           [0.4619883],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [2.       ],
           [1.       ],
           [0.       ],
           [1.       ],
           [0.       ],
           [2.       ],
           [0.       ],
           [0.       ],
           [0.4619883],
           [0.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [2.       ],
           [0.       ],
           [0.       ],
           [1.       ],
           [1.       ],
           [1.       ],
           [1.       ],
           [0.       ],
           [0.       ],
           [0.       ],
           [1.       ],
           [0.       ],
           [0.4619883],
           [0.       ],
           [0.       ]])




```python
imp_modeP = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
```


```python
imp_modeP.fit_transform(titanicmd[['Parch']])
```




    array([[0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [2.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [2.],
           [0.],
           [0.],
           [1.],
           [1.],
           [0.],
           [1.],
           [0.],
           [1.],
           [2.],
           [0.],
           [0.],
           [2.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [1.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [2.],
           [1.],
           [2.],
           [0.],
           [0.],
           [0.],
           [2.],
           [2.],
           [1.],
           [0.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [2.],
           [2.],
           [0.],
           [2.],
           [0.],
           [0.],
           [0.],
           [0.],
           [2.],
           [4.],
           [2.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [2.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [2.],
           [0.],
           [0.],
           [2.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [2.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [2.],
           [0.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [2.],
           [1.],
           [0.],
           [1.],
           [0.],
           [2.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [2.],
           [0.],
           [0.],
           [1.],
           [1.],
           [1.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.]])




```python
imp_medianP = SimpleImputer(missing_values=np.nan, strategy='median')
```


```python
imp_medianP.fit_transform(titanicmd[['Parch']])
```




    array([[0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [2.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [2.],
           [0.],
           [0.],
           [1.],
           [1.],
           [0.],
           [1.],
           [0.],
           [1.],
           [2.],
           [0.],
           [0.],
           [2.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [1.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [2.],
           [1.],
           [2.],
           [0.],
           [0.],
           [0.],
           [2.],
           [2.],
           [1.],
           [0.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [2.],
           [2.],
           [0.],
           [2.],
           [0.],
           [0.],
           [0.],
           [0.],
           [2.],
           [4.],
           [2.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [2.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [2.],
           [0.],
           [0.],
           [2.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [2.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [2.],
           [0.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [2.],
           [1.],
           [0.],
           [1.],
           [0.],
           [2.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [2.],
           [0.],
           [0.],
           [1.],
           [1.],
           [1.],
           [1.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.]])




```python
##ImputationFare
```


```python
imp_meanF = SimpleImputer(missing_values=np.nan, strategy='mean')
```


```python
imp_meanF.fit_transform(titanicmd[['Fare']])
```




    array([[ 71.2833    ],
           [ 53.1       ],
           [ 51.8625    ],
           [ 16.7       ],
           [ 26.55      ],
           [ 13.        ],
           [ 35.5       ],
           [263.        ],
           [ 76.7292    ],
           [ 61.9792    ],
           [ 83.475     ],
           [ 10.5       ],
           [ 78.95919086],
           [263.        ],
           [ 61.175     ],
           [ 34.6542    ],
           [ 63.3583    ],
           [ 77.2875    ],
           [ 52.        ],
           [247.5208    ],
           [ 13.        ],
           [ 77.2875    ],
           [ 26.2833    ],
           [ 53.1       ],
           [ 79.2       ],
           [ 26.        ],
           [ 66.6       ],
           [ 33.5       ],
           [ 30.6958    ],
           [ 28.7125    ],
           [ 39.        ],
           [ 26.        ],
           [ 27.7208    ],
           [146.5208    ],
           [ 10.4625    ],
           [ 31.        ],
           [113.275     ],
           [ 78.95919086],
           [ 90.        ],
           [ 83.475     ],
           [ 90.        ],
           [ 52.5542    ],
           [ 10.4625    ],
           [ 26.55      ],
           [ 78.95919086],
           [ 79.65      ],
           [  0.        ],
           [153.4625    ],
           [135.6333    ],
           [ 29.7       ],
           [ 77.9583    ],
           [ 91.0792    ],
           [ 12.875     ],
           [151.55      ],
           [247.5208    ],
           [151.55      ],
           [108.9       ],
           [ 56.9292    ],
           [ 83.1583    ],
           [262.375     ],
           [164.8667    ],
           [134.5       ],
           [135.6333    ],
           [ 13.        ],
           [ 57.9792    ],
           [ 28.5       ],
           [153.4625    ],
           [ 66.6       ],
           [134.5       ],
           [ 35.5       ],
           [ 26.        ],
           [263.        ],
           [ 13.        ],
           [ 55.        ],
           [ 75.25      ],
           [ 69.3       ],
           [ 55.4417    ],
           [211.5       ],
           [120.        ],
           [113.275     ],
           [ 16.7       ],
           [ 90.        ],
           [  8.05      ],
           [ 26.55      ],
           [ 55.9       ],
           [120.        ],
           [263.        ],
           [ 81.8583    ],
           [ 30.5       ],
           [ 27.75      ],
           [ 89.1042    ],
           [ 26.55      ],
           [ 26.55      ],
           [ 38.5       ],
           [ 13.7917    ],
           [ 91.0792    ],
           [ 90.        ],
           [ 29.7       ],
           [ 30.5       ],
           [ 78.2667    ],
           [151.55      ],
           [ 86.5       ],
           [108.9       ],
           [ 26.2875    ],
           [ 34.0208    ],
           [ 10.5       ],
           [ 93.5       ],
           [ 57.9792    ],
           [ 26.55      ],
           [ 49.5       ],
           [ 71.        ],
           [106.425     ],
           [110.8833    ],
           [ 39.6       ],
           [ 79.65      ],
           [ 51.4792    ],
           [ 26.3875    ],
           [ 55.9       ],
           [110.8833    ],
           [ 40.125     ],
           [ 79.65      ],
           [ 78.95919086],
           [ 78.2667    ],
           [ 56.9292    ],
           [153.4625    ],
           [ 39.        ],
           [ 52.5542    ],
           [ 32.3208    ],
           [ 77.9583    ],
           [ 30.        ],
           [ 30.5       ],
           [ 69.3       ],
           [ 76.7292    ],
           [ 35.5       ],
           [113.275     ],
           [ 25.5875    ],
           [ 52.        ],
           [512.3292    ],
           [ 76.7292    ],
           [211.3375    ],
           [ 57.        ],
           [110.8833    ],
           [  7.65      ],
           [227.525     ],
           [ 26.2875    ],
           [ 26.2875    ],
           [ 49.5042    ],
           [ 52.        ],
           [  7.65      ],
           [227.525     ],
           [ 10.5       ],
           [ 53.1       ],
           [211.3375    ],
           [512.3292    ],
           [ 78.85      ],
           [262.375     ],
           [ 71.        ],
           [ 53.1       ],
           [ 12.475     ],
           [ 86.5       ],
           [120.        ],
           [ 77.9583    ],
           [ 78.95919086],
           [ 78.95919086],
           [ 78.95919086],
           [ 30.        ],
           [ 79.2       ],
           [ 25.9292    ],
           [120.        ],
           [  0.        ],
           [ 53.1       ],
           [ 93.5       ],
           [ 12.475     ],
           [ 83.1583    ],
           [ 39.4       ],
           [ 26.55      ],
           [ 25.9292    ],
           [ 50.4958    ],
           [ 78.95919086],
           [  5.        ],
           [ 83.1583    ],
           [ 30.        ],
           [ 30.        ]])




```python
imp_medianF = SimpleImputer(missing_values=np.nan, strategy='median')
```


```python
imp_medianF.fit_transform(titanicmd[['Fare']])
```




    array([[ 71.2833],
           [ 53.1   ],
           [ 51.8625],
           [ 16.7   ],
           [ 26.55  ],
           [ 13.    ],
           [ 35.5   ],
           [263.    ],
           [ 76.7292],
           [ 61.9792],
           [ 83.475 ],
           [ 10.5   ],
           [ 56.9292],
           [263.    ],
           [ 61.175 ],
           [ 34.6542],
           [ 63.3583],
           [ 77.2875],
           [ 52.    ],
           [247.5208],
           [ 13.    ],
           [ 77.2875],
           [ 26.2833],
           [ 53.1   ],
           [ 79.2   ],
           [ 26.    ],
           [ 66.6   ],
           [ 33.5   ],
           [ 30.6958],
           [ 28.7125],
           [ 39.    ],
           [ 26.    ],
           [ 27.7208],
           [146.5208],
           [ 10.4625],
           [ 31.    ],
           [113.275 ],
           [ 56.9292],
           [ 90.    ],
           [ 83.475 ],
           [ 90.    ],
           [ 52.5542],
           [ 10.4625],
           [ 26.55  ],
           [ 56.9292],
           [ 79.65  ],
           [  0.    ],
           [153.4625],
           [135.6333],
           [ 29.7   ],
           [ 77.9583],
           [ 91.0792],
           [ 12.875 ],
           [151.55  ],
           [247.5208],
           [151.55  ],
           [108.9   ],
           [ 56.9292],
           [ 83.1583],
           [262.375 ],
           [164.8667],
           [134.5   ],
           [135.6333],
           [ 13.    ],
           [ 57.9792],
           [ 28.5   ],
           [153.4625],
           [ 66.6   ],
           [134.5   ],
           [ 35.5   ],
           [ 26.    ],
           [263.    ],
           [ 13.    ],
           [ 55.    ],
           [ 75.25  ],
           [ 69.3   ],
           [ 55.4417],
           [211.5   ],
           [120.    ],
           [113.275 ],
           [ 16.7   ],
           [ 90.    ],
           [  8.05  ],
           [ 26.55  ],
           [ 55.9   ],
           [120.    ],
           [263.    ],
           [ 81.8583],
           [ 30.5   ],
           [ 27.75  ],
           [ 89.1042],
           [ 26.55  ],
           [ 26.55  ],
           [ 38.5   ],
           [ 13.7917],
           [ 91.0792],
           [ 90.    ],
           [ 29.7   ],
           [ 30.5   ],
           [ 78.2667],
           [151.55  ],
           [ 86.5   ],
           [108.9   ],
           [ 26.2875],
           [ 34.0208],
           [ 10.5   ],
           [ 93.5   ],
           [ 57.9792],
           [ 26.55  ],
           [ 49.5   ],
           [ 71.    ],
           [106.425 ],
           [110.8833],
           [ 39.6   ],
           [ 79.65  ],
           [ 51.4792],
           [ 26.3875],
           [ 55.9   ],
           [110.8833],
           [ 40.125 ],
           [ 79.65  ],
           [ 56.9292],
           [ 78.2667],
           [ 56.9292],
           [153.4625],
           [ 39.    ],
           [ 52.5542],
           [ 32.3208],
           [ 77.9583],
           [ 30.    ],
           [ 30.5   ],
           [ 69.3   ],
           [ 76.7292],
           [ 35.5   ],
           [113.275 ],
           [ 25.5875],
           [ 52.    ],
           [512.3292],
           [ 76.7292],
           [211.3375],
           [ 57.    ],
           [110.8833],
           [  7.65  ],
           [227.525 ],
           [ 26.2875],
           [ 26.2875],
           [ 49.5042],
           [ 52.    ],
           [  7.65  ],
           [227.525 ],
           [ 10.5   ],
           [ 53.1   ],
           [211.3375],
           [512.3292],
           [ 78.85  ],
           [262.375 ],
           [ 71.    ],
           [ 53.1   ],
           [ 12.475 ],
           [ 86.5   ],
           [120.    ],
           [ 77.9583],
           [ 56.9292],
           [ 56.9292],
           [ 56.9292],
           [ 30.    ],
           [ 79.2   ],
           [ 25.9292],
           [120.    ],
           [  0.    ],
           [ 53.1   ],
           [ 93.5   ],
           [ 12.475 ],
           [ 83.1583],
           [ 39.4   ],
           [ 26.55  ],
           [ 25.9292],
           [ 50.4958],
           [ 56.9292],
           [  5.    ],
           [ 83.1583],
           [ 30.    ],
           [ 30.    ]])




```python
###EmbarkedImputation
```


```python
imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
```


```python
imp_modeE.fit_transform(titanicmd[['Embarked']])
```




    array([['C'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['C'],
           ['C'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['C'],
           ['C'],
           ['S'],
           ['S'],
           ['C'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['C'],
           ['S'],
           ['S'],
           ['S'],
           ['C'],
           ['C'],
           ['S'],
           ['S'],
           ['C'],
           ['C'],
           ['S'],
           ['C'],
           ['S'],
           ['C'],
           ['S'],
           ['S'],
           ['Q'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['C'],
           ['S'],
           ['C'],
           ['C'],
           ['S'],
           ['C'],
           ['S'],
           ['C'],
           ['C'],
           ['C'],
           ['C'],
           ['S'],
           ['C'],
           ['C'],
           ['S'],
           ['C'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['C'],
           ['C'],
           ['C'],
           ['C'],
           ['S'],
           ['C'],
           ['S'],
           ['Q'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['C'],
           ['C'],
           ['S'],
           ['S'],
           ['S'],
           ['C'],
           ['C'],
           ['S'],
           ['C'],
           ['S'],
           ['C'],
           ['S'],
           ['S'],
           ['C'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['C'],
           ['S'],
           ['C'],
           ['C'],
           ['C'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['C'],
           ['C'],
           ['S'],
           ['S'],
           ['C'],
           ['C'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['C'],
           ['C'],
           ['C'],
           ['C'],
           ['C'],
           ['S'],
           ['S'],
           ['C'],
           ['S'],
           ['S'],
           ['S'],
           ['C'],
           ['S'],
           ['C'],
           ['S'],
           ['S'],
           ['C'],
           ['S'],
           ['S'],
           ['C'],
           ['S'],
           ['S'],
           ['S'],
           ['C'],
           ['S'],
           ['C'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['C'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['C'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['S'],
           ['C']], dtype=object)




```python
from sklearn.linear_model import LinearRegression
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
##Age Linear Model
```


```python
print('SibSp avg without prediction:', round(titanicmd['SibSp'].mean(),2))
sns.distplot(titanicmd['SibSp'])
```

    SibSp avg without prediction: 0.46
    

    C:\Users\LuisP\anaconda3\envs\Lab8Wrangling\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    




    <AxesSubplot:xlabel='SibSp', ylabel='Density'>




    
![png](output_45_3.png)
    



```python
titanicmd.corr()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PassengerId</th>
      <td>1.000000</td>
      <td>0.148495</td>
      <td>-0.089136</td>
      <td>-0.048190</td>
      <td>-0.088806</td>
      <td>-0.062083</td>
      <td>0.022261</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>0.148495</td>
      <td>1.000000</td>
      <td>-0.034542</td>
      <td>-0.257703</td>
      <td>0.113987</td>
      <td>-0.003365</td>
      <td>0.119311</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>-0.089136</td>
      <td>-0.034542</td>
      <td>1.000000</td>
      <td>-0.297872</td>
      <td>-0.102294</td>
      <td>0.041969</td>
      <td>-0.304438</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>-0.048190</td>
      <td>-0.257703</td>
      <td>-0.297872</td>
      <td>1.000000</td>
      <td>-0.087951</td>
      <td>-0.279548</td>
      <td>-0.130979</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>-0.088806</td>
      <td>0.113987</td>
      <td>-0.102294</td>
      <td>-0.087951</td>
      <td>1.000000</td>
      <td>0.255152</td>
      <td>0.299061</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>-0.062083</td>
      <td>-0.003365</td>
      <td>0.041969</td>
      <td>-0.279548</td>
      <td>0.255152</td>
      <td>1.000000</td>
      <td>0.381445</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0.022261</td>
      <td>0.119311</td>
      <td>-0.304438</td>
      <td>-0.130979</td>
      <td>0.299061</td>
      <td>0.381445</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Age STandar Deviation
```


```python
f = 2
xl = titanicmd['Age'].mean() - (titanicmd['Age'].std() * f)
xu = titanicmd['Age'].mean() + (titanicmd['Age'].std() * f)
print('Lower value:', xl)
print('Upper value:', xu)
```

    Lower value: 4.410816187965786
    Upper value: 66.97424710317347
    


```python
sns.scatterplot(x = titanicmd['PassengerId'], y = titanicmd['Age'])
sns.lineplot(x = titanicmd['PassengerId'], y = xl, color = 'green')
sns.lineplot(x = titanicmd['PassengerId'], y = xu, color = 'orange')
```




    <AxesSubplot:xlabel='PassengerId', ylabel='Age'>




    
![png](output_49_1.png)
    



```python
tmdA_sd = titanicmd[(titanicmd['Age']>=xl) & (titanicmd['Age']<=xu)]
```


```python
tmdA_sd[['SibSp']].describe()
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
      <th>SibSp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>147.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.455782</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.621876</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanicmd['Age_sd'] = np.where(
    titanicmd['Age']<xl,
    xl,
    np.where(
        titanicmd['Age']>xu,
        xu,
        titanicmd['SibSp']
    )
)
```


```python
titanicmd[['Age', 'Age_sd']].describe()
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
      <th>Age</th>
      <th>Age_sd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>158.000000</td>
      <td>180.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>35.692532</td>
      <td>1.691042</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15.640858</td>
      <td>8.575543</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.920000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>35.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.000000</td>
      <td>66.974247</td>
    </tr>
  </tbody>
</table>
</div>




```python
## SibSp STandar Deviation
```


```python
f = 2
xl = titanicmd['SibSp'].mean() - (titanicmd['SibSp'].std() * f)
xu = titanicmd['SibSp'].mean() + (titanicmd['SibSp'].std() * f)
print('Lower value:', xl)
print('Upper value:', xu)
```

    Lower value: -0.8311328579137286
    Upper value: 1.7533550801359508
    


```python
sns.scatterplot(x = titanicmd['PassengerId'], y = titanicmd['SibSp'])
sns.lineplot(x = titanicmd['PassengerId'], y = xl, color = 'green')
sns.lineplot(x = titanicmd['PassengerId'], y = xu, color = 'orange')
```




    <AxesSubplot:xlabel='PassengerId', ylabel='SibSp'>




    
![png](output_56_1.png)
    



```python
tmd_sd = titanicmd[(titanicmd['SibSp']>=xl) & (titanicmd['SibSp']<=xu)]
```


```python
tmd_sd[['SibSp']].describe()
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
      <th>SibSp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>171.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.362573</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.482155</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanicmd['SibSp_sd'] = np.where(
    titanicmd['SibSp']<xl,
    xl,
    np.where(
        titanicmd['SibSp']>xu,
        xu,
        titanicmd['SibSp']
    )
)
```


```python
titanicmd[['SibSp', 'SibSp_sd']].describe()
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
      <th>SibSp</th>
      <th>SibSp_sd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>180.000000</td>
      <td>180.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.461111</td>
      <td>0.432112</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.646122</td>
      <td>0.559621</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>1.753355</td>
    </tr>
  </tbody>
</table>
</div>




```python
### Sd EMbarked
```


```python
f = 2
xl = titanicmd['SibSp'].mean() - (titanicmd['SibSp'].std() * f)
xu = titanicmd['SibSp'].mean() + (titanicmd['SibSp'].std() * f)
print('Lower value:', xl)
print('Upper value:', xu)
```

    Lower value: -0.8311328579137286
    Upper value: 1.7533550801359508
    


```python
sns.scatterplot(x = titanicmd['PassengerId'], y = titanicmd['Embarked'])
sns.lineplot(x = titanicmd['PassengerId'], y = xl, color = 'green')
sns.lineplot(x = titanicmd['PassengerId'], y = xu, color = 'orange')
```




    <AxesSubplot:xlabel='PassengerId', ylabel='Embarked'>




    
![png](output_63_1.png)
    



```python
titanic
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>PP 9549</td>
      <td>16.7000</td>
      <td>G6</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>Bonnell, Miss. Elizabeth</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>113783</td>
      <td>26.5500</td>
      <td>C103</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>178</th>
      <td>872</td>
      <td>1</td>
      <td>1</td>
      <td>Beckwith, Mrs. Richard Leonard (Sallie Monypeny)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>1</td>
      <td>11751</td>
      <td>52.5542</td>
      <td>D35</td>
      <td>S</td>
    </tr>
    <tr>
      <th>179</th>
      <td>873</td>
      <td>0</td>
      <td>1</td>
      <td>Carlsson, Mr. Frans Olof</td>
      <td>male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>695</td>
      <td>5.0000</td>
      <td>B51 B53 B55</td>
      <td>S</td>
    </tr>
    <tr>
      <th>180</th>
      <td>880</td>
      <td>1</td>
      <td>1</td>
      <td>Potter, Mrs. Thomas Jr (Lily Alexenia Wilson)</td>
      <td>female</td>
      <td>56.0</td>
      <td>0</td>
      <td>1</td>
      <td>11767</td>
      <td>83.1583</td>
      <td>C50</td>
      <td>C</td>
    </tr>
    <tr>
      <th>181</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>182</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
<p>183 rows × 12 columns</p>
</div>




```python
##Standarization Z value
```


```python
from sklearn.preprocessing import StandardScaler
```


```python
scaler = StandardScaler()
tmd_z = titanicmd.copy()
for col in tmd_z.select_dtypes(include=['float', 'int']).columns:
    tmd_z[col+'_z'] = scaler.fit_transform(tmd_z[[col]])
```


```python
tmd_z.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>...</th>
      <th>SibSp_sd</th>
      <th>PassengerId_z</th>
      <th>Survived_z</th>
      <th>Pclass_z</th>
      <th>Age_z</th>
      <th>SibSp_z</th>
      <th>Parch_z</th>
      <th>Fare_z</th>
      <th>Age_sd_z</th>
      <th>SibSp_sd_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>NaN</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>...</td>
      <td>1.0</td>
      <td>-1.840135</td>
      <td>0.698430</td>
      <td>-0.372256</td>
      <td>0.147997</td>
      <td>0.836362</td>
      <td>-0.614977</td>
      <td>-0.099939</td>
      <td>-0.080808</td>
      <td>1.017602</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>...</td>
      <td>1.0</td>
      <td>-1.832017</td>
      <td>0.698430</td>
      <td>-0.372256</td>
      <td>-0.044418</td>
      <td>0.836362</td>
      <td>-0.614977</td>
      <td>-0.336682</td>
      <td>-0.080808</td>
      <td>1.017602</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>...</td>
      <td>0.0</td>
      <td>-1.819841</td>
      <td>-1.431782</td>
      <td>-0.372256</td>
      <td>1.174212</td>
      <td>-0.715650</td>
      <td>-0.614977</td>
      <td>-0.352794</td>
      <td>-0.197744</td>
      <td>-0.774305</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>PP 9549</td>
      <td>16.7000</td>
      <td>...</td>
      <td>1.0</td>
      <td>-1.803606</td>
      <td>0.698430</td>
      <td>3.520480</td>
      <td>NaN</td>
      <td>0.836362</td>
      <td>NaN</td>
      <td>-0.810604</td>
      <td>-0.080808</td>
      <td>1.017602</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>Bonnell, Miss. Elizabeth</td>
      <td>female</td>
      <td>58.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>113783</td>
      <td>26.5500</td>
      <td>...</td>
      <td>NaN</td>
      <td>-1.799547</td>
      <td>0.698430</td>
      <td>-0.372256</td>
      <td>1.430765</td>
      <td>NaN</td>
      <td>-0.614977</td>
      <td>-0.682359</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
tmd_z.describe()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Age_sd</th>
      <th>SibSp_sd</th>
      <th>PassengerId_z</th>
      <th>Survived_z</th>
      <th>Pclass_z</th>
      <th>Age_z</th>
      <th>SibSp_z</th>
      <th>Parch_z</th>
      <th>Fare_z</th>
      <th>Age_sd_z</th>
      <th>SibSp_sd_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>158.000000</td>
      <td>180.000000</td>
      <td>171.000000</td>
      <td>175.000000</td>
      <td>180.000000</td>
      <td>180.000000</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.580000e+02</td>
      <td>1.800000e+02</td>
      <td>1.710000e+02</td>
      <td>175.000000</td>
      <td>1.800000e+02</td>
      <td>1.800000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>455.366120</td>
      <td>0.672131</td>
      <td>1.191257</td>
      <td>35.692532</td>
      <td>0.461111</td>
      <td>0.461988</td>
      <td>78.959191</td>
      <td>1.691042</td>
      <td>0.432112</td>
      <td>-2.499519e-16</td>
      <td>2.062709e-17</td>
      <td>4.307423e-17</td>
      <td>-2.009644e-16</td>
      <td>1.850372e-17</td>
      <td>3.116416e-17</td>
      <td>0.000000</td>
      <td>-4.132497e-17</td>
      <td>1.319932e-16</td>
    </tr>
    <tr>
      <th>std</th>
      <td>247.052476</td>
      <td>0.470725</td>
      <td>0.515187</td>
      <td>15.640858</td>
      <td>0.646122</td>
      <td>0.753435</td>
      <td>77.026328</td>
      <td>8.575543</td>
      <td>0.559621</td>
      <td>1.002743e+00</td>
      <td>1.002743e+00</td>
      <td>1.002743e+00</td>
      <td>1.003180e+00</td>
      <td>1.002789e+00</td>
      <td>1.002937e+00</td>
      <td>1.002869</td>
      <td>1.002789e+00</td>
      <td>1.002789e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.920000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.840135e+00</td>
      <td>-1.431782e+00</td>
      <td>-3.722562e-01</td>
      <td>-2.230255e+00</td>
      <td>-7.156502e-01</td>
      <td>-6.149769e-01</td>
      <td>-1.028035</td>
      <td>-1.977437e-01</td>
      <td>-7.743049e-01</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>263.500000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>24.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>29.700000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-7.787516e-01</td>
      <td>-1.431782e+00</td>
      <td>-3.722562e-01</td>
      <td>-7.499403e-01</td>
      <td>-7.156502e-01</td>
      <td>-6.149769e-01</td>
      <td>-0.641346</td>
      <td>-1.977437e-01</td>
      <td>-7.743049e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>457.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>35.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>56.929200</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.631637e-03</td>
      <td>6.984303e-01</td>
      <td>-3.722562e-01</td>
      <td>-1.234867e-02</td>
      <td>-7.156502e-01</td>
      <td>-6.149769e-01</td>
      <td>-0.286827</td>
      <td>-1.977437e-01</td>
      <td>-7.743049e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>676.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>48.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>90.539600</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>8.955150e-01</td>
      <td>6.984303e-01</td>
      <td>-3.722562e-01</td>
      <td>7.893814e-01</td>
      <td>8.363623e-01</td>
      <td>7.161756e-01</td>
      <td>0.150775</td>
      <td>-8.080772e-02</td>
      <td>1.017602e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>890.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>512.329200</td>
      <td>66.974247</td>
      <td>1.753355</td>
      <td>1.764104e+00</td>
      <td>6.984303e-01</td>
      <td>3.520480e+00</td>
      <td>2.841810e+00</td>
      <td>3.940387e+00</td>
      <td>4.709633e+00</td>
      <td>5.642402</td>
      <td>7.633955e+00</td>
      <td>2.367544e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
##Min Max Scaler
```


```python
from sklearn.preprocessing import MinMaxScaler
```


```python
scaler = MinMaxScaler()
tmd_norm = titanicmd.copy()
for col in tmd_norm.select_dtypes(include=['float', 'int']).columns:
    tmd_norm[col+'_norm'] = scaler.fit_transform(tmd_norm[[col]])
```


```python
tmd_norm.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>...</th>
      <th>SibSp_sd</th>
      <th>PassengerId_norm</th>
      <th>Survived_norm</th>
      <th>Pclass_norm</th>
      <th>Age_norm</th>
      <th>SibSp_norm</th>
      <th>Parch_norm</th>
      <th>Fare_norm</th>
      <th>Age_sd_norm</th>
      <th>SibSp_sd_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>NaN</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.468892</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.139136</td>
      <td>0.014931</td>
      <td>0.570335</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.002252</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.430956</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.103644</td>
      <td>0.014931</td>
      <td>0.570335</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.005631</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.671219</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.101229</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>PP 9549</td>
      <td>16.7000</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.010135</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.333333</td>
      <td>NaN</td>
      <td>0.032596</td>
      <td>0.014931</td>
      <td>0.570335</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>Bonnell, Miss. Elizabeth</td>
      <td>female</td>
      <td>58.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>113783</td>
      <td>26.5500</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.011261</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.721801</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.051822</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
tmd_norm.describe()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Age_sd</th>
      <th>SibSp_sd</th>
      <th>PassengerId_norm</th>
      <th>Survived_norm</th>
      <th>Pclass_norm</th>
      <th>Age_norm</th>
      <th>SibSp_norm</th>
      <th>Parch_norm</th>
      <th>Fare_norm</th>
      <th>Age_sd_norm</th>
      <th>SibSp_sd_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>158.000000</td>
      <td>180.000000</td>
      <td>171.000000</td>
      <td>175.000000</td>
      <td>180.000000</td>
      <td>180.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>158.000000</td>
      <td>180.000000</td>
      <td>171.000000</td>
      <td>175.000000</td>
      <td>180.000000</td>
      <td>180.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>455.366120</td>
      <td>0.672131</td>
      <td>1.191257</td>
      <td>35.692532</td>
      <td>0.461111</td>
      <td>0.461988</td>
      <td>78.959191</td>
      <td>1.691042</td>
      <td>0.432112</td>
      <td>0.510547</td>
      <td>0.672131</td>
      <td>0.095628</td>
      <td>0.439713</td>
      <td>0.153704</td>
      <td>0.115497</td>
      <td>0.154118</td>
      <td>0.025249</td>
      <td>0.246449</td>
    </tr>
    <tr>
      <th>std</th>
      <td>247.052476</td>
      <td>0.470725</td>
      <td>0.515187</td>
      <td>15.640858</td>
      <td>0.646122</td>
      <td>0.753435</td>
      <td>77.026328</td>
      <td>8.575543</td>
      <td>0.559621</td>
      <td>0.278212</td>
      <td>0.470725</td>
      <td>0.257593</td>
      <td>0.197785</td>
      <td>0.215374</td>
      <td>0.188359</td>
      <td>0.150345</td>
      <td>0.128042</td>
      <td>0.319172</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.920000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>263.500000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>24.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>29.700000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.294482</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.291856</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.057971</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>457.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>35.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>56.929200</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.512387</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.437279</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.111118</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>676.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>48.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>90.539600</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.759009</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.595346</td>
      <td>0.333333</td>
      <td>0.250000</td>
      <td>0.176722</td>
      <td>0.014931</td>
      <td>0.570335</td>
    </tr>
    <tr>
      <th>max</th>
      <td>890.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>512.329200</td>
      <td>66.974247</td>
      <td>1.753355</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


