import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
from sklearn.model_selection import train_test_split

df = pd.read_csv('USA_Housing.csv')
# df.head()
# df.info()
# sns.pairplot(df)
# sns.distplot(df['Price'])
# sns.heatmap(df.corr(),annot=True)
df.columns
# Features, the list of columns
x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms', 'Area Population'
        ]]

# The label or target variables
y = df['Price']
# The train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train, y_train)

print(lm.intercept_)

lm.coef_

cdf = pd.DataFrame(lm.coef_, x.columns, columns=['Coeff'])
print(cdf)

# Prediction
prediction = lm.predict(x_test)

plt.scatter(y_test, prediction)
sns.distplot((y_test - prediction))
metrics.mean_absolute_error(y_test, prediction)
metrics.mean_squared_error()
np.sqrt(metrics.mean_squared_error())
