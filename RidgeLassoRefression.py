from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

#Sample data (e.g., house size vs. house price)
X= np.array([[1400], [1600], [1700], [1875], [1100], [1550], [2350], [2450], [1425], [1700] ])
y= np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])

#Split the data into training and testing  sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Ridge Regression
ridge_model = Ridge(alpha=1.0) # a;pha controls the regularisation stregth
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict