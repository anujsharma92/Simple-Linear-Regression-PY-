#simple linear regression

import pandas as pd
import numpy as np
#importing Data Set
weight=pd.read_csv("C:\\Users\\Anuj Kumar\\Desktop\\data science\\data set\\Simple linear regression\\calories_consumed.csv")

weight.describe()

weight.columns=["Weight_gain","Cal_cons"]

from matplotlib import pyplot as plt
#execute line by line for graph
plt.bar(height = weight.Weight_gain, x = np.arange(0, 14, 1)) # weight gain bar graph
plt.bar(height=weight.Cal_cons, x  = np.arange(0, 14 , 1),color="r") # cal consumed bar graph

plt.hist(weight.Weight_gain)# weight gain Histogram
plt.hist(weight.Cal_cons)# cal consumed Histogram

plt.boxplot(weight.Weight_gain) #boxplot weight gain
plt.boxplot(weight.Cal_cons) #box plot cal

#Bivariate Analysis
#scatter plot
plt.scatter(x=weight['Cal_cons'],y=weight['Weight_gain'] ,color='r')

import statsmodels.formula.api as smf

#separating i/p and o/p

y=weight['Weight_gain']
x=weight['Cal_cons']
# Simple Linear Regression
model_1=smf.ols('y~x',data=weight).fit()
model_1.summary()
#R squred value is 0.897

pred_1=model_1.predict(pd.DataFrame(x))
print(pred_1)

#plot between predicted and actual
plt.scatter(x,y)
plt.plot(x,pred_1,'r')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error
res1=y-pred_1
rmse1=res1*res1
rmse1=np.mean(rmse1)
rmse1=np.sqrt(rmse1)
rmse1
#103.30250194726932

# Simple Linear Regression model 2 
plt.scatter(x = np.log(x), y = y, color = 'brown')
model_2=smf.ols('y~np.log(x)',data=weight).fit()
model_2.summary()
#R squred value is 0.792

pred_2=model_2.predict(pd.DataFrame(x))
print(pred_2)

#plot between predicted and actual
plt.scatter(x,y)
plt.plot(x,pred_2,'r')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error
res2=y-pred_2
rmse2=res2*res2
rmse2=np.mean(rmse2)
rmse2=np.sqrt(rmse2)
rmse2
# 141.00538169425104

# Simple Linear Regression model 3 exponential
plt.scatter(x = x, y = np.log(y), color = 'brown')
model_3=smf.ols('np.log(y)~x',data=weight).fit()
model_3.summary()
#R squred value is 0.792

pred_3=model_3.predict(pd.DataFrame(x))
pred_3=np.exp(pred_3)
pred_3

#plot between predicted and actual
plt.scatter(x,y)
plt.plot(x,pred_3,'r')
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error
res3=y-pred_3
rmse3=res3*res3
rmse3=np.mean(rmse3)
rmse3=np.sqrt(rmse3)
rmse3
#118.0451572011805

#### Polynomial transformation

model4 = smf.ols('np.log(y) ~ x + I(x*x)', data = weight).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(x))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = weight.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)


#plot between predicted and actual
plt.scatter(x, np.log(y))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = y - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4
#117.41450013144163

# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

#CHOOSING BEST MODEL

from sklearn.model_selection import train_test_split

train, test = train_test_split(weight, test_size = 0.06)
#applying model on on train data
final_model=smf.ols('Weight_gain~Cal_cons',data=train).fit()
final_model.summary()
#prediction on test data
test_pred = final_model.predict(pd.DataFrame(test))
test_pred

# Model Evaluation on Test data
test_res = test.Weight_gain - test_pred
test_res
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_mse
test_rmse = np.sqrt(test_mse)
test_rmse
# 124.3517411333122

#here we can see error value is being very large so we require more data

