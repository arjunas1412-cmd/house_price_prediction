import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error 
from sklearn.model_selection import cross_val_score

#loading data
housing = pd.read_csv("housing.csv")

#Income Category Creation
housing['income_category'] = pd.cut(housing['median_income'], 
                                    bins = [0, 1.5, 3.0, 4.5, 6, np.inf], 
                                    labels = [1,2,3,4,5])


#Training and Test Data
split = StratifiedShuffleSplit(n_splits = 1 ,test_size = 0.2, random_state = 42 )


for train_index, test_index in split.split(housing, housing['income_category']):
    strat_train_set = housing.loc[train_index].drop("income_category", axis = 1)
    strat_test_set = housing.loc[test_index].drop("income_category", axis = 1)

housing = strat_train_set.copy()

#Seperating Features and Labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value",axis = 1)

print(housing, housing_labels)

# Seperate Numerical and Categorical columns
num_attributes = housing.drop("ocean_proximity",axis = 1).columns.tolist()
cat_attributes = ["ocean_proximity"]

#Pipeline for Numerical columns
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown = "ignore"))
])


#Full Pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", cat_pipeline, cat_attributes)
])


#Transform Data 
housing_prepared = full_pipeline.fit_transform(housing)

print(housing_prepared.shape)

#Train Model

#linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
# print(f"The Root Mean Squared Error for Linear Regression is: {lin_rmse}")
lin_rmses = -cross_val_score(lin_reg,housing_prepared, housing_labels, scoring = "neg_root_mean_squared_error",cv = 10)
print(pd.Series(lin_rmses).describe())




#Decision Tree
dec_tree = DecisionTreeRegressor()
dec_tree.fit(housing_prepared,housing_labels)
dec_pred = dec_tree.predict(housing_prepared)
# dec_rmse = 
#root_mean_squared_error(housing_labels,dec_pred)
# print(f"The Root Means Squared Error for Decision Tree Regression is: {dec_rmse}")
dec_rmses = -cross_val_score(dec_tree,housing_prepared, housing_labels, scoring = "neg_root_mean_squared_error",cv = 10)
print(pd.Series(dec_rmses).describe())


#random_forest
ran_forest = RandomForestRegressor()
ran_forest.fit(housing_prepared,housing_labels)
ran_pred = ran_forest.predict(housing_prepared)
# ran_rmse = root_mean_squared_error(housing_labels,ran_pred)
# print(f"The Root Means Squared Error for Random Forest Regressor is: {ran_rmse}")
ran_rmses = -cross_val_score(ran_forest,housing_prepared, housing_labels, scoring = "neg_root_mean_squared_error",cv = 10)
print(pd.Series(ran_rmses).describe())

