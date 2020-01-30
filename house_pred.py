# %% [code]
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

from mlxtend.regressor import StackingCVRegressor

from lightgbm import LGBMRegressor

train = pd.read_csv("train.csv")


test = pd.read_csv("test.csv")



all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))


#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# %% [code]
#imputer  only for 2 features with most number of missing values

total = all_data[numeric_feats].isnull().sum().sort_values(ascending=False)
missing_data = pd.concat([total], axis=1, keys=['Total'])
#print(missing_data.head(20))

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.5]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

train["SalePrice"] = np.log1p(train["SalePrice"])

all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mode().iloc[0])


X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]
y = train.SalePrice

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression
from sklearn.model_selection import cross_val_score

#model_lasso = LassoCV(alphas = [1, 0.1,0.01, 0.001, 0.0005]).fit(X_train, y)

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# setup models    
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

ridge = make_pipeline(RobustScaler(),RidgeCV(alphas=alphas_alt, cv=kfolds))
                      

lasso = make_pipeline(RobustScaler(),LassoCV(max_iter=1e7, alphas=alphas2,random_state=42, cv=kfolds))
                              
                      

elasticnet = make_pipeline(RobustScaler(),ElasticNetCV(max_iter=1e7, alphas=e_alphas,cv=kfolds, l1_ratio=e_l1ratio))
                           
                                        

svr = make_pipeline(RobustScaler(),SVR(C=20, epsilon=0.008, gamma=0.0003, ))
                    

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,max_depth=4, max_features='sqrt',min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=42)
                                
                                
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)

xgb_preds = model_xgb.predict(X_test)
                               

lightgbm = LGBMRegressor(objective='regression',num_leaves=4,learning_rate=0.01,n_estimators=5000,max_bin=200, bagging_fraction=0.75,bagging_freq=5,bagging_seed=7,feature_fraction=0.2,  feature_fraction_seed=7,verbose=-1)
                        
     
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, model_xgb, lightgbm),meta_regressor = model_xgb)
                                           
                                
                                
#xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
 #                      max_depth=3, min_child_weight=0,
  #                     gamma=0, subsample=0.7,
  #                     colsample_bytree=0.7,
  #                     objective='reg:linear', nthread=-1,
  #                     scale_pos_weight=1, seed=27,
  #                     reg_alpha=0.00006)

# stack
#stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet,gbr, xgboost, lightgbm),meta_regressor=xgboost,use_features_in_secondary=True)


stack_gen_model = stack_gen.fit(np.array(X_train), np.array(y))

elastic_model_full_data = elasticnet.fit(X_train, y)

lasso_model_full_data = lasso.fit(X_train, y)

ridge_model_full_data = ridge.fit(X_train, y)

svr_model_full_data = svr.fit(X_train, y)

gbr_model_full_data = gbr.fit(X_train, y)

#xgb_model_full_data = xgboost.fit(X, y)

lgb_model_full_data = lightgbm.fit(X_train, y)





#lasso_preds = np.expm1(model_lasso.predict(X_test))


#regressor = LinearRegression()

#X_lr =  np.append(pd.DataFrame(np.expm1(model_xgb.predict(X_train))),pd.DataFrame(np.expm1(model_lasso.predict(X_train))),axis=1)
#Y_lr = np.expm1(y)

#print(X_lr,Y_lr)
#regressor.fit(X_lr,Y_lr)
#print(regressor.coef_)

#preds = 0.7*lasso_preds + 0.3*xgb_preds
#preds = -0.08*lasso_preds +1.09*xgb_preds
preds = (0.1* elastic_model_full_data.predict(X_test)) + (0.1 * lasso_model_full_data.predict(X_test)) + 0.3*stack_gen_model.predict(np.array(X_test)) + (0.05 * ridge_model_full_data.predict(X_test)) +  (0.1 * svr_model_full_data.predict(X_test)) +  (0.1 * gbr_model_full_data.predict(X_test)) + (0.15 * xgb_preds) +   (0.1 * lgb_model_full_data.predict(X_test)) 

preds = np.expm1(preds)
                  
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("ensemble.csv", index = False)
print("Done")

# %% [code]