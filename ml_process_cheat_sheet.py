# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 10:21:43 2022

@author: kenne
"""

# NOTE - add portion for creating pipelines
# Additional Project that goes through the whole process on one dataset (YouTube?)

##############################################################
# Import main libraries 
##############################################################
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
##############################################################
# Read in data 
##############################################################

#csv files
df = pd.read_csv('your_file_path.csv')
#json files 
df = pd.read_json('your_file_path.csv')

##############################################################
# Basic data exploration
##############################################################

#show columns 
df.columns

#show number of rows and columns
df.shape

#show descriptive statistics for numeric variables
df.describe()

#show descriptive statistics for categorical variables 
df.describe(include=np.object)

##############################################################
# Missing Values
##############################################################

from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

#Dropping null values - drops all rows with a null value in them
drop_df = df.dropna()

#Mean
df.loc[:,'variable_name'] = df['variable_name'].fillna(np.mean(df['variable_name']))

#Median
df.loc[:,'variable_name'] = df['variable_name'].fillna(np.median(df['variable_name']))

#Mode 
df.loc[:,'variable_name'] = df['variable_name'].fillna(np.mode(df['variable_name']))

#iterative imputer 
Imp = IterativeImputer(max_iter=10, random_state = 0)
Imp.fit(df)

imp_df = Imp.transform(df)

#Nearest Neighbors Imputer 
nn_imp = KNNImputer(n_neighbors=5, weights="uniform")
nn_imp.fit(df)

nn_imp_df = nn_imp.transform(df)

##############################################################
# Dealing with Outliers
##############################################################
import scipy.stats

#boxplots in seaborn 
sns.boxplot(df['variable_name'])

#function to extract outliers from boxplot 
def extract_outliers_from_boxplot(array):
    ## Get IQR
    iqr_q1 = np.quantile(array, 0.25)
    iqr_q3 = np.quantile(array, 0.75)
    med = np.median(array)

    # finding the iqr region
    iqr = iqr_q3-iqr_q1

    # finding upper and lower whiskers
    upper_bound = iqr_q3+(1.5*iqr)
    lower_bound = iqr_q1-(1.5*iqr)

    outliers = array[(array <= lower_bound) | (array >= upper_bound)]
    print('Outliers within the box plot are :{}'.format(outliers))
    return outliers

extract_outliers_from_boxplot(df['purchases'])

#Violin Plot 
plt.violinplot(df['variable_name'])


#Percentile Outlier detection 
def percentile_outliers(array,
                        lower_bound_perc,
                        upper_bound_perc):
    
    upper_bound = np.percentile(df['purchases'], upper_bound_perc)
    lower_bound = np.percentile(df['purchases'], lower_bound_perc)
    
    outliers = array[(array <= lower_bound) | (array >= upper_bound)]
    
    return outliers

#Z score outlier detection 
def z_score_outliers(array,
                     z_score_lower,
                     z_score_upper):

    z_scores = scipy.stats.zscore(array)
    outliers = (z_scores > 1.96) | (z_scores < -1.96)
    
    return array[outliers]

outliers = percentile_outliers(df['variable_name'],
               upper_bound_perc = 99,
               lower_bound_perc = 1)

z_score_outliers(df['variable_name'],
                     z_score_lower = -1.96,
                     z_score_upper = 1.96)

#Isolation forests 
from sklearn.ensemble import IsolationForest

features = ['var_1','var_2','var_3','var_4']

X = df[features]

clf = IsolationForest(n_estimators=50, max_samples=100)
clf.fit(df)

df['scores'] = clf.decision_function(df)
df['anomaly'] = clf.predict(X)

## Get Anomalies
outliers=df.loc[df['anomaly']==-1]


#z score & percentile outlier removal 
def z_score_removal(df, column, lower_z_score, upper_z_score):
    
    col_df = df[column]

    z_scores = scipy.stats.zscore(df['variable_name'])
    outliers = (z_scores > upper_z_score) | (z_scores < lower_z_score)
    return df[~outliers]

def percentile_removal(df, column, lower_bound_perc, upper_bound_perc):
    
    col_df = df[column]
    
    upper_bound = np.percentile(col_df, upper_bound_perc)
    lower_bound = np.percentile(col_df, lower_bound_perc)

    z_scores = scipy.stats.zscore(df['variable_name'])
    outliers = (z_scores > upper_bound) | (z_scores < lower_bound)
    return df[~outliers]

filtered_df = z_score_removal(df, 'purchases', -1.96, 1.96)
percentile_removal(df, 'purchases', lower_bound_perc = 1, upper_bound_perc = 99)

#winsorize 
def winsorize(df, column, upper, lower):
    col_df = df[column]
    
    perc_upper = np.percentile(df[column],upper)
    perc_lower = np.percentile(df[column],lower)
    
    df[column] = np.where(df[column] >= perc_upper, 
                          perc_upper, 
                          df[column])
    
    df[column] = np.where(df[column] <= perc_lower, 
                          perc_lower, 
                          df[column])
    
    return df

win_df = winsorize(df, 'variable_name', 97.5, 0.025)

###################################################################
#Exploratory Data Analysis
###################################################################

#Histogram 
plt.hist(df['variable_name'])

#box plot matplotlib (seaborn above)
plt.boxplot(df['variable_name'])

#scatter plot 
plt.scatter(df['var_1'] ,df['var_2'])

#scatter plot with trendline (seaborn)
sns.regplot(x='var_1',y='var_2', data = df)

#correlation matrix 
corr = df.corr()

#correlation plot 
sns.heatmap(corr)

#improved heatmap formatting 
sns.set_theme(style="white")
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(15, 10))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, annot_kws={"fontsize":8})

#pivot table 
pd.pivot_table(data=df, values = 'var_1', index = 'var_2', columns = 'var_3')

#Line charts 
sns.lineplot(data = df, x = 'time_series_data', y= 'variable_name')


######################################################################
# Feature Engineering 
######################################################################

#onehot encoding (all categoricals)
df = pd.get_dummies(df)

#onehot encoding specific categoricals 
mult_hot_enc = pd.get_dummies(df['variable_name'])

hot_encoded_df = pd.concat([df,mult_hot_enc], axis = 1)


#Ordinal Encoding 
from sklearn.preprocessing import OrdinalEncoder
data = np.asarray(df[['ordinal_variable_name']])
encoder = OrdinalEncoder()
result = encoder.fit_transform(data)
ord_encoded = pd.DataFrame(result)
ord_encoded.columns = ['ordinal_variable_name_encoded']
ord_encoded_df = pd.concat([df, ord_encoded], axis = 1)

#Frequency Encoder 
class FrequencyEncoder:
    def fit(self, train_df, column):
        self.train_df = train_df
        self.column = column
        
    def transform(self, test_df, column):
        frequency_encoded = self.train_df.groupby([self.column]).size()

        col_name = column + '_freq'
        test_df.loc[:,col_name] = test_df[column].apply(lambda x: frequency_encoded[x])
        return test_df


fe = FrequencyEncoder()
fe.fit(df, column='variable_name')
df_freq_enc = fe.transform(df, column='variable_name')

#Target encoding 
class TargetEncoder:
    def fit(self, train_df, target_col, categ_col):
        self.train_df = train_df
        self.target_col = target_col
        self.categ_col = categ_col
        
    def transform(self, test_df, column = None):
        if column is None:
            column = self.categ_col
        
        target_encoder = self.train_df.groupby([self.categ_col]).mean()[self.target_col]

        df[self.categ_col].apply(lambda x: target_encoder[x])

        col_name = column + '_target_enc'
        test_df.loc[:,col_name] = test_df[column].apply(lambda x: target_encoder[x])
        return test_df
    
te = TargetEncoder()
te.fit(df, target_col = 'var_1', categ_col = 'var_2')

te_df = te.transform(df)

#Probability Ratio Encoding 
class ProbabilityRatioEncoder:
    def fit(self, train_df, categ_col, target_col):
        self.train_df = train_df
        self.categ_col = categ_col
        self.target_col = target_col
        
    def transform(self, test_df, constant = 0):
        totals = self.train_df.groupby([self.categ_col]).size() 
        sums = self.train_df.groupby([self.categ_col]).sum()[self.target_col]

        ratio_encoder = (sums+ constant)/totals
        
        col_name = self.categ_col + '_prob_ratio'
        test_df.loc[:,col_name] = test_df[self.categ_col].apply(lambda x: ratio_encoder[x])
        return test_df

pre = ProbabilityRatioEncoder()

pre.fit(df, 'var_1','var_2')
pre_df = pre.transform(df)

#Weight of Evidence Encoding
class WeightofEvidenceEncoder:
    def fit(self, train_df, categ_col, target_col):
        self.train_df = train_df
        self.categ_col = categ_col
        self.target_col = target_col
        
    def transform(self, test_df, constant = 0):
        totals = self.train_df.groupby([self.categ_col]).size() 
        sums = self.train_df.groupby([self.categ_col]).sum()[self.target_col]

        woe_encoder = np.log((sums+ constant)/totals)
        
        col_name = self.categ_col + '_woe'
        test_df.loc[:,col_name] = test_df[self.categ_col].apply(lambda x: woe_encoder[x])
        return test_df
    
woe = WeightofEvidenceEncoder()

woe.fit(df, 'var_1','var_2')
woe_df = woe.transform(df)

# Absolute Max Scaling 
from sklearn.preprocessing import MaxAbsScaler
df_am = MaxAbsScaler().fit_transform(df)
df_am = pd.DataFrame(df_am, columns = df.columns)

#Min Max Scaling 
from sklearn.preprocessing import MinMaxScaler
df_min_max = MinMaxScaler().fit_transform(df)
df_min_max = pd.DataFrame(df_min_max, columns = df.columns)


# Z Score Normalization 
from sklearn.preprocessing import StandardScaler
df_std = df.copy()
df_std.loc[:,['var_1','var_2']] = StandardScaler().fit_transform(df_std.loc[:, ['var_1','var_2']])

# Robust Scaler
from sklearn.preprocessing import RobustScaler
df_rob = df.copy()
df_rob.loc[:,['var_1','var_2']] = RobustScaler().fit_transform(df_rob.loc[:, ['var_1','var_2']])


#Log Transform
from sklearn.preprocessing import FunctionTransformer

def log_transform(x):
    return np.log(x + 1)

transformer_log = FunctionTransformer(log_transform)
transformed_log = transformer_log.fit_transform(df)

#square root transform
def sqrt_transform(x):
    return np.sqrt(x)
transformer_sqrt = FunctionTransformer(sqrt_transform)
transformed_sqrt = transformer_sqrt.fit_transform(df)

#exponential transform 
def exp_transform(x):
    return np.exp(x)

transformer_exp = FunctionTransformer(exp_transform)

## In our dataset, car age may be something we want to magnify
transformed_exp = df.copy()

transformed_exp['variable_name'] = transformer_exp.fit_transform(transformed_exp['variable_name'])

#box cox transform
from scipy.stats import boxcox
boxcox_y_train = boxcox(df['variable_name'], lmbda = None)

#binning (set desired intervals )
bins = pd.IntervalIndex.from_tuples([(0, 50000), (50000, 100000), (100000,float("inf"))])
df['bin_variable'] = pd.cut(df['vairable'],bins)

##############################################################
# Cross Validation
##############################################################

#Train Test Split
from sklearn.model_selection import train_test_split

features = ['var_1','var_2','var_3','var_4','var_5']
X = df[features]
y = df['y_variable']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#K Fold Cross Validation 
from sklearn.model_selection import KFold

kf = KFold(n_splits=2, shuffle = True, random_state = 42)
kf.get_n_splits(X)

folds = {}

for train, test in kf.split(X):
    # Fold
    fold_number = 1
    # Store fold number
    folds[fold_number] = (df.iloc[train], df.iloc[test])
    print('train: %s, test: %s' % (df.iloc[train], df.iloc[test]))
    fold_number += 1

from sklearn.model_selection import cross_val_score
model = RandomForestClassifier() #any ml model 
scores = cross_val_score(model, X, y, scoring='accuracy', cv=kf, n_jobs=-1)
print(np.mean(scores))

#leave one out cross validation 
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

loo = LeaveOneOut()
loo.get_n_splits(X)


all_preds = []

for train_index, test_index in loo.split(X[:100]):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = RandomForestClassifier() #any ml model 

    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    correct = y_preds[0] == y_test.values[0]
    
    all_preds.append(correct)
    

#train test date split 

train_df = df[df['date'] < DATE].copy()
test_df = df[df['date'] >= DATE].copy()

X_train = train_df[features]
X_test = test_df[features]

y_train = train_df['y_var']
y_test = test_df['y_var']

model = RandomForestClassifier() #any ML Model

model.fit(X_train, y_train)
y_preds = model.predict(X_test)    

# sliding window time series k-fold 
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit()

all_scores = []

for train_index, test_index in tscv.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = RandomForestClassifier() #any ML Model here 

    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    pr_auc = average_precision_score(y_preds, y_test)
    
    all_scores.append(pr_auc)
    
#expanding window cross validation 
class ExpandingWindowCV:
    def fit(self, date_col, date_range = None, custom_range = None):
        self.date_col = date_col
        self.date_range = date_range
        self.custom_range = custom_range
        
        if date_range is not None and custom_range is not None:
            raise ValueError("Date Range and Custom Range both cannot be None.")
    
    def split(self, df):
        if self.date_range is None:         
            dates = list(set(df[self.date_col].astype(str).values))
        
        if self.date_range is not None:
            dates = pd.date_range(start=self.date_range[0], end=self.date_range[1])
            dates = [str(d.date()) for d in dates]
        
        if self.custom_range is not None:
            dates = self.custom_range
            
        for d in dates:
            df_train = df[df[self.date_col].astype(str) <= d].copy()
            df_test = df[df[self.date_col].astype(str) > d].copy()
            yield df_train, df_test
            
ew = ExpandingWindowCV()
ew.fit(date_col = 'date', date_range = ['2022-01-02','2022-01-08']) #choose date range
ew.split(df)

#monte Carlo cross validation 
from sklearn.model_selection import ShuffleSplit

rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
rs.get_n_splits(df)

all_scores = []
for train_index, test_index in rs.split(df):
#     print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = RandomForestClassifier() #any ML Model 

    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    pr_auc = average_precision_score(y_preds, y_test) #any scoring metric here 
    
    all_scores.append(pr_auc)

##################################################################
# Feature Selection 
##################################################################

target = 'variable_name'

#correlation / ANOVA 
correlation_threshold = 0.10 #choose threshold 

def correlation_selection(df,
                          features, 
                          target,
                          threshold):
    
    correlations = df[features + [target]].corr()[target]
    selected_features = correlations[abs(correlations)>threshold]
    
    remove_target = selected_features.index[selected_features.index != target]
    return selected_features[remove_target]

selected = correlation_selection(df,
                                 features,
                                 target,
                                 threshold = 0.10)

print(selected)

# Chi-Squares, ANOVA, F-Test, Mutual Info Gain
from sklearn.feature_selection import (
    SelectKBest, 
    chi2, 
    f_classif, 
    f_regression,
    r_regression,
    mutual_info_classif,
    mutual_info_regression
)

kb = SelectKBest(chi2, k=4)
X_new = kb.fit_transform(X,y)
X_new = pd.DataFrame(X_new)
X_new.columns = kb.get_feature_names_out()

# Forward Stepwise 
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression #any ml model here 

kb = SequentialFeatureSelector(LogisticRegression(),
                               n_features_to_select=4,
                              direction = 'forward')
X_new = kb.fit_transform(X,y)
X_new = pd.DataFrame(X_new)
X_new.columns = kb.get_feature_names_out()

# Backward Stepwise
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

kb = SequentialFeatureSelector(LogisticRegression(),
                               n_features_to_select=4,
                              direction = 'backward')
X_new = kb.fit_transform(X,y)
X_new = pd.DataFrame(X_new)
X_new.columns = kb.get_feature_names_out()

#Recursive Feature Selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

kb = RFE(LogisticRegression(), n_features_to_select=4)
X_new = kb.fit_transform(X,y)
X_new = pd.DataFrame(X_new)
X_new.columns = kb.get_feature_names_out()

#Exhaustive feature selection 
from sklearn.neighbors import KNeighborsClassifier #any ml model here 
from sklearn.datasets import load_iris
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

lr = LogisticRegression()

efs1 = EFS(lr, 
           min_features=1,
           max_features=4,
           scoring='accuracy',
           print_progress=True,
           cv=5)

efs1 = efs1.fit(X, y)

print('Best accuracy score: %.2f' % efs1.best_score_)
print('Best subset (indices):', efs1.best_idx_)
print('Best subset (corresponding names):', efs1.best_feature_names_)

#bi-directional elimination 
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sbs = SFS(LogisticRegression(),
         k_features=4,
         forward=True,
         floating=True,
         cv=0)
sbs.fit(X, y)
sbs.k_feature_names_

#Variance Threshold
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold()
selector.fit_transform(X)

##################################################################
# Imbalanced Data
##################################################################

#Random Oversampling 
from imblearn.over_sampling import RandomOverSampler

o_smpl = RandomOverSampler(random_state = 42) 

X_o_smpl, y_o_smpl = o_smpl.fit_resample(X_train,y_train)

#Random Undersampling 
from imblearn.under_sampling import RandomUnderSampler

u_smpl = RandomUnderSampler(random_state = 42) 

X_u_smpl, y_u_smpl = u_smpl.fit_resample(X_train,y_train)

#Synthetic Minority Oversampling (SMOTE)
from imblearn.over_sampling import SMOTE 

smote = SMOTE(random_state = 42) 

X_smote, y_smote = smote.fit_resample(X_train,y_train)

#Borderline SMOTE
from imblearn.over_sampling import BorderlineSMOTE

bsmote = BorderlineSMOTE(random_state = 42) 

X_bsmote, y_bsmote = bsmote.fit_resample(X_train,y_train)

# Adaptive Synthetic Oversampling (ADASYN)
from imblearn.over_sampling import ADASYN 
adasyn = ADASYN(random_state = 42) 

X_ada, y_ada = adasyn.fit_resample(X_train,y_train)

####################################################################
# Modeling 
####################################################################
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

#Basic model fitting & predicting 
model = GaussianNB() #Any ML Model Here

model.fit(X_train,y_train)
model.predict(X_test)

#Cross Validation Score 
nb_accuracy = cross_val_score(model,X_train,y_train.values.ravel(), cv=3, scoring ='accuracy')


#Randomized Search Paramater Tuning 
from sklearn.model_selection import RandomizedSearchCV

dt = DecisionTreeClassifier(random_state = 42) #any model here 

features = {'criterion': ['gini','entropy'], #relevant parameters to model 
            'splitter': ['best','random'],
           'max_depth': [2,5,10,20,40,None],
           'min_samples_split': [2,5,10,15],
           'max_features': ['auto','sqrt','log2',None]}

rs_dt = RandomizedSearchCV(estimator = dt, param_distributions =features, n_iter =100, cv = 3, random_state = 42, scoring ='f1')

rs_dt.fit(X_train,y_train)

#exhaustive parameter tuning (GridsearchCV) - decision tree example 
from sklearn.model_selection import GridSearchCV

features_gs = {'criterion': ['entropy'],
            'splitter': ['random'],
           'max_depth': np.arange(30,50,1), 
           'min_samples_split': [2,3,4,5,6,7,8,9],
           'max_features': [None]}

gs_dt = GridSearchCV(estimator = dt, param_grid =features_gs, cv = 3, scoring ='f1') #we don't need random state because there isn't randomization like before

gs_dt.fit(X_train,y_train)

#Bayesian Search CV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import StratifiedKFold

# Choose cross validation method 
cv = StratifiedKFold(n_splits = 3)

#replace 'dt' with any ML model & relevant parameters for that model 
bs_lr = BayesSearchCV(
    dt,
    {'criterion': Categorical(['gini','entropy']),
            'splitter': Categorical(['best','random']),
           'max_depth': Integer(10,50),
           'min_samples_split': Integer(2,15),
           'max_features': Categorical(['auto','sqrt','log2',None])},
    random_state=42,
    n_iter= 100,
    cv= cv,
    scoring ='f1')
 
bs_lr.fit(X_train,y_train.values.ravel())

#voting classifier 
from sklearn.ensemble import VotingClassifier

#replace models here with your models 
dt_voting = DecisionTreeClassifier() 
knn_voting = make_pipeline(StandardScaler(), KNeighborsClassifier())
lr_voting = LogisticRegression()

ens = VotingClassifier(estimators = [('dt', dt_voting), ('knn', knn_voting), ('lr',lr_voting)], voting = 'hard')

#stacking classifier
from sklearn.ensemble import StackingClassifier

## replace your models here 
ens_stack = StackingClassifier(estimators = [('dt', dt_voting), ('lr',lr_voting), ('nb',GaussianNB())], final_estimator = GaussianNB())


#################################################################
# Model Evaluation Metrics
#################################################################

#y_preds == predicted outcomes for your models 

#accuracy, precision, & recall
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


accuracy = accuracy_score(y_test, y_preds)
precision = precision_score(y_test, y_preds)
recall = recall_score(y_test, y_preds)


print("Accuracy: {0}".format(accuracy))
print("Precision: {0}".format(precision))
print("Recall: {0}".format(recall))

#f1 score 
f1 = f1_score(y_test, y_preds)

#Roc-Auc
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)


roc_auc = roc_auc_score(y_test, y_preds)

#PR-AUC
from sklearn.metrics import (
    average_precision_score
)

pr_auc = average_precision_score(y_test, y_preds)

#LOG Loss
from sklearn.metrics import (
    log_loss
)

log_loss = log_loss(y_test, y_preds)

#R^2, Mean Absolute Error, Mean Square Error 
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

r2 = r2_score(y_test, y_preds)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)


#Adjusted R^2
def adj_r2_score(X, y_test, y_preds):
    SS_reg = np.sum((y_test - y_preds)**2)
    SS_total = np.sum((y_test - np.mean(y_test))**2)
    r2 = 1-SS_reg/SS_total
    
    N = len(X)
    p = len(X.columns)
    
    adj_r2 = 1-((1-r2)*(N-1))/(N-p-1)
    return adj_r2
    
adj_r2_score(X, y_test, y_preds)

#Root Mean Square Error 
def mean_squared_error(y_test, y_preds):
    return np.sum((y_preds - y_test)**2)/len(y_preds)
    
np.sqrt(mean_squared_error(y_test, y_preds))

