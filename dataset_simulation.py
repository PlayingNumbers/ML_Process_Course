import pandas as pd
import numpy as np
from sklearn import datasets
import random

## Customer Lifetime Value Dataset

def make_clv_dataset():

	features, output, coef = datasets.make_regression(n_samples = 5000, n_features = 7,
	                            n_informative = 7, n_targets = 1,
	                            noise = 0.0, coef = True)

	df = pd.DataFrame(features, columns=['id',
		'age',
		'gender',
		'income',
		'days_on_platform',
		'city',
		'purchases'])

	def make_id(df, column):
		df[column] = df.index
		return df

	def create_from_list(df, column, col_list):

		final_list = []
		for i in range(0,len(df)):
			chosen = random.choice(col_list)
			final_list.append(chosen)

		df[column] = final_list
		return df

	def scale_by_list(df, column, col_scale):

		df[column] = abs(df[column])
		df[column] = df[column] * col_scale
		df[column] = df[column].astype(int)
		return df

	age_range = np.arange(10,90)
	gender = ['Male','Female']
	city = ['New York City', 'San Francisco','Miami','Tokyo','London']
	income_range = np.arange(50000,1000000,1000)


	df = make_id(df, column = 'id')
	df = create_from_list(df, 'age', age_range)
	df = create_from_list(df, 'gender', gender)
	df = create_from_list(df, 'city', city)
	df = scale_by_list(df, 'income', 100000)
	df = scale_by_list(df, 'purchases', 2)
	df = scale_by_list(df, 'days_on_platform', 30)

	return df


##### Missing Values Section ######
df = make_clv_dataset()

## Create MCAR dataset
df_missing_random = df.mask(np.random.random(df.shape) < .3)

## Create MNAR dataset
df = make_clv_dataset()
df['age'] = np.where(df['age'] > 50, np.nan, df['age'])
df['days_on_platform'] = np.where(df['days_on_platform'] < 1, np.nan, df['days_on_platform'])


df.to_csv("clv_data.csv")

print(df.head())

# print(pd.DataFrame(coef, columns=['True Coefficient Values']))

## Marketing Mix Model


## Customer Retention Dataset