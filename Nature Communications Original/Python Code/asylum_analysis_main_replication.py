import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# from autoimpute.imputations import MiceImputer,SingleImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.imputation.mice import MICE
from statsmodels.imputation.mice import MICEData
import statsmodels.api as sm
# from fancyimpute import MICE
# from fancyimpute import KNN#,SoftImpute


TRAINING_PERIOD = 4
LAG_PREDICTORS = True
NLAG = 1
DEPVAR = "asylum seekers" # "asylum seekers" | "asylum seekers (global stock)" | "refugees (global stock)"
LOG_TRANSFORM_DEPVAR = True
PER_CAPITA_TRANSFORM_DEPVAR = True
SPEI_MOVING_AVERAGES_3_6 = False
NSTEP_AHEAD = 1
CALCULATE_INTERACTIONS = True

# Read Dataset directly from file and pre-process
original_dataset = pd.read_csv("../Data/replication_data.csv")
original_dataset['n_asylum'] = original_dataset['n_asylum'].fillna(0)
original_dataset['n_asylum_eu28'] = original_dataset['n_asylum_eu28'].fillna(0)
original_dataset['n_refugees'] = original_dataset['n_refugees'].fillna(0)
original_dataset['spei3_gs_neg'] = original_dataset['spei3_gs_neg'].apply(lambda x: x*12)
original_dataset['spei3_gs_pos'] = original_dataset['spei3_gs_pos'].apply(lambda x: x*12)
original_dataset['physical_integrity'] = original_dataset['civil_liberties_combined']

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# print(original_dataset)

# Remove all microstates from the list
microstates = pd.read_csv("../Data/microstates.dat", sep = "\t", names=['gwcode','iso3','name','d1','d2'])
microstates_list = microstates['gwcode'].to_list()
dataset = original_dataset[~original_dataset['gwcode'].isin(microstates_list)]

''' 
Debug:
yy = original_dataset.groupby(by='gwcode')
for group,key in yy:
	if len(key)!=27:
		print(group, original_dataset[original_dataset['gwcode']==group]['country'] , len(key))

'''

# Original code did log(population+1),  I have directly did log population and inf values are 0
if DEPVAR == "asylum seekers":
	dataset['orig_depvar'] = dataset['n_asylum_eu28']
	dataset['ln_depvar_pop'] = dataset['n_asylum_eu28']
	dataset.loc[dataset['n_asylum_eu28']>0, 'ln_depvar_pop'] = dataset['n_asylum_eu28']/dataset['wdi_pop']/1e6
	dataset.loc[dataset['n_asylum_eu28']<=0, 'ln_depvar_pop'] = 1
	# dataset = dataset[dataset['ln_depvar_pop'].notna()]
	# dataset = dataset.dropna()

	# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	# 	print(dataset[dataset['ln_depvar_pop'].isnull()])
	# exit(0)
	dataset['ln_depvar_pop'] = pd.to_numeric(dataset['ln_depvar_pop'])
	dataset['ln_depvar_pop'] = dataset['ln_depvar_pop'].apply(lambda x: np.log(x))

# Has to be changed depending on what we are predicting
dataset['depvar'] = dataset['ln_depvar_pop']

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# 	print(dataset['depvar'])
	# print(dataset['orig_depvar'])
# exit(0)

impute_dataset = dataset[['country', 'year', 'depvar', 'highest_neighbor_dem', 'area', 'wdi_pop', 'wdi_urban_pop', 'distance_to_eu',\
						  'wdi_gdppc_growth', 'wdi_gdppc', 'perc_post_secondary', 'kof_index', 'wdi_imr',\
						  'tmp_pop', 'spei3_gs_pos', 'spei3_gs_neg', \
						  'casualties_brd', 'annually_affected_20k', 'physical_integrity', 'free_movement', 'homicide']]


impute_dataset['wdi_pop'] = impute_dataset['wdi_pop'].apply(lambda x: np.log(x))
impute_dataset['area'] = impute_dataset['area'].apply(lambda x: np.log(x))
impute_dataset['wdi_gdppc'] = impute_dataset['wdi_gdppc'].apply(lambda x: np.log(x))
impute_dataset['casualties_brd'] = impute_dataset['casualties_brd'].apply(lambda x: np.log(x+1))
impute_dataset['annually_affected_20k'] = impute_dataset['annually_affected_20k'].apply(lambda x: np.log(x+1))
impute_dataset['homicide'] = impute_dataset['homicide'].apply(lambda x: np.log(x+1))


# Second order tmp_pop
impute_dataset['tmp_pop_sq'] = impute_dataset['tmp_pop'].apply(lambda x: np.power(x,2))

impute_dataset['spei3_gs_pos'] = -1*impute_dataset['spei3_gs_pos']
impute_dataset['spei3_gs_neg'] = -1*impute_dataset['spei3_gs_neg']

# 3 year moving average
impute_dataset['spei3_gs_pos_r3'] = impute_dataset['spei3_gs_pos'].rolling(3,min_periods=1).mean()
impute_dataset['spei3_gs_neg_r3'] = impute_dataset['spei3_gs_neg'].rolling(3,min_periods=1).mean()
# 6 year moving average
# impute_dataset['spei3_gs_pos_r6'] = impute_dataset['spei3_gs_pos'].rolling(6,min_periods=1).mean()
# impute_dataset['spei3_gs_neg_r6'] = impute_dataset['spei3_gs_neg'].rolling(6,min_periods=1).mean()


# Imputations here
impute_dataset = impute_dataset[impute_dataset['year']>1998]

# Autoimpute
# impute_dataset = impute_dataset.drop(columns=['country','year']).copy()
# print(impute_dataset)
strats = {
	'highest_neighbor_dem':'pmm',
	'area':'pmm', 
	'wdi_pop':'pmm', 
	'wdi_urban_pop':'pmm', 
	'distance_to_eu':'pmm',
	'wdi_gdppc_growth':'pmm', 
	'wdi_gdppc':'pmm', 
	'perc_post_secondary':'pmm', 
	'kof_index':'pmm', 
	'wdi_imr':'pmm',
	'tmp_pop':'pmm', 
	'spei3_gs_pos':'pmm', 
	'spei3_gs_neg':'pmm',
	'spei3_gs_pos_r3':'pmm',
	'spei3_gs_neg_r3':'pmm',
	'tmp_pop_sq':'pmm',
	'casualties_brd':'pmm', 
	'annually_affected_20k':'pmm', 
	'physical_integrity':'pmm', 
	'free_movement':'pmm', 
	'homicide':'pmm'

}

# mice = MiceImputer(
#     n=2,
#     strategy=strats,
#     # predictors={"salary": "all", "gender": ["salary", "education", "weight"]},
#     # imp_kwgs={"pmm": {"fill_value": "random"}},
#     visit="left-to-right",
#     return_list=True
# )


# impute_dataset = impute_dataset.replace([np.inf, -np.inf], np.nan)
# mice.fit_transform(impute_dataset)
# c = impute_dataset['country']
# y = impute_dataset['year']
# d = impute_dataset['depvar']

# mod_impute = impute_dataset.drop(columns=['country','year','depvar']).copy()
# imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# col = impute_dataset.columns
# print(mod_impute.shape)
# mod_impute = imp.fit_transform(impute_dataset)
# print(mod_impute.shape)
# mod_impute = pd.DataFrame(data=mod_impute, columns=col)
# print(mod_impute.shape)
# # exit(0)
# mod_impute['country']=c
# mod_impute['year']=y
# mod_impute['depvar'] = d
# impute_dataset = mod_impute

# Naive Impute
# impute_dataset = impute_dataset.fillna(0)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(mod_impute['country'])
# print(mod_impute['depvar'])
# exit(0)

# Fancyimpute 
# fancyimpute.MICE().complete(impute_dataset)
# mice_imputer = IterativeImputer()
# mice_imputer = KNN()
# impute_dataset = mice_imputer.fit_transform(impute_dataset)


# statsmodel
# print(impute_dataset)
c = impute_dataset['country']
y = impute_dataset['year']
impute_dataset = impute_dataset.drop(columns=['country','year'])

print("Imputing.....")
imp = MICEData(impute_dataset)
fml = 'depvar ~ highest_neighbor_dem+ area+ wdi_pop+ wdi_urban_pop+ distance_to_eu+ wdi_gdppc_growth+ wdi_gdppc+ perc_post_secondary+ kof_index+ wdi_imr+tmp_pop+ spei3_gs_pos+ spei3_gs_neg+casualties_brd+ annually_affected_20k+ physical_integrity+ free_movement+ homicide'
mice = MICE(fml, sm.OLS, imp)
results = mice.fit(3, 10)
imputations = []
print(results.summary())
for j in range(2):
	imp.update_all()
	temp = imp.data
	temp['country'] = c
	temp['year'] = y

	temp.loc[temp['spei3_gs_neg']<=0, 'spei3_gs_neg'] = 0
	temp.loc[temp['spei3_gs_pos']<=0, 'spei3_gs_pos'] = 0
	temp.loc[temp['kof_index']<=0, 'kof_index'] = 0
	temp.loc[temp['wdi_urban_pop']<=0, 'wdi_urban_pop'] = 0
	temp.loc[temp['perc_post_secondary']<=0, 'perc_post_secondary'] = 0
	temp.loc[temp['wdi_imr']<=0, 'wdi_imr'] = 0
	temp.loc[temp['distance_to_eu']<=0, 'distance_to_eu'] = 0

	imputations.append(temp)


impute_dataset = imputations[0]
print(impute_dataset)

grouped_countries = impute_dataset.groupby(['country'])
# Lag the dataset
lagged_frame_original = pd.DataFrame()
for country, feats in grouped_countries:
	features = grouped_countries.get_group(country).sort_values(by='year')#.drop(columns=['country','year','depvar'])
	# print(features)
	features['depvar'] = features['depvar'].shift(-1)
	features = features[:-1]
	# print(features)
	lagged_frame_original = pd.concat([lagged_frame_original,features])
# print(lagged_frame)
# exit(0)

#Construct year slices
year_range = [1999,2018]
window = 4
step_ahead = 1
slices = []
for i in range(year_range[0],year_range[1]-window+1):
	slices.append(([j for j in range(i,i+window)],i+window-1+step_ahead))

slices = slices[:-1]

print(slices,len(slices))
# exit(0)


baseline_columns = ['highest_neighbor_dem', 'area', 'wdi_pop', 'wdi_urban_pop', 'distance_to_eu']
economy_columns = ['wdi_gdppc_growth', 'wdi_gdppc', 'perc_post_secondary', 'kof_index', 'wdi_imr']
climate_columns = ['tmp_pop', 'spei3_gs_pos', 'spei3_gs_neg','spei3_gs_pos_r3','spei3_gs_neg_r3','tmp_pop_sq']
violence_columns = ['casualties_brd', 'annually_affected_20k', 'physical_integrity', 'free_movement', 'homicide']


to_drop = [
	economy_columns+climate_columns, #only violence+baseline
	economy_columns+violence_columns, #only environment+baseline
	climate_columns+violence_columns, #only economy+baseline
	[], 								#all
	economy_columns+violence_columns+climate_columns #only baseline
]
for dropped in to_drop:
	avg_mae = 0
	for idx in range(len(slices)):
		s = slices[idx]
		print(idx,"/",len(slices))
		lagged_frame = lagged_frame_original.drop(columns=dropped)

		train_set = lagged_frame.loc[lagged_frame['year'].isin(s[0])]
		train_features = train_set.drop(columns=['country','year','depvar']).to_numpy()
		train_targets = train_set['depvar'].to_numpy()
		# print(train_features)
		scaler_train = StandardScaler(with_mean=True,with_std=True)
		train_features = scaler_train.fit_transform(train_features,train_targets)
		# print(train_features)
		# exit(0)

		test_set = lagged_frame.loc[lagged_frame['year'] == s[1]]
		test_features = test_set.drop(columns=['country','year','depvar']).to_numpy()
		test_targets = test_set['depvar'].to_numpy()
		scaler_test = StandardScaler(with_mean=True,with_std=True)
		test_features = scaler_test.fit_transform(test_features,test_targets)

		clf = RandomForestRegressor(n_estimators=1000, min_samples_split=5)
		clf.fit(train_features,train_targets)
		predictions = clf.predict(test_features)
		# print(predictions)
		# print(test_targets)
		new_pred = []
		new_test = []
		cnt=0
		for i in range(len(predictions)):
			if test_targets[i]!=0:
				new_pred.append(predictions[i])
				new_test.append(test_targets[i])
		# mae = mean_absolute_error(np.exp(new_test),np.exp(new_pred))
		mae = mean_absolute_error(new_test,new_pred)
		# mae = mean_squared_error(new_test,new_pred)
		avg_mae+=mae
		print("MAE:",mae,"Skipped",len(predictions)-len(new_pred))
	print("Average MAE",avg_mae/len(slices))


exit(0)


dataset_matrix = []
target_matrix = []

for country, feats in grouped_countries:
	features = grouped_countries.get_group(country)#.drop(columns=['country','year','depvar'])
	print(features)
	break



# print(dataset_matrix)
# print(target_matrix)

# from autoimpute.imputations import SingleImputer, MultipleImputer, MiceImputer
# mice = MiceImputer()


### Pre-process step
# dataset['spei3_gs_neg'] = dataset['spei3_gs_neg'].apply(lambda x: np.log(x))

