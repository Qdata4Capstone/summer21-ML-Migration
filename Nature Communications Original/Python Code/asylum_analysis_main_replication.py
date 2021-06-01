import pandas as pd
import os
import numpy as np



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
	dataset['ln_depvar_pop'] = dataset['ln_depvar_pop'].apply(lambda x: np.log(x))

dataset['depvar'] = dataset['ln_depvar_pop']
print(dataset.columns)

impute_dataset = dataset[['country', 'year', 'depvar', 'highest_neighbor_dem', 'area', 'wdi_pop', 'wdi_urban_pop', 'distance_to_eu',\
'wdi_gdppc_growth', 'wdi_gdppc', 'perc_post_secondary', 'kof_index', 'wdi_imr','tmp_pop', 'spei3_gs_pos', 'spei3_gs_neg', \
'casualties_brd', 'annually_affected_20k', 'physical_integrity', 'free_movement', 'homicide']]


impute_dataset['wdi_pop'] = impute_dataset['wdi_pop'].apply(lambda x: np.log(x))
impute_dataset['area'] = impute_dataset['area'].apply(lambda x: np.log(x))
impute_dataset['wdi_gdppc'] = impute_dataset['wdi_gdppc'].apply(lambda x: np.log(x))
impute_dataset['casualties_brd'] = impute_dataset['casualties_brd'].apply(lambda x: np.log(x))
impute_dataset['annually_affected_20k'] = impute_dataset['annually_affected_20k'].apply(lambda x: np.log(x))
impute_dataset['homicide'] = impute_dataset['homicide'].apply(lambda x: np.log(x))

# .seed(12345)

print(impute_dataset)

dataset_matrix = impute_dataset.drop(columns=['country','year','depvar'])
target_matrix = impute_dataset['depvar']

print(dataset_matrix)
print(target_matrix)

### Pre-process step
# dataset['spei3_gs_neg'] = dataset['spei3_gs_neg'].apply(lambda x: np.log(x))

