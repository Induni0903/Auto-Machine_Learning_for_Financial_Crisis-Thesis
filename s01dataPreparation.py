"""
This script loads crisis-related economic data, cleans and enriches it with 
country information, performs binning, and prepares datasets for machine 
learning tasks, including rolling-window datasets.

Overview
========

1. Load data from Stata and CSV files
2. Clean and preprocess missing values
3. Merge with country-level metadata
4. Feature Engineering (e.g binning GDPper capita and land area)
5. Building rolling window datset
6. Save output datasets  into SQLite database

Dataset
=======
 The proposed dataset is an unbalanced panel dataset of 129 countries, coverring a period
 from 1983 to 2017.It is composed with following variabels.

 * Dependant Variable
    System banking crisis (Defined based on Laeven & Valencia, 2018, whom this event
    occurs if, a given year, there are signs of inancial distress in the banking system, such as bank runs,
    losses in the banking system (such as bank runs, losses in the banking system and the bank liquidations
    and government interventions due to significant losses in the banking system)).

    * **crisis0**: binary target variable (1 if crisis occurred, 0 otherwise) (If the observations for a given  in a given year are missing, the corresponding row has been added by setting the target
         variable crisis_0 to 0 (it is assumed that there was no crisis) and setting the covariate values to NA
         (missing value).

    * **crisis0_1**: one-year lagged target variable.

* Explanatory Variables
   The list of explanotary vriables are compiled in EWS, by the contributions off Demirguç-Kunt and Detragiache (2005),Davis and Karim (2008),
   and Caggianoetal.(2016). These expalnatory variables are categorised into 3 groups.
   
   1. Macroeconomic Information(All these variables, with the exeption of real interest rate, negatively affect the probability of a systemic 
      banking crisis.):
      
      * *real_gdp_growth_ifs_1*: Real GDP growth rate.
      * *lgdp_per_capita_wdi_1*: Logorithm of capita GDP.
      * *defl_gdp_growth_wdi_1*: Inflation.
      * *realirate_ifs_1*: Real interest rate.

   2. Monetary Variables:
      * *m2_to_fereserves_wdi_1*: Broad money(M2) over foreign exchange (expected to positively affect
         the probability of crisis occuring).
      * *domcrpriv_growth_wdi_1*: Growth rate of private credit (supposed to exert a positive effect on the
         crisis probability, which can be expected to be increasing with overindebtedness and deterioration
         of banks’asset quality).
   3. Financial Information:
      * *ld_nfassets_to_gdp_wdi_1*: Growth rate of net foreign assests to GDP (it is expected to negatively
         affect the likelihood of a systemic banking crisis).

Configuration
=============

The following parameters can be adjusted in the script:

- ``fn``: Input `.dta` file with crisis data
- ``cdfn``: CSV file with country-level dimensions
- ``dboutfn``: Output SQLite database name
- ``nLabelsLandAreaKm2``: Binning levels for land area
- ``nLabelsLgdp_per_capita``: Binning levels for GDP per capita
- ``testSizePerc``: Fraction for test split
- ``randomSeed``: Random state for reproducibility

Processing Steps
================

1. **Load the Crisis Data and Data Imputation**

   - Reads from a Stata file using `pandas.read_stata`.
   - Basic stats and null checks are printed and impute data.
   - Following columns are droped from the original dataset, as they contain many null value. The dropped columns along with the null values.
          - `liquidity_wdi`: 1545
          - `depositr_ifs` : 308
          - `discountr_ifs` : 1726
          - `change_nb_tot_wdi`: 468
          - `lendirate_wdi`: 257
          - `crisis2`: 161
          - `domcrpriv_to_gdp_growth_wdi`: 5

   - Imputes `ld_nfassets_to_gdp_wdi_1`(contains 160 null values) with 0 and create new column `aux1` to tracks changes(imputation).

2. **Add Country Metadata and binning variables**

   - Country metadata and binning variables(nLabelsLandAreaKm2, nLabelsLgdp_per_capita) are created to ensure that the countries from each continent, land area and gdp per capita were propotionally represented in the stratified sampling phase.
   - All the countries in the dataset is mapped for following continents: `North America and Oceania` (Oceania countries have no crisis. Mapping North America and Oceania"), `South America`, `Europe`, `Asia`, `Africa`
   - Based on country's land area, refering to this variable `LandArea(km2)`, countries are binned (Number of bins are 2).
   - Based on gdp per capita, refering to this variable `lgdp_per_capita_wdi_1`, countrie are binned (Number of bins are 2).

4. **Stratified Sampling**

   - Constructs a `sel` column, which is created using the information coolected from the metadata and binning variables.
   - The ``grp`` column in ``df2`` is used to track how each record is handled during grouped undersampling and stratified train/test splitting, particularly to balance the `crisis0` label.
   - Main group labels:
       - ``balanced_train`` : Record is in the balanced training set.
       - ``balanced_test``  : Record is in the balanced test set.
       - ``test2`` : Record was discarded during undersampling of a majority class.
       - ``test3`` : Record from a group with insufficient class diversity for splitting.
   - Each group (based on a value in the `sel` column) is processed as follows:
       - If a group has **only one class**, it is assigned to ``test3``.
       - If a group has **only one record of the minority class**, both classes are sent to the ``balanced_train``.
       - If a group has **enough samples from both classes**, balanced undersampling is performed and a stratified **train/test split** is executed.
       - Records not selected during sampling are marked as ``test2`` (discarded).
   - The ``balanced_train`` target variable(`crisis0`) value frequencies are: 0-149, 1-149
   - The ``balanced_test`` target variable value frequencies are: 1-56, 0-56

5. **Save Intermediate Results**

   - Saves cleaned and enriched data to:
     - ``cr01_original``: raw crisis data
     - ``cr02_quality``: after cleaning and merging
     - ``cr03_ml``: stratified ML dataset

6. **Create Rolling Window Dataset**

   - Rolling window size of 5, is created only for following variables: `crisis0`, `real_gdp_growth_ifs_1`, `lgdp_per_capita_wdi_1`, `defl_gdp_growth_wdi_1`, `realirate_ifs_1`, `m2_to_fereserves_wdi_1`, `domcrpriv_growth_wdi_1`, `ld_nfassets_to_gdp_wdi_1`, `aux1`
   - The code is shown below:

"""

import pandas as pd
import sqlite3 as db




#---------------------------------------------------
#: Configuration
#---------------------------------------------------

# The file name containing the original data
fn = 'ews_data_nomiss.dta'

# The file containing the country information
cdfn = 'country_dimensions.csv'

# The output file, containing the dataset for the ML tasks
dboutfn = 'ews_data_nomiss.sqlite'

# Message to confirm an operation execution
okMsg='Done\n\n'

# Number of bins for bining the LandAreaKm2 variable
nLabelsLandAreaKm2=2 #  3, 4 

# Number of bins for bining the Lgdrp_per_capita_variable
nLabelsLgdp_per_capita=2 #  3, 4 

# The test size will be ... of the whole training+test dataset
testSizePerc=0.3

# seed for random operations
randomSeed=42



#:-------------Data Preparation------------------------------
#: Crisis Data loading
#:""" Loading the crisis data """
#:print('Loading data from Stata file {fn}')
#:df = pd.read_stata(fn)
#:print(okMsg)

#:print('Matrix shape')
#:print(df.shape)
#:print(okMsg)

#:print('Columns')
#:print(list(df.columns))
#:print(okMsg)

#:print(f'Erasing the output db file {dboutfn}')
#:with open(dboutfn, 'w'):pass 
#:print(okMsg)
#:

#:print('Showing some numbers')
#:df.year.value_counts();
#:print(df.year.min(), df.year.max() )
#:df.crisis0.value_counts()
#:print(okMsg)

#:print(f'Saving orginal data into table cr01_original in {dboutfn}')
#:con = db.connect(dboutfn)
#:df.to_sql('cr01_original', con, index=False, if_exists='replace')
#:con.close()
#:print(okMsg)

#:print('Null values per column')
#:print(df.isna().sum(axis=0))
#: An output example 

#:"""

#:imfcode                           0
#:year                              0
#:isocode                           0
#:countryname                       0
#:crisis0                           0
#:d_crisis                          0
#:defl_gdp_growth_wdi               0
#:liquidity_wdi                  1545
#:depositr_ifs                    308
#:discountr_ifs                  1726
#:defl_gdp_growth_ifs               0
#:usdexrate_growth_ifs              0
#:domcrpriv_to_gdp_growth_wdi       5
#:domcrpriv_growth_wdi              0
#:real_gdp_growth_wdi               0
#:real_gdp_growth_ifs               0
#:lgdp_per_capita_wdi               0
#:change_nb_tot_wdi               468
#:m2_to_fereserves_wdi              0
#:realirate_ifs                     0
#:realirate_wdi                     0
#:lendirate_wdi                   257
#:nfassets_to_gdp_wdi               0
#:crisis1                           0
#:crisis2                         161
#:crisis3                           0
#:esample                           0
#:real_gdp_growth_ifs_1             0
#:lgdp_per_capita_wdi_1             0
#:defl_gdp_growth_wdi_1             0
#:realirate_ifs_1                   0
#:m2_to_fereserves_wdi_1            0
#:domcrpriv_growth_wdi_1            0
#:ld_nfassets_to_gdp_wdi            0
#:ld_nfassets_to_gdp_wdi_1        160
#:crisis0_1                         0

#:"""
#:print(okMsg)

#:print('Dropping columns containing too mmanych null values')
#:col2NulFreq={
#:    'liquidity_wdi':1545,
#:   'depositr_ifs':308,
#:   'discountr_ifs':1726,
#:   'change_nb_tot_wdi':468,
#:   'lendirate_wdi':257,
#:  'crisis2':161,
#:    #'ld_nfassets_to_gdp_wdi_1':160,
#:   'domcrpriv_to_gdp_growth_wdi':5,
#:}
#:col2drop = list(col2NulFreq.keys())
#:print('Columns to drop', col2drop)
#:df.drop(columns=col2drop, inplace=True)
#:print(okMsg)

#:print("Inputing varialble ld_nfassets_to_gdp_wdi_1")
#:# Managing 'ld_nfassets_to_gdp_wdi_1'
#:# in case of null value, 1 is inputed. 
#:# An auxiliary variable aux1 will be set 
#:# to 1 in case of input, 0 otherwise.
#:df['aux1']=0 # Creating a new column, full of 0. I assume the default is no inputation

#:selMask = df['ld_nfassets_to_gdp_wdi_1'].isna()
#:df.loc[selMask, 'aux1']=1
#:df.loc[selMask, 'ld_nfassets_to_gdp_wdi_1']=0 # Using 0 as input value
#:print(okMsg)

#:print('Counting the remaining Null values')
#:print(df.isna().sum().sum())
#:print(okMsg)

#:"""
#:2. Add Country Metadata and binning variables

#:   - Country metadata and binning variables(nLabelsLandAreaKm2, nLabelsLgdp_per_capita) are created to ensure that the countries from each continent, land area and gdp per capita were propotionally represented in the stratified sampling phase.
#:   - All the countries in the dataset is mapped for following continents: `North America and Oceania` (Oceania countries have no crisis. Mapping North America and Oceania"), `South America`, `Europe`, `Asia`, `Africa`
#:   - Based on country's land area, refering to this variable `LandArea(km2)`, countries are binned (Number of bins are 2).
#:   - Based on gdp per capita, refering to this variable `lgdp_per_capita_wdi_1`, countrie are binned (Number of bins are 2).

#:"""

#:#####################################
#:# Adding Country Information


#:# Country to Continent mapping information
#:country2continent={
#:    # North America
#:    "United States": "North America",
#:    "Canada": "North America",
#:    "Mexico": "North America",
#:    "Costa Rica": "North America",
#:    "Dominican Republic": "North America",
#:    "El Salvador": "North America",
#:    "Guatemala": "North America",
#:    "Haiti": "North America",
#:    "Honduras": "North America",
#:    "Nicaragua": "North America",
#:    "Panama": "North America",
#:    "Belize": "North America",
#:    "Jamaica": "North America",
#:    "St. Kitts and Nevis": "North America",
#:    "Trinidad and Tobago": "North America",
#:    "Barbados": "North America",
#:    "Dominica": "North America",
#:    "Grenada": "North America",

#:    # South America
#:    "Argentina": "South America",
#:    "Bolivia": "South America",
#:    "Brazil": "South America",
#:    "Chile": "South America",
#:    "Colombia": "South America",
#:    "Ecuador": "South America",
#:    "Paraguay": "South America",
#:    "Peru": "South America",
#:    "Uruguay": "South America",
#:    "Venezuela": "South America",
#:    "Guyana": "South America",
#:    "Suriname": "South America",

#:    # Europe
#:    "United Kingdom": "Europe",
#:    "Sweden": "Europe",
#:    "Switzerland": "Europe",
#:    "Iceland": "Europe",
#:    "Albania": "Europe",
#:    "Belarus": "Europe",
#:    "Bulgaria": "Europe",
#:    "Czech Republic": "Europe",
#:    "FYR Macedonia": "Europe",
#:    "Hungary": "Europe",
#:    "Poland": "Europe",
#:    "Moldova": "Europe",
#:    "Serbia": "Europe",
#:    "Croatia": "Europe",
#:    "Bosnia and Herzegovina": "Europe",
#:    "Ukraine": "Europe",
#:    "Georgia": "Europe",
#:    "Armenia": "Europe",
#:    "Azerbaijan": "Europe",

#:    # Asia
#:    "Japan": "Asia",
#:    "Turkey": "Asia",
#:    "Israel": "Asia",
#:    "Jordan": "Asia",
#:    "Kuwait": "Asia",
#:    "Lebanon": "Asia",
#:    "Yemen": "Asia",
#:    "Bangladesh": "Asia",
#:    "Bhutan": "Asia",
#:    "Brunei Darussalam": "Asia",
#:    "Myanmar": "Asia",
#:
#:    "Cambodia": "Asia",
#:    "Sri Lanka": "Asia",
#:    "Hong Kong SAR": "Asia",
#:    "India": "Asia",
#:    "Indonesia": "Asia",
#:    "Korea": "Asia",
#:    "Lao P.D.R.": "Asia",
#:    "Malaysia": "Asia",
#:    "Maldives": "Asia",
#:    "Nepal": "Asia",
#:    "Pakistan": "Asia",
#:    "Philippines": "Asia",
#:    "Singapore": "Asia",
#:    "Thailand": "Asia",
#:    "Vietnam": "Asia",
#:    "Kyrgyz Republic": "Asia",
#:    "Tajikistan": "Asia",
#:    "China": "Asia",
#:    "Mongolia": "Asia",

#:    # Africa
#:    "South Africa": "Africa",
#:    "Algeria": "Africa",
#:    "Angola": "Africa",
#:    "Burundi": "Africa",
#:    "Cameroon": "Africa",
#:    "Cabo Verde": "Africa",
#:    "Central African Republic": "Africa",
#:    "Chad": "Africa",
#:    "Comoros": "Africa",
#:    "Republic of Congo": "Africa",
#:    "Democratic Republic of the Congo": "Africa",
#:    "Benin": "Africa",
#:    "Equatorial Guinea": "Africa",
#:    "Ethiopia": "Africa",
#:    "Gabon": "Africa",
#:    "The Gambia": "Africa",
#:    "Ghana": "Africa",
#:    "Guinea-Bissau": "Africa",
#:    "Guinea": "Africa",
#:    "Cote d'Ivoire": "Africa",
#:    "Kenya": "Africa",
#:    "Lesotho": "Africa",
#:    "Liberia": "Africa",
#:    "Libya": "Africa",
#:    "Madagascar": "Africa",
#:    "Mali": "Africa",
#:    "Mauritania": "Africa",
#:    "Mauritius": "Africa",
#:    "Morocco": "Africa",
#:    "Mozambique": "Africa",
#:    "Niger": "Africa",
#:    "Nigeria": "Africa",
#:    "Zimbabwe": "Africa",
#:    "Rwanda": "Africa",
#:    "Sao Tome and Principe": "Africa",
#:    "Seychelles": "Africa",
#:    "Senegal": "Africa",
#:    "Sierra Leone": "Africa",
#:    "Namibia": "Africa",
#:    "Swaziland": "Africa",
#:    "Tanzania": "Africa",
#:    "Togo": "Africa",
#:    "Uganda": "Africa",
#:    "Burkina Faso": "Africa",
#:    "Zambia": "Africa",
#:    "Egypt": "Africa",

#:    # Oceania
#:    "Australia": "Oceania",
#:    "New Zealand": "Oceania",
#:    "Fiji": "Oceania",
#:    "Papua New Guinea": "Oceania",
#:}

#:print('Oceania countries have no crisis. Mapping North America and Oceania into "North America and Oceania"')
#:country2continent = {country:('North America and Oceania' if continent=='Oceania' or continent=='North America' else  continent) for country,continent in country2continent.items() }
#:continentLi = list(set(country2continent.values()))
#:df['continent'] = df['countryname'].replace(country2continent)
#:print(okMsg)

#:print('Showing country numbers')
#:print(df.continent.value_counts())
#:print(okMsg)

#:print(f'Loading country data from file {cdfn}')
#:dfcd = pd.read_csv(cdfn)
#:# Renaming columns to simplify their management
#:dfcd = dfcd.rename(columns={'Continent':'Continent2', 'LandArea(km2)':'LandAreaKm2'})
#:print(okMsg)

#:print('Merging Crisis and Country data')
#:df2 = pd.merge(df, dfcd, left_on='countryname', right_on='Country_x',    ) 


#:print('Bining Countries based on their Land Area. Each continent is managed independently from the other ones i.e., the bining focues only on the countries of the same continent')

#:# Creating the Bining variable for LandAreaKm2
#:df2['QR_LandAreaKm2']=-1
#:for cont in continentLi: # ['Asia', 'South America', 'North America', 'Europe', 'Africa']
#:    # Performing the bining operation for each continent separately
#:    contMask=df2.continent==cont
#:    # Quantile-based discretization function
#:    tempRes = pd.qcut(
#:        df2.loc[contMask, 'LandAreaKm2'], 
#:        nLabelsLandAreaKm2, # Number of bins
#:        labels = False
#:    )
#:    #print(cont, tempRes.value_counts())
#:    df2.loc[contMask, 'QR_LandAreaKm2'] = tempRes
#:print("df2['QR_LandAreaKm2'].value_counts()")
#:print(df2['QR_LandAreaKm2'].value_counts())
#:#print(df2.head())
#:print(okMsg)



#:print('Bining Countries based on the variable lgdp_per_capita_wdi_1')
#:df2['QR_lgdp_per_capita_wdi_1']=-1
#:for cont in continentLi: # ['Asia', 'South America', 'North America', 'Europe', 'Africa']
#:    contMask=df2.continent==cont
#:    df2.loc[contMask, 'QR_lgdp_per_capita_wdi_1'] = pd.qcut(
#:        df2.loc[contMask, 'lgdp_per_capita_wdi_1'], 
#:        nLabelsLgdp_per_capita,
#:        labels = False
#:    )
#:print("df2['QR_lgdp_per_capita_wdi_1'].value_counts()")
#:print(df2['QR_lgdp_per_capita_wdi_1'].value_counts())
#:#print(df2.head())
#:print(okMsg)

#:print('Saving the Dataset v2 i.e., the dataset enriched with the country data and where the null values have been either dropped or inputed')
#:print(f'Saved table cr01_original in {dboutfn}')
#:df2.to_sql('cr02_quality', con, index=False, if_exists='replace')
#:print(okMsg)

#:"""
#:4. **Stratified Sampling**

#:   - Constructs a `sel` column, which is created using the information coolected from the metadata and binning variables.
#:   - The ``grp`` column in ``df2`` is used to track how each record is handled during grouped undersampling and stratified train/test splitting, particularly to balance the `crisis0` label.

#:"""


#:#######################################
#:# Creating the datasets for Machine Learning

#:#assert nLabelsLgdp_per_capita==2 and nLabelsLandAreaKm2==2

#:print('Creating the stratification variable "sel"')
#:df2['sel']=''
#:df2['grp']='test3' # It wil be changed later, therefore this value doesn't matter 

#:# Possible values for df2['grp']
#:# * balanced_train # This is the balanced train subset
#:# * balanced_test  # This is the balanced test subset
#:# test2 # If a group must be undersampled (to reduce a majority class) 
#:#       # this label is uded for the records discarded during the undersample activity.
#:# test3 # To split the records of a group into train and test subsets, a group must have at least 2 records.
#:#       # This label is used for the records of groups where one label of the two is missing (e.g., crisis is missing or non crisis is missing)
#:#       # If the minority label features only one record in the group, then it is sent to the training set
#:# The set of features used to build groups (which will drive the undersampling and stratified train and test split)
#:gbCol = ['continent', 'Country_y', 'QR_lgdp_per_capita_wdi_1', 'QR_LandAreaKm2']


#:# creating the sel column. This variable will be used as stratification variable
#:# Each sel value is the '.' separated merge of all elements in the previously selected columns
#:df2['sel']=df2.loc[:, gbCol].apply(lambda li: '.'.join([str(el) for el in li]), axis=1)
#:print(df2['sel'].value_counts())
#:print(okMsg) 

#:"""
#:- Main group labels:
#:       - ``balanced_train`` : Record is in the balanced training set.
#:       - ``balanced_test``  : Record is in the balanced test set.
#:       - ``test2`` : Record was discarded during undersampling of a majority class.
#:       - ``test3`` : Record from a group with insufficient class diversity for splitting.
#:   - Each group (based on a value in the `sel` column) is processed as follows:
#:       - If a group has **only one class**, it is assigned to ``test3``.
#:       - If a group has **only one record of the minority class**, both classes are sent to the ``balanced_train``.
#:       - If a group has **enough samples from both classes**, balanced undersampling is performed and a stratified **train/test split** is executed.
#:       - Records not selected during sampling are marked as ``test2`` (discarded).
#:"""


#:print('Performing undersampling and train & test split')
#:for groupName, freq in df2['sel'].value_counts().items():
#:    #print(groupName,freq) # North America and OceaniaHonduras01 35
#:    df2sub = df2.loc[df2.sel==groupName,:]
#:    crisisNum=len(df2sub[df2sub.crisis0==1])
#:    nonCrisisNum=len(df2sub[df2sub.crisis0==0])
#:    #print(groupName, freq, crisisNum, nonCrisisNum)
#:    minNum = min(crisisNum, nonCrisisNum)
#:    if minNum==0:
#:        # crisis or non crisis is missing, these will go to test3
#:        df2.loc[df2sub.index, 'grp']='test3'
#:    elif minNum==1: 
#:        # there won't be enough elements to split into train test
#:        # I'm sending this to the train
#:        for target in (0,1):
#:            indLi = df2sub.loc[df2sub.crisis0==target].sample(minNum).index
#:            df2.loc[indLi, 'grp']='balanced_train'
#:    else:
#:        # At the beginning I assume everything goes to test2, 
#:        df2.loc[df2sub.index, 'grp']='test2'
#:        testSize=max(1, int(minNum*testSizePerc+0.49))
#:        trainSize=minNum-testSize
#:        #print(minNum, trainSize, testSize)
#:        # sampling from both crisis and not crisis a number of records equal to minNum
#:        for target in (0,1):
#:            indLi = df2sub.loc[df2sub.crisis0==target].sample(
#:                minNum, 
#:                random_state=randomSeed,
#:            ).index
#:            # ****** train test split is performed here
#:            indLiTest =df2sub.loc[indLi,:].sample(
#:                testSize, 
#:                random_state=randomSeed,
#:                ).index
#:            indLiTrain = indLi.difference(indLiTest)
#:            #df2.loc[indLi, 'grp']='balanced'
#:            df2.loc[indLiTrain, 'grp']='balanced_train'
#:            df2.loc[indLiTest, 'grp']='balanced_test'
            
#:print('Groups value_counts()')            
#:print(df2.grp.value_counts())
#:print()
#:print('balanced train and subsets value_counts()')
#:print(df2.loc[(df2.grp=='balanced_train')|(df2.grp=='balanced_test'), 'crisis0'].value_counts())
#:print(okMsg)


#:print(f'Saving (Non rolling window) dataset for ML into table cr03_ml of file {dboutfn}')
#:df2.to_sql('cr03_ml', con, index=False, if_exists='replace')
#:print(okMsg)

#:print('Checking the balanced_train target variable value frequencies')
#:print(df2.loc[df2.grp=='balanced_train', 'crisis0'].value_counts())

#:"""
#:- The ``balanced_train`` target variable(`crisis0`) value frequencies are: 0-149, 1-149
#:- The ``balanced_test`` target variable value frequencies are: 1-56, 0-56

#:crisis0
#:0    149
#:1    149

#:"""
#:print(okMsg)

#:print('Checking the balanced_test target variable value frequencies')
#:print(df2.loc[df2.grp=='balanced_test', 'crisis0'].value_counts())

#:"""

#:crisis0
#:1    56
#:0    56

#:"""
#:print(okMsg)

#:"""
#:5. **Save Intermediate Results**

#:   - Saves cleaned and enriched data to:
#:     - ``cr01_original``: raw crisis data
#:     - ``cr02_quality``: after cleaning and merging
#:     - ``cr03_ml``: stratified ML dataset

#:6. **Create Rolling Window Dataset**

#:   - Rolling window size of 5, is created only for following variables: `crisis0`, `real_gdp_growth_ifs_1`, `lgdp_per_capita_wdi_1`, `defl_gdp_growth_wdi_1`, `realirate_ifs_1`, `m2_to_fereserves_wdi_1`, `domcrpriv_growth_wdi_1`, `ld_nfassets_to_gdp_wdi_1`, `aux1`
#:   - The code is shown below:

#:"""

#:#######################################
#:# Creating the rolling Window Dataset for Machine Learning

#:print(f'Creating table cr03_ml_rolling on {dboutfn}. Relax! It takes time')
#:#print('Creating the rolling Window Dataset for Machine Learning')

#:initialFeaturesLi = [
#:    # These are the initial features followed by the rolled features i.e., 
#:    # the features that will repeat several times. 
#:    # Some of the features are not repeated during the rolling window building (see below
#:    # The features are oranized in groups
#:    # grp variables
#:    'sel', 'grp',
#:    # informative
#:    'imfcode', 'year', 'isocode', 'countryname', 'd_crisis',
#:    'defl_gdp_growth_wdi', 'liquidity_wdi', 'depositr_ifs', 'discountr_ifs',
#:       'defl_gdp_growth_ifs', 'usdexrate_growth_ifs',
#:       'domcrpriv_to_gdp_growth_wdi', 'domcrpriv_growth_wdi',
#:       'real_gdp_growth_wdi', 'real_gdp_growth_ifs', 'lgdp_per_capita_wdi',
#:       'change_nb_tot_wdi', 'm2_to_fereserves_wdi', 'realirate_ifs',
#:       'realirate_wdi', 'lendirate_wdi', 'nfassets_to_gdp_wdi', 'crisis1',
#:       'crisis2', 'crisis3', 'esample', 
#:    # 'crisis0_1' # removed because it will be added by the rolling process
#:    # * Covariates # See cov2roll
#:    # * Target variable 'crisis0', see cov2roll
#:    ]

#:cov2roll=[
#:    'crisis0',
#:    'real_gdp_growth_ifs_1', 'lgdp_per_capita_wdi_1', 
#:    'defl_gdp_growth_wdi_1', 'realirate_ifs_1', 
#:    'm2_to_fereserves_wdi_1', 'domcrpriv_growth_wdi_1', 
#:    'ld_nfassets_to_gdp_wdi_1', 'aux1'
#:]

#:minYear = df.year.min()
#:maxYear = df.year.max() 

#:windowLength=5

def getNewFeatureName(originalFeatureName, windowOffset):
    """
    Generate a new feature (column) name for rolling window datasets.

    This function is used when constructing lagged or offset features 
    for machine learning models. Given the original feature name and 
    a time offset, it returns the appropriate column name.

    - If the offset is 0, the function returns the original feature name 
      (representing the current time period).
    - If the offset is greater than 0, the function prefixes the name with 
      "os{offset}_" to indicate how many steps back in time the value 
      was taken from (e.g., os1_ for a 1-period lag, os2_ for a 2-period lag).

    Parameters
    ----------
    originalFeatureName : str
        The name of the original feature/column.
    windowOffset : int
        The time step offset (0 for current, 1 for one lag, etc.).

    Returns
    -------
    str
        A new feature name that distinguishes lagged values.

    Examples
    --------
    >>> getNewFeatureName("real_gdp_growth_ifs", 0)
    'real_gdp_growth_ifs'
    >>> getNewFeatureName("real_gdp_growth_ifs", 2)
    'os2_real_gdp_growth_ifs'
    """
    if windowOffset==0:
        return originalFeatureName
    newName = f'os{windowOffset}_{originalFeatureName}'
    return newName

#:"""
#:Breif explanation of the following code:

#:- For early years (the first 4 rows), there might not be enough past data to fill the window. So, here's what the code does:
#:      - 1. It searches backward from the current year (`baseYear`) to find up to 5 available past years for that same country(`isocode`).
#:      - 2. If it can not find 5 distinct years, it repeats the last available year until the list has length 5.
#:- Final dataset saved as ``cr03_ml_rolling``.
#:"""

#:df3 = df2.copy(deep=True) # deep copy, so that df3 can be modified independently by df2
#:for ind, row in df2.iterrows():
#:    baseYear = row['year']
#:    isocode = row['isocode']
#:    # print(year, isocode)
#:    availablePastYearsLi = []
#:    cursorYear=baseYear
#:    while cursorYear>=minYear and len(availablePastYearsLi)<windowLength:
#:        # checking if there is another record/row for the given country (identified by the isocode) and the target year (i.e., cursorYear).
#:        if len(df2.loc[(df2.isocode==isocode) & (df2.year==cursorYear), :])>0:
#:            availablePastYearsLi.append(cursorYear)
#:        #else:
#:        #    availablePastYearsLi.append(-1) # i.e., year not available
        
#:        cursorYear-=1
    
#:    # if there are not enough elements for the rolling window, the last (available) year is added several time
#:    while len(availablePastYearsLi)<windowLength:
#:        availablePastYearsLi.append(availablePastYearsLi[-1])

        
#:    if len(availablePastYearsLi)<windowLength:
#:        # Due to the previous while, the execution should no more enter here
#:        # not enough years for the rolling window, dropping the row
#:        df3.drop(index=ind, inplace=True)
#:    else:
#:        offset=0
#:        for cursorYear in availablePastYearsLi:
#:            if cursorYear==-1:
#:                # non existing year
#:                pass
#:            else:
#:                # Creating the new column mames
#:                newFeatureNames = [getNewFeatureName(oldFeatureName, offset) for oldFeatureName in cov2roll]
#:                #extracting the data for the new columns
#:                extractedRow = df2.loc[(df2.isocode==isocode) & (df2.year==cursorYear), cov2roll]
#:                tempCrisis0_1 = df2.loc[(df2.isocode==isocode) & (df2.year==cursorYear), 'crisis0_1']
#:                if extractedRow.shape[0]>1:
#:                    print('*** Problems')
#:                    print('ind', ind)
#:                    print('extractedRow.shape', extractedRow.shape)
#:                extractedRow = extractedRow.values[0] # The result is bidimensional
#:                tempCrisis0_1 = tempCrisis0_1.values[0]
#:                #print(extractedRow)
#:                #print('tempCrisis0_1', tempCrisis0_1)
#:                df3.loc[ind, newFeatureNames]=extractedRow
#:            offset+=1

#:        # Adding the last lagged variable, which otherwise would have been not included
#:        df3.loc[ind, 'crisis0_last']=tempCrisis0_1

#:df3.to_sql('cr03_ml_rolling', con)
#:con.close()
#:print(okMsg)

#:print('balanced_train crisis0 value_counts()')
#:print(df3.loc[df3.grp=='balanced_train', 'crisis0'].value_counts())
#:print(okMsg)

#:print('balanced_test crisis0 value_counts()')
#:print(df3.loc[df3.grp=='balanced_test', 'crisis0'].value_counts())
#:print(okMsg)