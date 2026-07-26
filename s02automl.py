"""
This script implements an automated machine learning pipeline using the TPOT library to detect economic crises based on macroeconomic indicators.

Overview
========

This project uses TPOT, an AutoML library in Python, to automatically discover the best machine learning pipeline for predicting economic crises (variable crisis0).Instead of manually testing models, preprocessing, and hyperparameters, TPOT applies genetic programming to explore many combinations of models and parameters, and then exports the best pipeline it finds.

The code:
 - Reads a dataset from an SQLite database.
 - Splits the data into training and testing subsets.
 - Runs TPOT to search for the best model pipeline.
 - Saves the results (best model, performance metrics, and parameters) to files for later use.

It supports two types of datasets:

- **Standard**: a balanced dataset with lagged crisis indicators.
- **Rolling window**: includes time-lagged covariates for temporal analysis.

"""

import pandas as pd
import sqlite3 as db
import time 
import pickle

from sklearn.metrics import classification_report

#:###################
#:# Installing the library
#:# The tpot last release has some bugs, installing the previous version
#:#! pip install tpot==0.12.2
#from tpot import TPOTClassifier
#from tpot.config import classifier_config_dict

#from datetime import datetime

#:# Version 0.12.2 of tpot is outdated. Version 1.0.0 was released Wednesday February 26, 2025.


#:# tpot search spaces https://github.com/EpistasisLab/tpot/blob/main/Tutorial/2_Search_Spaces.ipynbv
# less /home/cesarini/python/earlyw/venv3/lib/python3.10/site-packages/tpot/base.py


def log(*var, **kvar):
    """
    **Logging utility Function**
    

    This function writes messages to a log file (`s02automl.log2.txt`) and 
    simultaneously prints them to the console. It is primarily used to 
    document the progress of AutoML runs and record key information about 
    parameters, configurations, and performance.

    **Workflow**
    
    - A timestamp (formatted as YYYY_MM_DD__HH_MM_SS) is prepended to every log entry.
    - Any  arguments (`*var`) are joined by commas and added to the log line.
    - Any keyword arguments (`**kvar`) are appended to the log line as "key:value" pairs,
      separated by commas and preceded by an asterisk ("*").
    - The resulting log line is written to a text file (`s02automl.log2.txt`) and printed.

    Parameters
    ----------
    
    *`var` : list
        Variable-length positional arguments. These will be stringified 
        and concatenated into the log message.
    **`kvar` : dict
        Variable-length keyword arguments. These will be formatted as 
        "key:value" pairs in the log message.
    """
    fn = 's02automl.log.txt'
    with open(fn,'a') as fd:
        line = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        line+=' '
        if var is not None:
            line += ','.join([str(el) for el in var])
        #:if kvar is not None:
        if len(kvar)>0:
            line+=' * '
            line+=','.join([str(k)+':'+str(v) for k,v in kvar.items()])
        line+='\n'
        fd.write(line)
        print(line)

def runAutoML(
        dbfn='ews_data_nomiss.sqlite', # The sqlite file name containing the datasets (both rolling window and not rolling window)
        rolling=False, # True if using the rolling window dataset
        tpotGenerations=100,
        tpotPopulationSize=100,
        tpotScoring='f1_macro',
        tpotCv=5,
        tpotConfigDict=None, # 'TPOT light' # None # 'TPOT sparse' uses a configuration dictionary with a one-hot-encoder and the  operators normally included in TPOT that also support sparse matrices
        # classifier_config_dict_light 
        tpotRandomState=42,
    ):
    """
    **TPOT AutoML Function**

    This function orchestrates the AutoML process using TPOT (Tree-based Pipeline 
    Optimization Tool). It handles dataset loading, feature selection (with or 
    without rolling-window features), model optimization, evaluation, and 
    saving the resulting pipeline code and metadata.

    **Workflow**
    
    1. Select the dataset table:
       - If `rolling=False`, use `cr03_ml` (base covariates).
       - If `rolling=True`, use `cr03_ml_rolling` (base + lagged covariates).
    2. Load dataset from the SQLite database (`dbfn`).
    3. Split into:
       - `dfTrain`, `yTrain` - training features and labels
       - `dfTest`, `yTest` - testing features and labels
    4. Initialize TPOT with the given parameters (`generations`, `population_size`, etc.).
    5. Fit TPOT on the training data and evaluate on test data.
    6. Export the optimal pipeline as a Python script and append metadata.
    7. Save a dictionary report (`dictReport`) including:
       - Dataset information
       - Training configuration
       - Best score achieved
       - Pipeline code
    8. Pickle the report for later reuse.

    Parameters
    ----------
    
    dbfn : str, default='ews_data_nomiss.sqlite'
        SQLite database file containing the dataset tables.
    rolling : bool, default=False
        Whether to use the rolling window dataset (`True`) or the static dataset (`False`).
    tpotGenerations : int, default=100
        Number of evolutionary generations for TPOT to run.
    tpotPopulationSize : int, default=100
        Population size for the genetic programming process.
    tpotScoring : str, default='f1_macro'
        Scoring metric to optimize (e.g., 'accuracy', 'roc_auc', 'f1_macro').
    tpotCv : int, default=5
        Number of folds in cross-validation.
    tpotConfigDict : dict or str, default=None
        Search space of models and preprocessors TPOT can use.
        Can also be 'TPOT light' or 'TPOT sparse'.
    tpotRandomState : int, default=42
        Random seed for reproducibility.

    """

    TimeFormatStr = "%Y_%m_%d___%H_%M_%S"

    # Configuration
    #dbfn='ews_data_nomiss.sqlite'
    #tableName='cr03_ml_rolling' # 'cr03_ml_rolling' 'cr03_ml'
    #tableName='cr03_ml'
    
    log('\n'+'*'*20+'\n'+'*'*20+'\n')
    
    
    #rolling=True

    targetVarName='crisis0'

    baseCovariates = ['real_gdp_growth_ifs_1', 'lgdp_per_capita_wdi_1', 'defl_gdp_growth_wdi_1', 'realirate_ifs_1', 'm2_to_fereserves_wdi_1', 'domcrpriv_growth_wdi_1', 
                'ld_nfassets_to_gdp_wdi_1', 'aux1'   # dropped because the null values
                ]
    
    

    rollingAdditionalCovariates = ['os1_crisis0',
       'os1_real_gdp_growth_ifs_1', 'os1_lgdp_per_capita_wdi_1',
       'os1_defl_gdp_growth_wdi_1', 'os1_realirate_ifs_1',
       'os1_m2_to_fereserves_wdi_1', 'os1_domcrpriv_growth_wdi_1',
       'os1_ld_nfassets_to_gdp_wdi_1', 'os1_aux1', 'os2_crisis0',
       'os2_real_gdp_growth_ifs_1', 'os2_lgdp_per_capita_wdi_1',
       'os2_defl_gdp_growth_wdi_1', 'os2_realirate_ifs_1',
       'os2_m2_to_fereserves_wdi_1', 'os2_domcrpriv_growth_wdi_1',
       'os2_ld_nfassets_to_gdp_wdi_1', 'os2_aux1', 'os3_crisis0',
       'os3_real_gdp_growth_ifs_1', 'os3_lgdp_per_capita_wdi_1',
       'os3_defl_gdp_growth_wdi_1', 'os3_realirate_ifs_1',
       'os3_m2_to_fereserves_wdi_1', 'os3_domcrpriv_growth_wdi_1',
       'os3_ld_nfassets_to_gdp_wdi_1', 'os3_aux1', 'os4_crisis0',
       'os4_real_gdp_growth_ifs_1', 'os4_lgdp_per_capita_wdi_1',
       'os4_defl_gdp_growth_wdi_1', 'os4_realirate_ifs_1',
       'os4_m2_to_fereserves_wdi_1', 'os4_domcrpriv_growth_wdi_1',
       'os4_ld_nfassets_to_gdp_wdi_1', 'os4_aux1',
       'crisis0_last'
    ]

    
    if rolling==True:
        covariates = baseCovariates + rollingAdditionalCovariates
        tableName='cr03_ml_rolling'
    else:
        covariates = baseCovariates + ['crisis0_1']
        tableName='cr03_ml'



     # 'cr03_ml_rolling' 'cr03_ml'

    def readDataset(
            dbfn = dbfn,
            tableName=tableName,
        ):
        conn = db.connect(dbfn)
        query=f'select * from {tableName}'
        df = pd.read_sql(query,conn)
        conn.close()
        return df

    df = readDataset(tableName=tableName)
    if 'index' in df.columns:
        df.drop(columns=['index'], inplace=True)
    #print(df.columns)


    dfTrain = df.loc[df.grp=='balanced_train', covariates]
    dfTest = df.loc[df.grp=='balanced_test', covariates]

    yTrain = df.loc[df.grp=='balanced_train', targetVarName]
    yTest = df.loc[df.grp=='balanced_test', targetVarName]


    # See function vars for config params

    

    
    tpot = TPOTClassifier(
        generations=tpotGenerations, #Number of iterations to the run pipeline optimization process
                    # 2 is to keep the computation time limited.
                    # Better 100 to get optimal (and stable) values.
        population_size=tpotPopulationSize, # default=100 # Number of individuals to retain in the
                            # genetic programming population every generation.
                            # Generally, TPOT will work better when you give it
                            # more individuals with which to optimize the pipeline.
        verbosity=2, # How many explanation output will be printed during simulation
        scoring=tpotScoring, # Metric score to optimize
        cv=tpotCv, # cross-validation, default=5
        n_jobs=-1, # Number of processes to use in parallel for evaluating pipelines
                # during the TPOT optimization process. Default = 1 (1 processor/core)
                # -1 is all available processors
                # -2 is all available processors minus one (useful if you are running tpot
                # on your pc and you want to work on something else during the computation)
        config_dict=tpotConfigDict, # The set of initial classifiers
                                # and preprocessing component evaluated. In this set,
                                # every preprocessing element can handle sparse matrices.
        random_state=tpotRandomState # The seed to make the process repeatable
    )

    startTime = datetime.now()

    dictReport = {
        "dfTrainShape":str(dfTrain.shape),
        "covariates":str(list(dfTrain.columns)),
        'targetVarName':targetVarName,
        "startTime:":startTime.strftime(TimeFormatStr),
        'tpotPopulationSize':tpotPopulationSize,
        'tpotGenerations':tpotGenerations,
        'tpotScoring':tpotScoring,
        'tpotCv':tpotCv,
        'tpotConfigDict':str(tpotConfigDict),
        'tpotRandomState':tpotRandomState,
        "yTest.values":yTest.values,
        #'dfTrain':dfTrain,
        #'dfTest':dfTest
    }

    log(f'dbfn {dbfn}, rolling {rolling}' )
    log(f'str(dictReport)')


    tpot.fit(dfTrain, yTrain)

    endTime=datetime.now()


    duration = str(endTime - startTime)
    dictReport['duration']=duration

    bestScore = tpot.score(dfTest, yTest)
    log('Duration {duration}, Tpot Best Score: ', bestScore)
    dictReport['bestScore']=bestScore

    tpot.export('z_tpot_last_optimal_pipeline.py')
    optimalPipelineFileName = f'z_{startTime.strftime(TimeFormatStr)}__{bestScore}__{tableName}__tpot_optimal'
    tpot.export(optimalPipelineFileName+'.py')

    pipelineCode=''
    with open(optimalPipelineFileName+'.py', 'r') as ppfn:
        pipelineCode = ppfn.read()

    #Appending the tpot params to the optimal pipeline script
    with open(optimalPipelineFileName+'.py', 'a') as ppfn:
        addendum=f'''


"""
{str(dictReport)}

"""
'''
        ppfn.write(addendum)


    dictReport['pipelineCode']=pipelineCode

    # Pickling the results:


    with open(optimalPipelineFileName+'.pkl', 'wb') as file:
        pickle.dump(dictReport, file)

    #print('dictReport')
    #print(dictReport)

    log(f'dictReport saved in {optimalPipelineFileName}.pkl')

    

    """
    Output
    ======


    Generation 100 - Current best internal CV score: 0.843540068008787
                                                                                    
    Best pipeline: BernoulliNB(KNeighborsClassifier(XGBClassifier(BernoulliNB(LinearSVC(input_matrix, C=25.0, dual=True, loss=squared_hinge, penalty=l2, tol=0.01), alpha=0.001, fit_prior=True), learning_rate=1.0, max_depth=1, min_child_weight=15, n_estimators=100, n_jobs=1, subsample=0.8, verbosity=0), n_neighbors=30, p=2, weights=uniform), alpha=0.1, fit_prior=False)
    /home/cesarini/python/earlyw/venv3/lib/python3.10/site-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but LinearSVC was fitted with feature names
    warnings.warn(
    Score:  0.8042582417582418


    """



    """

    Best pipeline: RandomForestClassifier(BernoulliNB(BernoulliNB(SelectPercentile(RandomForestClassifier(input_matrix, bootstrap=False, criterion=gini, max_features=0.7500000000000001, min_samples_leaf=18, min_samples_split=7, n_estimators=100), percentile=94), alpha=0.1, fit_prior=False), alpha=0.1, fit_prior=False), bootstrap=True, criterion=entropy, max_features=0.15000000000000002, min_samples_leaf=20, min_samples_split=8, n_estimators=100)
    /home/cesarini/python/earlyw/venv3/lib/python3.10/site-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
    warnings.warn(
    /home/cesarini/python/earlyw/venv3/lib/python3.10/site-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
    warnings.warn(
    Score:  0.7666666666666666

    """


# AutoML Code is executed in the main because otherwise some issues related to concurrent processing may arise
if __name__=="__main__":
    
    #runAutoML(rolling=False)

    #runAutoML(rolling=False)    
    #runAutoML(rolling=False, tpotConfigDict='TPOT light', tpotGenerations=1000)
    #runAutoML(rolling=True, tpotConfigDict='TPOT light', tpotGenerations=1000)
    #runAutoML(rolling=False, tpotConfigDict=None, tpotGenerations=500)
    #runAutoML(rolling=True, tpotConfigDict=None, tpotGenerations=500)

    selected_models=['sklearn.tree.DecisionTreeClassifier', 'sklearn.naive_bayes.BernoulliNB', 'sklearn.ensemble.ExtraTreesClassifier', 'xgboost.XGBClassifier', 'sklearn.linear_model.SGDClassifier', 'sklearn.svm.LinearSVC', 'sklearn.feature_selection.RFE', 'sklearn.feature_selection.SelectFromModel']
    custom_tpot_dict={ k:classifier_config_dict[k] for k in selected_models}
    runAutoML(rolling=False, tpotConfigDict=custom_tpot_dict, tpotGenerations=100)
    runAutoML(rolling=True, tpotConfigDict=custom_tpot_dict, tpotGenerations=100)

    
    
#:# scp "cesarini@desktop:/home/cesarini/python/earlyw/z_*.py" ./

#:# scp s03automl.py  cesarini@desktop:/home/cesarini/python/earlyw/

#:# from tpot.config import classifier_config_dict
#:# print(classifier_config_dict.keys())
#:#dict_keys(['sklearn.naive_bayes.GaussianNB', 'sklearn.naive_bayes.BernoulliNB', 'sklearn.naive_bayes.MultinomialNB', 'sklearn.tree.DecisionTreeClassifier', 'sklearn.ensemble.ExtraTreesClassifier', 'sklearn.ensemble.RandomForestClassifier', 'sklearn.ensemble.GradientBoostingClassifier', 'sklearn.neighbors.KNeighborsClassifier', 'sklearn.svm.LinearSVC', 'sklearn.linear_model.LogisticRegression', 'xgboost.XGBClassifier', 'sklearn.linear_model.SGDClassifier', 'sklearn.neural_network.MLPClassifier', 'sklearn.preprocessing.Binarizer', 'sklearn.decomposition.FastICA', 'sklearn.cluster.FeatureAgglomeration', 'sklearn.preprocessing.MaxAbsScaler', 'sklearn.preprocessing.MinMaxScaler', 'sklearn.preprocessing.Normalizer', 'sklearn.kernel_approximation.Nystroem', 'sklearn.decomposition.PCA', 'sklearn.preprocessing.PolynomialFeatures', 'sklearn.kernel_approximation.RBFSampler', 'sklearn.preprocessing.RobustScaler', 'sklearn.preprocessing.StandardScaler', 'tpot.builtins.ZeroCount', 'tpot.builtins.OneHotEncoder', 'sklearn.feature_selection.SelectFwe', 'sklearn.feature_selection.SelectPercentile', 'sklearn.feature_selection.VarianceThreshold', 'sklearn.feature_selection.RFE', 'sklearn.feature_selection.SelectFromModel'])
#:#
#:# custom_tpot_dict=custom_config = {
#:#    'sklearn.ensemble.RandomForestClassifier': classifier_config_dict['sklearn.ensemble.RandomForestClassifier'],
#:#    ...
#:#}
#:# 
