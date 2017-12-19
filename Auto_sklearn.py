import requests
import numpy as np
import scipy as sp
import sys
import platform
import pandas as pd
from time import time
from operator import itemgetter
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
import re
import warnings
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV, ParameterSampler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score,roc_curve,auc,mean_absolute_error
import collections
#import xgboost as xgb
import autosklearn.regression
from tpot import TPOTRegressor
########################################################################################################################
#Springleaf Marketing Response
########################################################################################################################
#--------------------------------------------Algorithm : Random Forest :------------------------------------------------
#Random Forest :
#--------------------------------------------Algorithm : XGB------------------------------------------------------------
#XGB :

#--------------------------------------------Suggestions, Ideas---------------------------------------------------------
#Suggestions, Ideas

#--------------------------------------------with only 7K records-------------------------------------------------------
########################################################################################################################
#Utility function to report best scores
########################################################################################################################
def report(grid_scores, n_top):

    cols_key = []
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]

    for i, score in enumerate(top_scores):
        if( i < 5):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")

        dict1 = collections.OrderedDict(sorted(score.parameters.items()))

        if i==0:
            for key in dict1.keys():
                cols_key.append(key)
            Parms_DF =  pd.DataFrame(columns=cols_key)

        cols_val = []
        for key in dict1.keys():
            cols_val.append(dict1[key])

        Parms_DF.loc[i] =  cols_val

    return Parms_DF

########################################################################################################################
#Cross Validation and model fitting
########################################################################################################################
def Nfold_Cross_Valid(X, y, clf):

    print("***************Starting Nfold Cross validation***************")

    X =np.array(X)
    scores=[]

    #ss = StratifiedShuffleSplit(y, n_iter=5,test_size=0.2, random_state=21, indices=None)
    ss = KFold(len(y), n_folds=5,shuffle=False)

    i = 1

    for trainCV, testCV in ss:
        X_train, X_test = X[trainCV], X[testCV]
        y_train, y_test = y[trainCV], y[testCV]

        clf.fit(X_train, np.log(y_train))
        y_pred = np.exp(clf.predict(X_test))

        scores.append(mean_absolute_error(y_test, y_pred))
        print(" %d-iteration... %s " % (i,scores))

    # Average MAE from cross validation
    scores = np.array(scores)
    print("CV Score:", np.mean(scores))

    print("***************Ending Nfold Cross validation***************")

    return scores

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS,Actual_DS):

    print("***************Starting Data cleansing***************")

    global  Train_DS1

    y = Train_DS.loss.values
    Train_DS = Train_DS.drop(["loss"], axis = 1)

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))
    #-------------------------------------------------------------------------------------------------------------------
    #Devectorize / one hot encoding fro numeric cols

    New_DS = pd.concat([Train_DS, Actual_DS])

    r = re.compile("cat*")
    cat_cols = list(filter(r.match, list(New_DS.columns)))

    # print("Starting dummies.....")
    # for column in cat_cols:
    #     print("dummies for "+ column)
    #     dummies = pd.get_dummies(New_DS[column])
    #     cols_new = [ column+"_"+str(s) for s in list(dummies.columns)]
    #     New_DS[cols_new] = dummies

    print("Starting label encoding.....")
    label_enc_cols = cat_cols
    for i in range(len(label_enc_cols)):
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(New_DS[label_enc_cols[i]].astype(str)))
            New_DS[label_enc_cols[i]] = lbl.transform(New_DS[label_enc_cols[i]])

    #New_DS = New_DS.drop(cat_cols, axis = 1)
    Train_DS = New_DS.head(len(Train_DS))
    Actual_DS = New_DS.tail(len(Actual_DS))

    ####################################################################################################################
    #Shuffle the Dataset
    Train_DS = Train_DS.fillna(0)
    Actual_DS = Actual_DS.fillna(0)

    print("shuffling")
    Train_DS, y = shuffle(Train_DS, y, random_state=21)

    print("stdScaler")
    stdScaler = StandardScaler(with_mean=True, with_std=True)
    stdScaler.fit(Train_DS,y)
    Train_DS = stdScaler.transform(Train_DS)
    Actual_DS = stdScaler.transform(Actual_DS)

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))

    print("***************Ending Data cleansing***************")

    return Train_DS, Actual_DS, y

########################################################################################################################
#Random Forest Classifier (around 80%)
########################################################################################################################
def RFR_Regressor(Train_DS, y, Actual_DS, Sample_DS, Grid):

    print("***************Starting Random Forest Regressor***************")
    t0 = time()



    if Grid:
        #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        # specify parameters and distributions to sample from
        param_dist = {
                      "max_depth": [1, 3, 5,8,10,12,15,20,25,30, None],
                      "max_features": sp_randint(1, 49),
                      "min_samples_split": sp_randint(1, 49),
                      "min_samples_leaf": sp_randint(1, 49),
                      "bootstrap": [True, False]
                     }

        clf = RandomForestRegressor(n_estimators=100)

        # run randomized search
        n_iter_search = 25
        clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, scoring = RMSLE_scorer,cv=10,n_jobs=2)

        #Remove tube_assembly_id after its been used in cross validation
        Train_DS    = np.delete(Train_DS,0, axis = 1)

        start = time()
        clf.fit(Train_DS, y)

        print("RandomizedSearchCV took %.2f seconds for %d candidates"
                " parameter settings." % ((time() - start), n_iter_search))
        report(clf.grid_scores_)

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        print(clf.grid_scores_)
        print(clf.best_score_)
        print(clf.best_params_)
        print(clf.scorer_)
    else:

        #for testing purpose
        #clf = RandomForestRegressor(n_estimators=50,n_jobs=-1)

        ################################################################################################################
        #Auto Sklearn
        #clf = autosklearn.regression.AutoSklearnRegressor()

        #print(clf)

        #y = np.log1p(y)
        #clf.fit(Train_DS, y)

        #print("after training")
        #print(clf)

        #pred_Actual = np.expm1(clf.predict(Actual_DS))
        #print("Actual Model predicted")

        ################################################################################################################
        #tpot
        tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)
        tpot.fit(Train_DS, y)
        tpot.export('tpot_iris_pipeline.py')

        pred_Actual = np.expm1(tpot.score(Actual_DS))
        print("Actual Model predicted")

    #Predict actual model
    #pred_Actual = np.expm1(clf.predict(Actual_DS))
    #print("Actual Model predicted")


    sys.exit(0)

    #Get the predictions for actual data set
    preds = pd.DataFrame(pred_Actual, index=Sample_DS.id.values, columns=Sample_DS.columns[1:])
    preds.to_csv(file_path+'output/Submission_Roshan_RF_2.csv', index_label='id')

    print("***************Ending Random Forest Regressor***************")
    return pred_Actual


########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, Train_DS1, Featimp_DS

    #random.seed(1)

    if(platform.system() == "Windows"):
        file_path = 'C:/Python/Others/data/ACS/'
    else:
        file_path = '/mnt/hgfs/Python/Others/data/ACS/'

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Train_DS      = pd.read_csv(file_path+'train.csv',sep=',',index_col=0)
    Actual_DS     = pd.read_csv(file_path+'test.csv',sep=',',index_col=0)
    Sample_DS     = pd.read_csv(file_path+'sample_submission.csv',sep=',')


    Train_DS      = pd.read_csv(file_path+'train.csv',sep=',', index_col=0,nrows = 5000 ).reset_index(drop=True)
    Actual_DS     = pd.read_csv(file_path+'train.csv',sep=',', index_col=0,nrows = 5000).reset_index(drop=True)

    Train_DS, Actual_DS, y =  Data_Munging(Train_DS,Actual_DS)

    #pred_Actual = XGB_Regressor(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
    pred_Actual  = RFR_Regressor(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
    #pred_Actual  = Misc_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys.argv)