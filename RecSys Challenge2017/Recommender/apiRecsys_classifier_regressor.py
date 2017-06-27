import numpy as np
import pandas as pd
import time

MODE="XGB"
#MODE="FOREST"

FEATURE_SET=['career_level_item_gt_4','career_level_item_st_user','career_level_nearly_level_match','career_level_user_gt_4',\
             'cat_rank','country_match','discipline_item1','discipline_item17','discipline_item2','discipline_item7',\
             'discipline_match','discipline_user1','discipline_user17','discipline_user2','discipline_user7',\
             'experience_n_entries_class','experience_n_entries_gt_3','experience_years_current_gt_5',\
             'experience_years_current_st_2','experience_years_exp_gt_4','experience_years_experience',\
             'experience_years_in_current','field_of_studies_8','industry_item_16','industry_item_18','industry_item_9',\
             'industry_match','industry_user_14','industry_user_16','industry_user_18','industry_user_21','industry_user_23',\
             'industry_user_3','industry_user_7','industry_user_9','is_payed','premium','region_match','regions_like',\
             'user_searching_more_one_country', 'user_searching_more_one_region','wtcj',\
             'n_jobroles_tags','n_jobroles_title','pct_jobroles_tags','pct_jobroles_title','recent']

## Load the original targetItems.csv file or equivalent formatted one
## Return the dataframe appropiately adapted
## Note: original file does not have header --> user header=None
def apiRecsysLoadTargetItems(path='datasets/targetItems.csv'):
    targetItems=pd.read_table(path,header=None,dtype=np.int32)
    targetItems.columns=['item_id']
    return targetItems

## Load the original targetUsers.csv file or equivalent formatted one
## Return the dataframe appropiately adapted
## Note: this original file does have header --> user header=0
def apiRecsysLoadTargetUsers(path='datasets/targetUsers.csv'):
    targetUsers=pd.read_table(path,header=0,dtype=np.int32)
    targetUsers.columns=['user_id']
    return targetUsers

## Save the targeted User Data dataframe
def apiRecsySaveTargetedUserData(targetedUserData,path='datasets/targetedUserData.pkl',gzip=False):
    if (gzip):
        targetedUserData.to_pickle(path,'gzip')
    else:
        targetedUserData.to_pickle(path)

## Save the targeted Item Data dataframe
def apiRecsySaveTargetedItemData(targetedItemData,path='datasets/targetedItemData.pkl',gzip=False):
    if (gzip):
        targetedItemData.to_pickle(path,'gzip')
    else:
        targetedItemData.to_pickle(path)

## Load the targeted User Data dataframe
def apiRecsyLoadTargetedUserData(path='datasets/targetedUserData.pkl',gzip=False):
    if (gzip):
        targetedUserData=pd.read_pickle(path)
    else:
        targetedUserData=pd.read_pickle(path)
    return targetedUserData

## Load the targeted Item Data dataframe
def apiRecsyLoadTargetedItemData(path='datasets/targetedItemData.pkl',gzip=False):
    if (gzip):
        targetedItemData=pd.read_pickle(path,'gzip')
    else:
        targetedItemData=pd.read_pickle(path)
    return targetedItemData

######## FUNCTIONS FOR XGB TRAINING MODEL ###################

### required for xgb windows install
def apiRecsysWinPrerequisitesXGB():
    #https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en
    import os
    ## my own computer:
    mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
    ## work laptop:
    mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev0\\mingw64\\bin'
    os.environ['PATH']=mingw_path+';'+os.environ['PATH']

## build XGB Mode (saves internally models). XGB parameters hardcode inside.
## return XGB model
def apiRecsysTrainXGBModel(X,Y):
    import xgboost as xgb
    dataset=xgb.DMatrix(X,label=Y)
    dataset.save_binary("xgb/recsys2017.buffer")
    evallist = [(dataset, 'train')]
    param = {'bst:max_depth': 8, 'bst:eta': 0.01, 'silent': 1, 'objective': 'reg:linear' }
    #param = {'bst:max_depth': 4, 'bst:eta': 0.1, 'silent': 1, 'objective': 'reg:linear' , 'booster': 'gblinear'}
    param['nthread']     = 4
    param['eval_metric'] = 'rmse'
    param['base_score']  = 0.0
    num_round            = 1000
    bst = xgb.train(param, dataset, num_round, evallist)
    bst.save_model('xgb/recsys2017.model')
    return bst

def apiRecsysLoadXGBModel(path="xgb/recsys2017.model"):
    import xgboost as xgb
    bst=xgb.Booster({'nthread':4}) #init model
    bst.load_model("xgb/recsys2017.model") # load data
    return bst

### REPLICATED FUNCTIONS TO REPRODUCE FEATURES AS WERE DONE IN TRAINING SET ###
###############################################################################

def add_features_geography(data):
    # people form DE like regions 1, 2, 3, 18, 9
    # people from AT like 17, 18, 1
    # people from CH like 18,1

    data['regions_like'] = False
    data.loc[data.country_user == 'de', 'regions_like'] = data.region_item.isin([1, 2, 3, 18, 19])
    data.loc[data.country_user == 'at', 'regions_like'] = data.region_item.isin([17, 18, 1])
    data.loc[data.country_user == 'ch', 'regions_like'] = data.region_item.isin([18, 1])
    data.loc[data.country_user == 'non_dach', 'regions_like'] = data.region_item.isin([2, 1, 18, 3, 7, 9, 6])

    #data['country_match'] = data['country_user'] == data['country_item']
    #data.country_match = data.country_match | data.apply(lambda row: intersect_country(row),axis=1)
    data['country_match']=[((i==j) or (j in set(k))) for i,j,k in zip(data['country_user'],data['country_item'],data['country_list'])]

    #data['region_match'] = data['region_user'] == data['region_item']
    #data.region_match = data.region_match | data.apply(lambda row: intersect_region(row),axis=1)
    data['region_match']=[((i==j) or (j in set(k))) for i,j,k in zip(data['region_user'],data['region_item'],data['region_list'])]
    
    return data

# Add features matching
def add_features_interaction(data):

    # Features jobroles intersection
    data['n_jobroles_title']=[float(len(set(i).intersection(set(j)))) for i,j in zip(data['jobroles'],data['title'])]
    
    ## Filter
    data=data[(data['n_jobroles_title']>0.0)].copy()
    
    data['n_jobroles_tags']=[float(len(set(i).intersection(set(j)))) for i,j in zip(data['jobroles'],data['tags'])]
    data['pct_jobroles_title']=[(i/float(len(set(j))) if len(set(j))>0 else 0.0) for i,j in zip(data['n_jobroles_title'],data['jobroles'])]
    data['pct_jobroles_tags']=[(i/float(len(set(j))) if len(set(j))>0 else 0.0) for i,j in zip(data['n_jobroles_tags'],data['jobroles'])]    ## Do the filtering

    ## The only difference is the filter
    #data=data[(data['n_jobroles_title']>0.0) | (data['n_jobroles_tags']>2.0)].copy()
    #data=data[(data['pct_jobroles_title']>0.50) | (data['pct_jobroles_tags']>0.66)].copy()
    #data=data[(data['n_jobroles_title']>0.0)].copy()

    # Features that match
    data['discipline_match'] = data['discipline_id_user'] == data['discipline_id_item']
    data['industry_match'] = data['industry_id_user'] == data['industry_id_item']

    # Features that are similar
    data['career_level_nearly_level_match'] = np.abs(data.career_level_user - data.career_level_item)<=1
    data['career_level_item_st_user'] = data.career_level_item < data.career_level_user

    # Features related to geography
    data = add_features_geography(data)

    return data

##################################################
##################  Workers ######################
##################################################

## Version for Random Forest
def apiRecsysClassifyWorkerFullRandForest(targetItemData,targetUserData,TH,outputFile,funcInteractionFeaturesFull):
    #start timer
    t_start = time.time()

    from sklearn.externals import joblib
    model=joblib.load('clf/recsys_randforest_model.pkl')

    #end timer
    t_end=time.time()
    print "CLF Model reload -- Time invested "+str((t_end-t_start))+" s"

    with open(outputFile,'w') as fp:
        pos=0
        average_score=0.0
        num_evaluated=0.0
        for ind in targetItemData.index:
            itemUsers=pd.merge(targetItemData[targetItemData.index==ind],targetUserData,on='_joinkey',suffixes=['_item','_user'])  
            featData=funcInteractionFeaturesFull(itemUsers)
            if (len(featData))>0:
                ids=featData['user_id'].tolist()
                ## Respect the "Sorted"
                X=featData[sorted(FEATURE_SET)]
                ypred=model.predict(X)
                # compute average score
                average_score += sum(ypred)
                num_evaluated += float(len(ypred))
                # use all items with a score above the given threshold and sort the result
                user_ids = sorted(
                    [
                        (ids_j, ypred_j) for ypred_j, ids_j in zip(ypred, ids) if ypred_j > TH
                    ],
                    key = lambda x: -x[1]
                )[0:100]                                                        
                # write the results to file
                if len(user_ids) > 0:            
                    item_id = str(targetItemData['item_id'][ind]) + "\t"
                    fp.write(item_id)
                    for j in range(0, len(user_ids)-1):
                        user_id = str(user_ids[j][0]) + ","
                        fp.write(user_id)
                    user_id = str(user_ids[-1][0]) + "\n"
                    fp.write(user_id)
                    fp.flush()         
            # Every 100 iterations print some stats
            if pos % 100 == 0:
                try:
                    score = str(average_score / num_evaluated)
                except ZeroDivisionError:
                    score = '0'
                percentageDown = str(pos / float(len(targetItemData)))
                print(outputFile+" "+percentageDown+" "+score+" "+str(num_evaluated))
            pos += 1  
    fp.close()

## Version for XGB
def apiRecsysClassifyWorkerFullXGB(targetItemData,targetUserData,TH,outputFile,funcInteractionFeaturesFull,model):
    import xgboost as xgb
    with open(outputFile,'w') as fp:
        pos=0
        average_score=0.0
        num_evaluated=0.0
        for ind in targetItemData.index:
            itemUsers=pd.merge(targetItemData[targetItemData.index==ind],targetUserData,on='_joinkey',suffixes=['_item','_user'])  
            featData=funcInteractionFeaturesFull(itemUsers)
            if (len(featData))>0:
                ids=featData['user_id'].tolist()
                ## Respect the "Sorted"
                ## Respect the "Sorted"
                X=featData[sorted(FEATURE_SET)]
                dtest=xgb.DMatrix(X)
                ypred=model.predict(dtest)
                # compute average score
                average_score += sum(ypred)
                num_evaluated += float(len(ypred))
                # use all items with a score above the given threshold and sort the result
                user_ids = sorted(
                    [
                        (ids_j, ypred_j) for ypred_j, ids_j in zip(ypred, ids) if ypred_j > TH
                    ],
                    key = lambda x: -x[1]
                )[0:100]                                                        
                # write the results to file
                if len(user_ids) > 0:            
                    item_id = str(targetItemData['item_id'][ind]) + "\t"
                    fp.write(item_id)
                    for j in range(0, len(user_ids)-1):
                        user_id = str(user_ids[j][0]) + ","
                        fp.write(user_id)
                    user_id = str(user_ids[-1][0]) + "\n"
                    fp.write(user_id)
                    fp.flush()         
            # Every 100 iterations print some stats
            if pos % 100 == 0:
                try:
                    score = str(average_score / num_evaluated)
                except ZeroDivisionError:
                    score = '0'
                percentageDown = str(pos / float(len(targetItemData)))
                print(outputFile + " " + percentageDown + " " + score+" "+str(num_evaluated))
            pos += 1  
    fp.close()

##################################################
##################  MAIN  ########################
##################################################

def loadPrepareDatasets():

    ### USERS ###

    # start timer
    t_start=time.time() 
    # load user data
    userData=pd.read_pickle('enriched_users.pkl')
    userData['cat_rank']=userData['cat_rank'].astype(float)
    # end timer
    t_end=time.time()
    print "Load enriched users -- Time invested "+str((t_end-t_start))+" s"
    print userData.shape
    print userData.columns
    userData.head()

    ### ITEMS ###

    # start timer
    t_start=time.time()
    # load item data
    itemData=pd.read_pickle('enriched_items.pkl')
    # end timer
    t_end=time.time()
    print "Load enriched items -- Time invested "+str((t_end-t_start))+" s"
    print itemData.shape
    print itemData.columns
    itemData.head()

    ### TRAINING SET ###

    # start timer
    t_start=time.time()
    # load train set
    featuresData=pd.read_pickle('train_set.pkl') 
    featuresData['cat_rank']=featuresData['cat_rank'].astype(float)
    #end timer
    t_end=time.time()
    print "Load train set -- Time invested "+str((t_end-t_start))+" s"
    print featuresData.shape
    print sorted(featuresData.columns)
    featuresData.head()

    ### TARGET USERS ###

    #start timer
    t_start = time.time()

    targetUsers=apiRecsysLoadTargetUsers()

    #end timer
    t_end=time.time()
    print "Load Target Users -- Time invested "+str((t_end-t_start))+" s"

    # Show structure
    print targetUsers.shape

    ### TARGET ITEMS ###

    #start timer
    t_start = time.time()

    targetItems=apiRecsysLoadTargetItems()

    #end timer
    t_end=time.time()
    print "Load Target Items -- Time invested "+str((t_end-t_start))+" s"

    # Show structure
    print targetItems.shape

    ## Create targeted Item/Users datasets
    print userData.shape
    targetUserData=targetUsers.join(userData.set_index(['user_id']),on="user_id")
    print targetUserData.shape
    print itemData.shape
    targetItemData=targetItems.join(itemData.set_index(['item_id']),on="item_id")
    print targetItemData.shape

    ## Apply banned user list
    print "loading banned users"
    #bannedUsers=pd.read_pickle('datasets/userBannedList.pkl')
    bannedUsers=pd.read_pickle('userBannedList_plus.pkl')
    print bannedUsers.shape
    bannedUsers['banned']=1
    bannedUsers=bannedUsers.set_index(['user_id'],drop=True)
    print targetUserData.shape
    targetUserData=targetUserData.join(bannedUsers,on="user_id")
    print targetUserData.shape
    ##targetUserData[targetUserData['banned']==1]
    targetUserData=targetUserData[targetUserData['banned']!=1]
    print targetUserData.shape

    ## Save for latter use or resume
    apiRecsySaveTargetedUserData(targetUserData)
    apiRecsySaveTargetedItemData(targetItemData)  

    # We return the only needed dataset (the targeted ones we're saved)
    return featuresData


def buildRandomForestModel(featuresData):

    print "Building CLF Model"

    #start timer
    t_start=time.time()

    ### Build X and Y sets for training
    X=featuresData[sorted(FEATURE_SET)]
    Y=featuresData['label']

    ### try sklearn
    from sklearn import tree
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor)

    #start timer
    t_start=time.time()

    ## We use 70% as we're going to use later to predict (so use most of data)
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

    # Build & Fit model
    clf = RandomForestClassifier(n_estimators=40,n_jobs=4)
    clf = clf.fit(X_train, y_train)

    # test predict 
    y_predict=clf.predict(X_test)

    #end timer
    t_end=time.time()
    print "Time invested "+str((t_end-t_start))+" s"

    print float(sum(y_predict-y_test==0))/len(X_test)*100

    from sklearn.metrics import accuracy_score
    print accuracy_score(y_test,y_predict)

    #start timer
    t_start=time.time()

    ## save model
    from sklearn.externals import joblib
    joblib.dump(clf,'clf/recsys_randforest_model.pkl',compress=1) 

    #end timer
    t_end=time.time()
    print "CLF Model built & saved -- Time invested "+str((t_end-t_start))+" s"


def buildXGBModel(featuresData):

    print "Building XGB Model"

    #start timer
    t_start=time.time()

    ### Build X and Y sets for training
    X=featuresData[sorted(FEATURE_SET)]
    Y=featuresData['label']

    apiRecsysWinPrerequisitesXGB()
    bst=apiRecsysTrainXGBModel(X,Y)

    #end timer
    t_end=time.time()
    print "XGB Model built & saved -- Time invested "+str((t_end-t_start))+" s"


def resumeToPredictionRandomForest():
    #We need to create "target" versions of items & users (to reduce dataframes to target objects)
    #And we add "_joinkey" column in bot new "target" dataframe for later helping on cartesian product
    ## We assume were created before, so Reload targeted users/items  (to test are saved correctly)
    #targetUserData=apiRecsyLoadTargetedUserData()
    targetUserData=apiRecsyLoadTargetedUserData()
    targetItemData=apiRecsyLoadTargetedItemData()
    print targetUserData.shape
    print targetItemData.shape

    ## Prepare vectors for cartesian product
    targetUserData['_joinkey']=1
    targetItemData['_joinkey']=1

    ### DO MULTOPROCESSING PREDICTION
    import multiprocessing

    ## Select function to use (full one-shot mode)
    funcInteractionFeaturesFull=add_features_interaction
    print targetUserData.shape

    #targetItemData=targetItemData[0:4600]

    #start timer
    t_start=time.time()

    N_WORKERS=4
    TH=0.5
    bucket_size=len(targetItemData)/N_WORKERS
    start= 0
    jobs=[]
    for i in range(0,N_WORKERS):
        stop=int(min(len(targetItemData),start+bucket_size))
        filename="clf/solution_"+str(i)+".csv"
        process=multiprocessing.Process(target=apiRecsysClassifyWorkerFullRandForest,args=(targetItemData[start:stop],targetUserData,TH,filename,funcInteractionFeaturesFull))
        jobs.append(process)
        start = stop
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

    #end timer
    t_end=time.time()
    print "Full prediction done -- Time invested "+str((t_end-t_start))+" s"


def resumeToPredictionXGB():
    #We need to create "target" versions of items & users (to reduce dataframes to target objects)
    #And we add "_joinkey" column in bot new "target" dataframe for later helping on cartesian product
    ## We assume were created before, so Reload targeted users/items  (to test are saved correctly)
    #targetUserData=apiRecsyLoadTargetedUserData()
    targetUserData=apiRecsyLoadTargetedUserData()
    targetItemData=apiRecsyLoadTargetedItemData()
    print targetUserData.shape
    print targetItemData.shape

    ## Prepare vectors for cartesian product
    targetUserData['_joinkey']=1
    targetItemData['_joinkey']=1

    ### DO MULTOPROCESSING PREDICTION
    import multiprocessing

    ## Select function to use (full one-shot mode)
    funcInteractionFeaturesFull=add_features_interaction
    print targetUserData.shape

    #start timer
    t_start = time.time()

    bst=apiRecsysLoadXGBModel()

    #end timer
    t_end=time.time()
    print "XGB Model reload -- Time invested "+str((t_end-t_start))+" s"

    #targetItemData=targetItemData[0:4605]

    #start timer
    t_start = time.time()

    N_WORKERS=4
    TH=-2.5
    bucket_size=len(targetItemData)/N_WORKERS
    start=0
    jobs=[]
    for i in range(0,N_WORKERS):
        stop=int(min(len(targetItemData),start+bucket_size))
        filename="xgb/solution_"+str(i)+".csv"
        process=multiprocessing.Process(target=apiRecsysClassifyWorkerFullXGB,args=(targetItemData[start:stop],targetUserData,TH,filename,funcInteractionFeaturesFull,bst))
        jobs.append(process)
        start = stop
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

    #end timer
    t_end=time.time()
    print "Full prediction done -- Time invested "+str((t_end-t_start))+" s"


if __name__ == "__main__":
    print "Begin prediction task"
    featuresData=loadPrepareDatasets()
    if MODE=="XGB":
        buildXGBModel(featuresData)
        resumeToPredictionXGB()
    else:
        buildRandomForestModel(featuresData)
        resumeToPredictionRandomForest()
    print "End prediction task"

## THEN DO:  copy /b solution_?.csv total_solution.csv
    