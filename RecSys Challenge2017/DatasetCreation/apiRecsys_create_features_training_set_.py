#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" create_features_training_set
Usage:
  create_features_training_set.py <foldername>
  create_features_training_set.py -h | --help
  create_features_training_set.py --version

Options:
  -h --help     Show this screen.
  --version
"""

import pandas as pd
import numpy as np
from docopt import docopt

import os
import ast


def import_interactions(foldername):
    interactions = pd.read_pickle(foldername+'interactions123DataAll.pkl')
    interactions = interactions.append(pd.read_pickle(foldername+'interactions4Data.pkl'), ignore_index=False,
                                               verify_integrity=True)
    interactions = interactions.append(pd.read_pickle(foldername+'interactions5Data.pkl'), ignore_index=False,
                                               verify_integrity=True)

    # add interactions 0
    interaction0 = np.load(foldername + 'userItemsDiff9WeekF.npy')
    #interaction0 = interaction0[0:1500000]  # select only a few

    int0 = [interaction0[k].split('-') for k in range(len(interaction0))]
    int0 = pd.DataFrame(int0, columns=['user_id', 'item_id'])
    int0['interaction_type'] = 0

    # I invent the timestamp as we don´t have it in this file. I am forcing a timestamp that will be in week 4
    int0['created_at'] = 1.485576e+09
    int0['created_at_date'] = pd.to_datetime(1.485576e+09, unit='s')

    interactions = interactions.append(int0, ignore_index=True)

    ## Add more 0s
    interaction0 = np.load(foldername + 'userItemsDiff9WeekD.npy')
    #interaction0 = interaction0[0:1500000]  # select only a few

    int0 = [interaction0[k].split('-') for k in range(len(interaction0))]
    int0 = pd.DataFrame(int0, columns=['user_id', 'item_id'])
    int0['interaction_type'] = 0

    # I invent the timestamp as we don´t have it in this file. I am forcing a timestamp that will be in week 4
    int0['created_at'] = 1.485576e+09
    int0['created_at_date'] = pd.to_datetime(1.485576e+09, unit='s')

    interactions = interactions.append(int0, ignore_index=True)

    ## Final casts
    interactions.user_id = interactions.user_id.astype(np.int32)
    interactions.item_id = interactions.item_id.astype(np.int32)
    return interactions


def import_users(filename):
    # Import users with estimated career_levl, industry_id and dicipline_id previously calculated
    users = pd.read_pickle(filename)
    users.columns = ['user_id', 'jobroles', 'career_level', 'discipline_id', 'industry_id', 'country', 'region',
                     'experience_n_entries_class', 'experience_years_experience', 'experience_years_in_current',
                     'edu_degree', 'edu_fieldofstudies', 'wtcj', 'premium', 'est_discipline_id', 'est_industry_id',
                     'est_career_level']
    users.career_level = users.est_career_level
    users.discipline_id = users.est_discipline_id
    users.industry_id = users.est_industry_id
    users.drop(['est_discipline_id', 'est_industry_id','est_career_level'], axis=1, inplace=True)
    return users


def import_items(filename):
    items = pd.read_csv(filename, sep='\t')
    items.columns = ['item_id', 'title', 'career_level', 'discipline_id', 'industry_id', 'country', 'is_payed',
                     'region', 'latitude', 'longitude', 'employment', 'tags', 'created_at']
    items['title']=items['title'].apply(lambda x: [] if pd.isnull(x) else [np.int32(i) for i in x.split(',')])                         
    items['tags']=items['tags'].apply(lambda x: [] if pd.isnull(x) else [np.int32(i) for i in x.split(',')])
    return items


def enrich_users_data(users):

    users.ix[users.country == 'at', 'region'] = 17
    users.ix[users.country == 'ch', 'region'] = 18
    users.ix[users.country == 'non_dach', 'region'] = 19

    users['career_level_user_gt_4'] = users.career_level >= 4
    users['experience_n_entries_gt_3'] = users.experience_n_entries_class >= 3
    users['experience_years_exp_gt_4'] = users.experience_years_experience >= 4
    users['experience_years_current_st_2'] = users.experience_years_in_current <= 2
    users['experience_years_current_gt_5'] = users.experience_years_in_current >= 5
    users['discipline_user1'] = users.discipline_id == 1
    users['discipline_user2'] = users.discipline_id == 2
    users['discipline_user7'] = users.discipline_id == 7
    users['discipline_user17'] = users.discipline_id == 17
    users['industry_user_3'] = users.industry_id == 3
    users['industry_user_7'] = users.industry_id == 7
    users['industry_user_9'] = users.industry_id == 9
    users['industry_user_14'] = users.industry_id == 14
    users['industry_user_16'] = users.industry_id == 16
    users['industry_user_18'] = users.industry_id == 18
    users['industry_user_21'] = users.industry_id == 21
    users['industry_user_23'] = users.industry_id == 23
    users.edu_fieldofstudies.fillna('', inplace=True)
    users['field_of_studies_8'] = users.edu_fieldofstudies.apply(lambda x: '8' in x)

    return users


def enrich_items_data(items):
    items.ix[items.country == 'at', 'region'] = 17
    items.ix[items.country == 'ch', 'region'] = 18
    items.ix[items.country == 'non_dach', 'region'] = 19
    items['career_level_item_gt_4'] = items.career_level >= 4
    items['discipline_item1'] = items.discipline_id == 1
    items['discipline_item2'] = items.discipline_id == 2
    items['discipline_item7'] = items.discipline_id == 7
    items['discipline_item17'] = items.discipline_id == 17
    items['industry_item_9'] = items.industry_id == 9
    items['industry_item_16'] = items.industry_id == 16
    items['industry_item_18'] = items.industry_id == 18
    return items


def length_list(l):
    try:
        length = len(set(l))
    except:
        length = 0
    return length


def intersect_country(x):
    try:
        out = x.country_item in set(x.country_list)
    except:
        out = False
    return out


def intersect_region(x):
    try:
        #out = x.country_item in set(x.country_list)
        out = x.region_item in set(x.region_list)
    except:
        out = False
    return out


def add_features_users_preferred_location_based_on_past_interactions(users, items, interactions):

    data = merge_users_and_items(interactions, users, items)
    subset = data[['user_id', 'country_user', 'region_user', 'country_item', 'region_item', 2, 3]]
    subset = subset[(subset[2]>0) | (subset[3]>0)]

    countryGroupByUser = subset.groupby(['user_id'])['country_item'].unique()#apply(list)
    regionGroupByUser = subset.groupby(['user_id'])['region_item'].unique()#apply(list)
    countryRegionGroupByUser = pd.concat([countryGroupByUser, regionGroupByUser], axis=1)
    print countryRegionGroupByUser.head()
    print "***"
    print countryRegionGroupByUser.shape
    print "***"
    print countryRegionGroupByUser.columns
    print "***"
    print countryRegionGroupByUser.dtypes
    print "***"
    #countryRegionGroupByUser.reset_index(level=0, inplace=True)
    countryRegionGroupByUser.reset_index(inplace=True)
    countryRegionGroupByUser.columns = ['user_id', 'country_list', 'region_list']

    users = users.merge(countryRegionGroupByUser, on='user_id', how='left')

    users['user_searching_more_one_country'] = users.country_list.apply(lambda row: length_list(row) > 1)
    users['user_searching_more_one_region'] = users.region_list.apply(lambda row: length_list(row) > 1)

    users['country_list']=[x if type(x)==type([]) else [] for x in users['country_list']]
    users['region_list']=[x if type(x)==type([]) else [] for x in users['region_list']]

    return users


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
    data['n_jobroles_tags']=[float(len(set(i).intersection(set(j)))) for i,j in zip(data['jobroles'],data['tags'])]
    data['pct_jobroles_title']=[(i/float(len(set(j))) if len(set(j))>0 else 0.0) for i,j in zip(data['n_jobroles_title'],data['jobroles'])]
    data['pct_jobroles_tags']=[(i/float(len(set(j))) if len(set(j))>0 else 0.0) for i,j in zip(data['n_jobroles_tags'],data['jobroles'])]

    # Features that match
    data['discipline_match'] = data['discipline_id_user'] == data['discipline_id_item']
    data['industry_match'] = data['industry_id_user'] == data['industry_id_item']

    # Features that are similar
    data['career_level_nearly_level_match'] = np.abs(data.career_level_user - data.career_level_item) <= 1
    data['career_level_item_st_user'] = data.career_level_item < data.career_level_user

    # Features related to geography
    data = add_features_geography(data)

    return data


def apiRecsyBuild12345UserItemAggInteractions(interactions):
    user_item_int = interactions.groupby(['user_id','item_id','interaction_type']).size().unstack(fill_value=0.0)
    user_item_int.columns.name = None
    user_item_int = user_item_int.reset_index()
    return user_item_int


def merge_users_and_items(interactions, users, items):

    # Group all interactions
    interactions = apiRecsyBuild12345UserItemAggInteractions(interactions)

    # Select all users and items who have been involved with interactions
    idx = users.user_id.isin(interactions.user_id)
    users = users.loc[idx].reset_index(drop=True)
    idx = items.item_id.isin(interactions.item_id)
    items = items.loc[idx].reset_index(drop=True)

    # merge users and items
    data = interactions.merge(users, on='user_id', how='inner')
    data = data.merge(items, on='item_id', how='left', suffixes=('_user', '_item'))

    return data


def create_label(data):
    label = ((data[1] > 0).astype(int) + 5 * ((data[2] > 0) | (data[3] > 0)).astype(int) +
     20 * ((data[5] > 0).astype(int)) - 10 * (((data[4] > 0) & (data[[1, 2, 3, 5]].sum(axis=1) < 1e-5)).astype(int))) * (
        data.premium.astype(int) + 1)
    return label


def add_rankings_for_users(users, items, interactions):

    data = merge_users_and_items(interactions, users, items)
    data['label'] = create_label(data)

    targetUsersRank = data.groupby(['user_id'])['label'].sum().sort_values(ascending=False).to_frame()
    targetUsersRank['norm_rank'] = (targetUsersRank['label'] - targetUsersRank['label'].min()) / (
    targetUsersRank['label'].max() - targetUsersRank['label'].min())

    targetUsersRank['cat_rank'] = '0'
    targetUsersRank.loc[targetUsersRank['label'] > 0.0, 'cat_rank'] = '1'  ## Top users
    targetUsersRank.loc[targetUsersRank['label'] > 10.0, 'cat_rank'] = '2'  ## Top users
    targetUsersRank.loc[targetUsersRank['label'] > 50.0, 'cat_rank'] = '3'  ## Top users
    targetUsersRank.loc[targetUsersRank['label'] > 100.0, 'cat_rank'] = '4'  ## Top users
    targetUsersRank.loc[targetUsersRank['label'] < 0.0, 'cat_rank'] = '-1'  ## Top users
    targetUsersRank.loc[targetUsersRank['label'] < -10.0, 'cat_rank'] = '-2'  ## Top users
    targetUsersRank.loc[targetUsersRank['label'] < -50.0, 'cat_rank'] = '-3'  ## Top users
    targetUsersRank.loc[targetUsersRank['label'] < -100.0, 'cat_rank'] = '-4'  ## Top users
    targetUsersRank['benefit']=targetUsersRank['label']
    targetUsersRank.drop('label',axis=1,inplace=True)

    # Create banned users lists
    banned_users=pd.DataFrame(targetUsersRank[targetUsersRank['benefit']<-9].index.tolist(),columns=['user_id'])
    banned_users.to_pickle('userBannedList_mid.pkl')
    banned_users=pd.DataFrame(targetUsersRank[targetUsersRank['benefit']<-5].index.tolist(),columns=['user_id'])
    banned_users.to_pickle('userBannedList_plus.pkl')
    banned_users=pd.DataFrame(targetUsersRank[targetUsersRank['benefit']<-15].index.tolist(),columns=['user_id'])
    banned_users.to_pickle('userBannedList_minus.pkl')

    ## Add rankings
    users=users.join(targetUsersRank, on='user_id')

    users['cat_rank'].fillna(0,inplace=True)
    users['benefit'].fillna(0,inplace=True)

    ## Add "recent" activity score (very basic)
    intForMaxTS=interactions[(interactions['interaction_type']>0)]
    intForMaxTS=intForMaxTS.groupby(['user_id'])['created_at'].max()
    intForMaxTS=intForMaxTS.to_frame()
    from datetime import date
    intForMaxTS['recent']=[(7*date.fromtimestamp(i).isocalendar()[1]-7+date.fromtimestamp(i).isocalendar()[2]\
                            if date.fromtimestamp(i).isocalendar()[1]<=6 else 0) for i in intForMaxTS['created_at']]

    ## Add rankings
    users=users.join(intForMaxTS['recent'],on='user_id',how='left')
    users['recent'].fillna(0,inplace=True)

    return users


def reduce_interactions(interactions):
    interactions['Week'] = interactions.created_at_date.dt.week
    subset = interactions[(interactions.Week>=2) & (interactions.Week<=4)]
    user_items = subset.groupby(['user_id', 'item_id']).size()
    user_items = user_items.to_frame()
    reduced_interactions = interactions.join(user_items, how='inner', on=['user_id', 'item_id'])
    reduced_interactions.drop(['Week', 0], inplace=True, axis=1)
    reduced_interactions = reduced_interactions[[u'user_id',u'item_id', u'interaction_type']]
    return reduced_interactions


def convert_to_list(x):
    try:
        out = ast.literal_eval(x)
    except:
        out = np.nan
    return out


def create_features_training_set(arg):

    folderName = arg['<foldername>']

    # Import interactions
    print 'Importing interactions...'
    interactions = import_interactions(folderName)
    print 'done'

    if os.path.isfile('enriched_users.pkl') and os.path.isfile('enriched_items.pkl'):
        #users = pd.read_csv('enriched_users.csv', low_memory=False)
        #items = pd.read_csv('enriched_items.csv', low_memory=False)
        users=pd.read_pickle('enriched_users.pkl')
        items=pd.read_pickle('enriched_items.pkl')
        #users.jobroles = users.jobroles.apply(lambda x: convert_to_list(x))
        #users.country_list = users.country_list.apply(lambda x: convert_to_list(x))
        #users.region_list = users.region_list.apply(lambda x: convert_to_list(x))
    else:
        # Import users
        print 'Importing users...'
        users = import_users(folderName + 'users_fill_discipline_industry_career.pkl') #the users have estimated values for missing ones already
        print 'done'

        # Import all items
        print 'Importing items...'
        items = import_items(folderName + 'items.csv')
        print 'done'

        # Enrich users data
        print 'Enriching users...'
        users = enrich_users_data(users)
        print 'done'

        # Enrich items data
        print 'Enriching items...'
        items = enrich_items_data(items)
        print 'done'

        # Enrich users with past interactions
        print 'Enriching users with past...'
        print '...geography...'
        users = add_features_users_preferred_location_based_on_past_interactions(users, items, interactions)
        print '...rankings...'
        users = add_rankings_for_users(users, items, interactions)
        print 'done'

        # Save enriched user and items
        print 'Saving data...'
        #users.to_csv('enriched_users.csv', index=False)
        users.to_pickle('enriched_users.pkl')
        #items.to_csv('enriched_items.csv', index=False)
        items.to_pickle('enriched_items.pkl')
        print 'done'

    # Reduce interactions set. Only keep weeks 2-4
    interactions = reduce_interactions(interactions)

    # merge data sets and find all interactions between user and item
    print 'Merging...'
    data = merge_users_and_items(interactions, users, items)
    print 'done'

    # Add features level matching
    print 'Adding features...'
    data = add_features_interaction(data)
    print 'done'

    # Drop variables not needed
    cols = [u'career_level_user', u'country_user', u'region_user', u'edu_degree', u'career_level_item', u'country_item',
            u'region_item', u'latitude', u'longitude', u'employment', u'title', u'tags', u'created_at', u'industry_id_user',
            u'industry_id_item', u'discipline_id_item', u'discipline_id_user', u'edu_fieldofstudies', u'country_list',
            u'region_list']
    data.drop(cols, axis=1, inplace=True)

    # Create target or label variable
    print 'Creating label...'
    data['label'] = create_label(data)
    print 'done'

    # Save training set
    print 'Saving train set...'
    data.to_csv('train_set.csv', index=False)
    data.to_pickle('train_set.pkl')
    print 'done'


if __name__ == "__main__":
    arguments = docopt(__doc__, version=1.0)
    create_features_training_set(arguments)
