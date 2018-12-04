# -*- coding: utf-8 -*-
"""
@author: Brad Pershon
Data Mining
HW #5
K-Means
"""

from __future__ import division
from random import randint
import pandas as pd
import numpy as np
import util
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None #default='warn'

input_wine = "wine.csv"
input_hard = "TwoDimHard.csv"
output_wine = "wine_output.csv"
output_hard = "hard_output.csv"
wine_df = pd.read_csv(input_wine, header = 0)
hard_df = pd.read_csv(input_hard, header = 0)
end_of_wine = 1599
end_of_hard = 400

#WINE PREPROCESSING
#Remove label and dependent variable
wine_label = wine_df['quality']
del wine_df['quality']
del wine_df['class']
#Normalize data
#remove ID field to normalize and then replace
wID = wine_df['ID']
del wine_df['ID']
wine_df = (wine_df - wine_df.min()) / (wine_df.max() - wine_df.min())
wine_df.insert(loc=0, column='ID', value = wID)
del wID

#HARD PREPROCESSING
#remove label
hard_label = hard_df['cluster']
del hard_df['cluster']
cID = wine_df['ID']
del hard_df['ID']
hard_df = (hard_df - hard_df.min()) / (hard_df.max() - hard_df.min())
hard_df.insert(loc=0, column='ID', value = cID)
del cID

#get user input for K
k = int(raw_input("Please enter the value of k(enter 0 for default): "))
if k == 0: 
    k = 4
print "K value: ", k

#Create K random centroids
wine_headers = list(wine_df.columns.values)
wine_cent = pd.DataFrame(columns = wine_headers)
hard_headers = list(hard_df.columns.values)
hard_cent = pd.DataFrame(columns = hard_headers)
for i in range(k):
    w_arr = np.random.ranf(len(wine_headers))
    w_arr[0] = i + 1
    wine_cent.loc[i] = w_arr
    h_arr = np.random.ranf(len(hard_headers))
    h_arr[0] = i + 1
    hard_cent.loc[i] = h_arr
del w_arr
del h_arr
wine_cent['ID'] = wine_cent['ID'].astype(int)
hard_cent['ID'] = hard_cent['ID'].astype(int)    

wine_df['cluster'] = 0
hard_df['cluster'] = 0

#K-Means algorithm
it = 1; #REMOVEME
while(True):
    it = it + 1 #REMOVEME    
    old_wine_cents = wine_cent.values.flatten().tolist()
    old_hard_cents = hard_cent.values.flatten().tolist()
    #Data assignment step
    for i in range(end_of_wine):
        wine_df.loc[i, 'cluster'] = util.calc_cluster(wine_cent, wine_df.loc[i], len(wine_headers))
        if(i < end_of_hard):
            hard_df.loc[i, 'cluster'] = util.calc_cluster(hard_cent, hard_df.loc[i], len(hard_headers))    
    
    #Centroid update step
    for i in range(k):
        clust_id = i + 1
        #Wine
        cent = wine_df.loc[wine_df.cluster == clust_id]
        cent = cent.mean()
        #If cent gets a NaN, select random point
        if(cent.isnull().values.any()):
            r = randint(0, end_of_wine)
            cent = wine_df.loc[r]
        wine_cent.loc[i] = cent
        wine_cent.loc[i, 'ID'] = clust_id
        #Hard
        cent = hard_df.loc[hard_df.cluster == clust_id]
        cent = cent.mean()
        #If cent gets a NaN, select random point
        if(cent.isnull().values.any()):
            r = randint(0, end_of_hard)
            cent = hard_df.loc[r]
        hard_cent.loc[i] = cent
        hard_cent.loc[i, 'ID'] = clust_id
        
    #If centroids have converged, exit K-Means
    new_wine_cents = wine_cent.values.flatten().tolist()
    new_hard_cents = hard_cent.values.flatten().tolist()
    print 'Iteration #: ', it
    if((old_wine_cents == new_wine_cents) and (old_hard_cents == new_hard_cents)):
        print '# of iterations to converge: ', it
        print ""
        break
#Extract ID and cluster #
wine_output = wine_df[['ID', 'cluster']]
hard_output = hard_df[['ID', 'cluster']]

#OUTPUT

#Output to csv files as specified in spec
wine_output.to_csv(output_wine)
hard_output.to_csv(output_hard)

#Compute SSE and SSB for hard
hard_sse = []
hard_ssb = []
real_labels = []
assigned_labels = []
hard_mean = hard_df[['X.1', 'X.2']]
hard_mean = hard_mean.mean()
for i in range(k):
    clust_id = i + 1
    #SSE
    points = hard_df.loc[hard_df.cluster == clust_id]
    cent = hard_cent.loc[i]
    diff = hard_cent.loc[i][1:3] - points[['X.1', 'X.2']]
    diff = diff.values.flatten().tolist()
    diff = util.square(diff)
    hard_sse.append(sum(diff))
    print "Cluster: ", i, " SSE: ", hard_sse[i]
    #Cross tabulation
    assigned_labels.append(len(points))
    real_labels.append(len(hard_label.loc[hard_label == clust_id]))
    #SSB
    diff = hard_cent.loc[i][1:3] - hard_mean
    diff = diff.values.flatten().tolist()
    diff = util.square(diff) 
    diff = sum(diff)
    diff = len(points) * diff
    hard_ssb.append(diff)
print "Assigned Total SSE: ", sum(hard_sse)
print "Assigned SSB: ", sum(hard_ssb)
SST = sum(hard_sse) + sum(hard_ssb)

#Scatter plot
print ""
print "Original data scatter plot:"
plt.scatter(hard_df['X.1'], hard_df['X.2'], c=hard_label)
print ""
print "Assigned data scatter plot:"
plt.scatter(hard_df['X.1'], hard_df['X.2'], c=hard_df['cluster'])

# Cross tabulatoin between actual and assigned clusters
print ""
print "             Real Label  Assigned Label"
for i in range(len(real_labels)):
    print "Cluster ", i, "    ", real_labels[i], "           ", assigned_labels[i]   


#Calculate the True SSE and SSB
print ""
del hard_df['cluster']
hard_df.insert(loc=3, column='cluster', value = hard_label)
hard_sse = []
hard_mean = hard_df[['X.1', 'X.2']]
hard_mean = hard_mean.mean()
for i in range(k):
    clust_id = i + 1
    #SSE
    points = hard_df.loc[hard_df.cluster == clust_id]
    cent = points.mean()
    diff = cent[['X.1', 'X.2']] - points[['X.1', 'X.2']]
    diff = diff.values.flatten().tolist()
    diff = util.square(diff)
    hard_sse.append(sum(diff))
    print "Cluster: ", i, " SSE: ", hard_sse[i]
print "Real Total SSE: ", sum(hard_sse)
print "Real SSB: ", (SST - sum(hard_sse))


#Compute SSE and SSB for wine
print ""
print "Wine: "
wine_sse = []
wine_ssb = []
del wine_df['ID']
wine_mean = wine_df.mean()
for i in range(k):
    clust_id = i + 1
    #SSE
    points = wine_df.loc[wine_df.cluster == clust_id]
    cent = wine_cent.loc[i]
    diff = wine_cent.loc[i][1:12] - points
    diff = diff.values.flatten().tolist()
    diff = util.square(diff)
    wine_sse.append(sum(diff))
    print "Cluster: ", i, " SSE: ", hard_sse[i]
    #SSB
    diff = wine_cent.loc[i][1:12] - wine_mean
    diff = diff.values.flatten().tolist()
    diff = util.square(diff) 
    diff = sum(diff)
    diff = len(points) * diff
    wine_ssb.append(diff)
print "Assigned Total SSE: ", sum(hard_sse)
print "Assigned SSB: ", sum(hard_ssb)
    