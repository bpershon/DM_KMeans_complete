# -*- coding: utf-8 -*-
"""
@author: Brad Pershon
Data Mining
HW #5
"""
import math

def calc_cluster(centroids, entry, size):
    cluster = 0;
    dist = []
    for i in range(len(centroids)):
        #Calculate L2 distance of measurable attributes
        diff = entry[1:size] - centroids.loc[i][1:size]
        diff = diff.values.flatten().tolist()
        diff = square(diff)
        dist.append(math.sqrt(sum(diff)))
    #Return closest centroids ID
    cluster = dist.index(min(dist)) + 1  
    return cluster

def square(list):
    return [i ** 2 for i in list]