# import required packages
import pandas as pd
import numpy as np

# read the data that contains the probability distributions that we need to compute the distances from each other
data = pd.read_csv("./data.csv")

# the function assumes that the data contains the classes labels, and probability of each class
# it means that the data would have as many classes as the reference sample has.
# For uniform distribution, the probability of each class would be simply 1/(n classes)

# enter the name of distributions (probability column)
d1 = 'd1' #(the name of column that contains probability of distribution 1)
d2 = 'd2' #(the name of column that contains probability of distribution 2)

# function to compute the Chi Square Distance
def csd(d1, d2):
    csd = 0.5*(np.sum([((p1-p2)**2)/(p1+p2)
    for (p1,p2) in zip(d1,d2)]))
    return csd

# compute the Chi Square distance between two distributions

csd(data[d1], data[d2])