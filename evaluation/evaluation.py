from random import random
import networkx as nx
import numpy as np
import argparse
import time
import os
import sys
import scipy.sparse as sp
from sklearn.metrics import jaccard_score
try:
    import cPickle as pickle
except ImportError:
    import pickle
from scipy.sparse import csr_matrix


def check_with_identity(mb):
    count = 0
    for i in range(len(mb)):
        if i == mb[i]-1:
            count = count + 1
    return count/len(mb)


def accuracy(gma, gmb, mb, ma):
    nodes = len(gma)
    count = 0
    for i in range(nodes):
        if ma[i] == gma[i]:
            if (gmb[i]) == (mb[i]):
                count = count + 1
        else:
            print("mistake", ma[i], gma[i])
    print(count)
    return count / nodes


def accuracydiff(gma, gmb, mb, ma):
    nodes = len(ma) - 1
    nodes1 = len(gma)
    count = 0
    j = 0
    i = 0
    while i < nodes:
        if (ma[i] == gma[j]):
            if (gmb[j]) == (mb[i]):
                count = count + 1
                j = j+1
            i = i+1
        else:
            j = j+1
            # print("mistake", ma[i], gma[i])
    print(count)
    return count / nodes, count/nodes1


def accuracy2(gmb, mb):
    nodes = len(gmb)
    count = 0
    for i in range(nodes):
        print(gmb[i], mb[i])
        if (gmb[i]) == (mb[i]):
            count = count + 1
    print(count)
    return count / nodes


def split(Matching):
    Tempxx = (Matching[0])
    dd = len(Tempxx)

    split1 = np.zeros(len(Tempxx), int)
    split2 = np.zeros(len(Tempxx), int)
    for i in range(dd):
        tempMatching = Tempxx.pop()
        split1[i] = int(tempMatching[0])
        split2[i] = int(tempMatching[1])
    return split1, split2


def transformRAtoNormalALign(alignment_matrix):

    n_nodes = alignment_matrix.shape[0]
    sorted_indices = np.argsort(alignment_matrix)

    mb = np.zeros(n_nodes)
    for node_index in range(n_nodes):
        target_alignment = node_index
        row, possible_alignments, possible_values = sp.find(
            alignment_matrix[node_index])
        node_sorted_indices = possible_alignments[possible_values.argsort()]
        mb[node_index] = node_sorted_indices[-1:]
    np.savetxt("results/matching.txt",
               np.vstack((range(n_nodes), mb)).T, fmt="%i")
    mar = range(0, len(mb))
    return mar, mb

#works
def S3(A,B,mb):
    A1=np.sum(A,0)
    B1=np.sum(B,0)
    EdA1=np.sum(A1)
    EdB1=np.sum(B1)
    Ce=0
    source=0
    target=0
    res=0
    for i in range(len(mb)):
        source=A1[i]
        target=B1[mb[i]]
        if source==target:#equality goes in either of the cases below, different case for...
            Ce=Ce+source
        elif source<target:
            Ce=Ce+source
        elif source>target:
            Ce=Ce+target
    div=EdA1+EdB1-Ce
    res=Ce/div
    return res    
#works
def ICorS3GT(A,B,mb,gmb,IC):
    A1=np.sum(A,0)
    B1=np.sum(B,0)
    EdA1=np.sum(A1)
    EdB1=np.sum(B1)
    Ce=0
    source=0
    target=0
    res=0
    for i in range(len(mb)):
        if (gmb[i]==mb[i]):
            source=A1[i]
            target=B1[mb[i]]
            if source==target: #equality goes in either of the cases below, different case for...
                Ce=Ce+source
            elif source<target:
                Ce=Ce+source
            elif source>target:
                Ce=Ce+target
    if IC==True:
        res=Ce/EdA1
    else:
        div=EdA1+EdB1-Ce
        res=Ce/div
    return res  

def permute(B,mb):
    S=np.zeros(len(mb))
    for i in range(len(mb)):
        S[mb[i]]=i
    print(S)
    S.astype(int)
    k=S[B]
    print(k)
    return k
#needs correct perm
def jacardSim(A,B1,mb):
    B = mb[B1]
    #B=permute(B1,mb)
    print(B)
    print(A)
    #B.astype(int)
    JI=0
    for i in range(len(mb)):
        #print(jaccard_score(A[i,:],B[mb[i],:]))
        #JI=JI+jaccard_score(A[i,:],B[mb[i],:])
        JI=JI+jaccard_score(A[:,i],B[:,i],average='macro')
    return JI/len(mb)

#needs correct perm
def jacardSimSepCm(A,B1,mb,gmb):
    #B=permute(B1,mb)
    B=mb[B1]
    B.astype(int)
    JIC=0
    JIW=0
    countercorr=0
    counterwrong=0
    for i in range(len(mb)):
        if ( mb[i]==gmb[i]):
            #JIC=JIC+jaccard_score(A[:,i],B[:,mb[i]])
            JIC=JIC+jaccard_score(A[i,:],B[i,:],average='weighted')
            countercorr=countercorr+1
        else:
            #JIW=JIW+jaccard_score(A[:,i],B[:,mb[i]])
            JIW=JIW+jaccard_score(A[i,:],B[i,:],average='weighted')
            counterwrong=counterwrong+1
    if countercorr==0:
        countercorr=1
    if counterwrong==0:
        counterwrong=1
    return JIC/countercorr,JIW/counterwrong