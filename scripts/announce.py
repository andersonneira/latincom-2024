#!/usr/bin/env python
# coding: utf-8
#https://pypi.org/project/sklearn-som/  
#https://github.com/rileypsmith/sklearn-som/blob/main/sklearn_som/som.py
from statistics import *
from sklearn.preprocessing import *
from sklearn.mixture import GaussianMixture
from sklearn_som.som import SOM
from sklearn.cluster import *
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import BisectingKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from s_dbw import S_Dbw
from cdbw import CDbw
#https://pypi.org/project/s-dbw/
#https://pypi.org/project/cdbw/
from sklearn import metrics
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

'''
data: dados para clusterizar

cluster_validity_metric: define qual métrica será usada para identificar o melhor algoritmo (Davies-Bouldin Index, Calinski-Harabasz Index, Silhouette Coefficient)
    opcoes: sdbw, cdbw, silhouette_score or calinski_harabasz_score, avg (default)

number_clusters: o usuário pode de limitar a busca por quantidade de cluster
    2 (default)
    None testa vários
'''
def announce(data, cluster_validity_metric = 'avg', number_clusters = 2):
    return selection_mode_default(data, number_clusters, cluster_validity_metric)
    
def selection_mode_default(data, number_clusters, cluster_validity_metric):
    resultado = []
    algos = algorithms(number_clusters, data.shape[1])

    for algo in algos:
        score = sdbw = cdbw = kh = db = ss = 0
        algo_str = ''

        if("SOM" in str(algo)):
            fd = format_data(data)
            labels = algo.fit_predict(fd)
            algo_str = "SOM(m="+str(algo.m)+", n="+str(algo.n)+", dim="+str(algo.dim)+ ", random_state=0)"
        elif ('GaussianMixture' in str(algo)):
            algo.fit(data)
            labels = algo.predict(data)
            algo_str = str(algo)
        else:
            algo.fit(data)
            labels = algo.labels_
            algo_str = str(algo)

        if cluster_validity_metric == 'calinski_harabasz_score':
            score = compute_calinski_harabasz_score(data, labels)

        if cluster_validity_metric == 'silhouette_score':
            score = compute_silhouette_score(data, labels)
        
        if cluster_validity_metric == 'davies_bouldin_score':
            score = compute_davies_bouldin_score(data, labels)

        if cluster_validity_metric == 'sdbw':
            score = compute_sdbw(data, labels)

        if cluster_validity_metric == 'cdbw':
            score = compute_cdbw(data, labels)

        if cluster_validity_metric == 'avg':
            kh = compute_calinski_harabasz_score(data, labels)
            ss = compute_silhouette_score(data, labels)
            db = compute_davies_bouldin_score(data, labels)
            sdbw = compute_sdbw(data, labels)
            cdbw = compute_cdbw(data, labels)
                       
        if(cluster_validity_metric == 'avg'):
            resultado.append((algo_str, kh, ss, db, sdbw, cdbw))
        else:
            resultado.append((algo_str, score))
                

    if(cluster_validity_metric == 'avg'):
        temp = pd.DataFrame(resultado, columns=['Algorithm','Calinski_harabasz_score',
                                                'Silhouette_score', 'Davies_bouldin_score',
                                                'SDBW', 'CDBW'])  
        colunas_minmax = ['Calinski_harabasz_score','Silhouette_score', 'Davies_bouldin_score',
                          'SDBW', 'CDBW']
        
        scaler = MinMaxScaler()
        scaler.fit(temp[colunas_minmax])
        temp2 = pd.DataFrame(data=scaler.transform(temp[colunas_minmax]),
                     index=[i for i in range(temp.shape[0])],
                     columns=colunas_minmax)
        df =  pd.concat([temp, temp2], axis=1).reindex(temp.index)
        df.columns =['Algorithm','Calinski_harabasz_score',
                                                'Silhouette_score', 'Davies_bouldin_score',
                                                'SDBW', 'CDBW',
                                                'Calinski_harabasz_score_minmax',
                                                'Silhouette_score_minmax',
                                                'Davies_bouldin_score_minmax',
                                                'SDBW_minmax', 'CDBW_minmax']

        
        norm_db = df[['Davies_bouldin_score_minmax']].apply(lambda x:  1 - x,axis=1)
        norm_sd = df[['SDBW_minmax']].apply(lambda x:  1 - x,axis=1)
        
        df1 =  pd.concat([df, norm_db, norm_sd], axis=1).reindex(df.index)
        
        df1.columns=['Algorithm','Calinski_harabasz_score',
                                                'Silhouette_score', 'Davies_bouldin_score',
                                                'SDBW','CDBW',
                                                'Calinski_harabasz_score_minmax',
                                                'Silhouette_score_minmax',
                                                'Davies_bouldin_score_minmax',
                                                'SDBW_minmax', 'CDBW_minmax',
                                                'Davies_bouldin_score_minmax_normalizado',
                                                'SDBW_minmax_normalizado']
            
        df1 =  df1[['Algorithm','Calinski_harabasz_score',
                                                'Silhouette_score', 'Davies_bouldin_score',
                                                'SDBW','CDBW',
                                                'Davies_bouldin_score_minmax',
                                                'SDBW_minmax',
                                                'Calinski_harabasz_score_minmax',
                                                'Silhouette_score_minmax',
                                                'CDBW_minmax',
                                                'Davies_bouldin_score_minmax_normalizado',
                                                'SDBW_minmax_normalizado' ]]
        
        mean = df1[['Calinski_harabasz_score_minmax',
                    'Silhouette_score_minmax',
                    'CDBW_minmax',
                    'Davies_bouldin_score_minmax_normalizado',
                    'SDBW_minmax_normalizado']].apply(['mean'],axis=1)
        
        f = pd.concat([df1, mean], axis=1).reindex(df1.index).sort_values(
            by=['mean', 'Algorithm'], ascending=False)   
        final = f[['Algorithm','mean',
                        'Calinski_harabasz_score_minmax',
                        'Silhouette_score_minmax',
                        'CDBW_minmax',
                        'Davies_bouldin_score_minmax_normalizado',
                        'SDBW_minmax_normalizado',
                        'Calinski_harabasz_score',
                        'Silhouette_score', 'Davies_bouldin_score',
                        'SDBW','CDBW',
                        'Davies_bouldin_score_minmax',
                        'SDBW_minmax',
                        ]]
        return final.reset_index(drop=True)
    
    else:
        resultado = pd.DataFrame(resultado, columns=['Algorithm', cluster_validity_metric])

        if cluster_validity_metric == 'davies_bouldin_score' or cluster_validity_metric == 'sdbw':
            return resultado.sort_values(by=[cluster_validity_metric, 'Algorithm'], ascending=True).reset_index(drop=True) 
        else:
            return resultado.sort_values(by=[cluster_validity_metric, 'Algorithm'], ascending=False).reset_index(drop=True)
    
def compute_davies_bouldin_score(data, labels):
    score = 0
    try:
        score = metrics.davies_bouldin_score(data, labels)
    except:
        score = 0
    return score

def compute_silhouette_score(data, labels):
    score = 0
    try:
        score = metrics.silhouette_score(data, labels)
    except:
        score = 0
    return score
        
def compute_calinski_harabasz_score(data, labels):
    score = 0
    try:
        score = metrics.calinski_harabasz_score(data, labels)
    except:
        score = 0
    return score

def compute_sdbw(data, labels):
    score = 0 
    try:
        fd = format_data(data)
        score = S_Dbw(fd, labels)
    except:
        score = 0
    return score

def compute_cdbw(data, labels):
    score = 0 
    try:
        fd = format_data(data)
        score = CDbw(fd, labels)
    except:
        score = 0
    return score

def format_data(data):
    formated_data = []
    for i in range(data.index[0], data.index[0] + len(data)):
        temp = []
        for col in data.columns:
            temp.append(data[col][i])
        formated_data.append(temp)
    return np.asarray(formated_data)

def algorithms(number_clusters: None, ndim:3):
    algos = []
    algos.append(MeanShift())
    
    if number_clusters == None:
        for i in range(2,8): 
            algos.append(KMeans(n_clusters=i, random_state=0)) 
            algos.append(Birch(n_clusters=i))
            algos.append(DBSCAN(eps=i*0.1, min_samples=i))
            algos.append(OPTICS(min_samples=i))
            algos.append(BisectingKMeans(n_clusters=i, random_state=0)) 
            algos.append(SOM(m=i, n=1, dim=ndim, random_state = 0))
            algos.append(GaussianMixture(n_components=i, random_state=0))
            algos.append(AgglomerativeClustering(n_clusters=i))
            algos.append(HDBSCAN(min_cluster_size=i))           
            algos.append(SpectralClustering(n_clusters=i, random_state=0))
        for i in range (2,6):
            algos.append(AffinityPropagation(damping=(i*0.1)+0.3, random_state=0))
            
    else:
        algos.append(KMeans(n_clusters=number_clusters, algorithm = 'lloyd', random_state=0))
        algos.append(KMeans(n_clusters=number_clusters, algorithm = 'lloyd', random_state=0, tol=0.001))
        algos.append(KMeans(n_clusters=number_clusters, algorithm = 'lloyd', random_state=0, tol=0.01))
        algos.append(KMeans(n_clusters=number_clusters, algorithm = 'lloyd', random_state=0, tol=0.1))
        algos.append(KMeans(n_clusters=number_clusters, algorithm = 'lloyd', random_state=0, tol=1))
        
        algos.append(KMeans(n_clusters=number_clusters, algorithm = 'elkan', random_state=0))
        algos.append(KMeans(n_clusters=number_clusters, algorithm = 'elkan', random_state=0, tol=1))
        algos.append(KMeans(n_clusters=number_clusters, algorithm = 'elkan', random_state=0, tol=0.1))
        algos.append(KMeans(n_clusters=number_clusters, algorithm = 'elkan', random_state=0, tol=0.01))
        algos.append(KMeans(n_clusters=number_clusters, algorithm = 'elkan', random_state=0, tol=0.001))
        
#         algos.append(KMeans(n_clusters=number_clusters, algorithm = 'auto', random_state=0))
#         algos.append(KMeans(n_clusters=number_clusters, algorithm = 'full', random_state=0))

#         algos.append(KMeans(n_clusters=number_clusters, n_init = 'auto', algorithm = 'elkan', random_state=0))
#         algos.append(KMeans(n_clusters=number_clusters, n_init = 'auto', algorithm = 'auto', random_state=0))
#         algos.append(KMeans(n_clusters=number_clusters, n_init = 'auto', algorithm = 'full', random_state=0))



        algos.append(Birch(n_clusters=number_clusters))
        algos.append(Birch(n_clusters=number_clusters, threshold=0.3))
        algos.append(Birch(n_clusters=number_clusters, threshold=0.4))
        algos.append(Birch(n_clusters=number_clusters, threshold=0.6))
        algos.append(Birch(n_clusters=number_clusters, threshold=0.7))


        algos.append(BisectingKMeans(n_clusters=number_clusters, random_state=0)) 
        algos.append(BisectingKMeans(n_clusters=number_clusters, init='k-means++', random_state=0)) 
        algos.append(BisectingKMeans(n_clusters=number_clusters, init='k-means++', algorithm = 'elkan', random_state=0)) 
        algos.append(BisectingKMeans(n_clusters=number_clusters, init='k-means++', bisecting_strategy = 'largest_cluster', random_state=0)) 
        algos.append(BisectingKMeans(n_clusters=number_clusters, bisecting_strategy = 'largest_cluster', random_state=0)) 

        algos.append(GaussianMixture(n_components=number_clusters, random_state=0))
        algos.append(GaussianMixture(n_components=number_clusters, covariance_type='tied', random_state=0))
        algos.append(GaussianMixture(n_components=number_clusters, covariance_type='diag', random_state=0))
        algos.append(GaussianMixture(n_components=number_clusters, covariance_type='spherical', random_state=0))
        algos.append(GaussianMixture(n_components=number_clusters, init_params='k-means++', random_state=0))
        algos.append(GaussianMixture(n_components=number_clusters, init_params='random', random_state=0))
        algos.append(GaussianMixture(n_components=number_clusters, init_params='random_from_data', random_state=0))


        algos.append(AgglomerativeClustering(n_clusters=number_clusters))
        algos.append(AgglomerativeClustering(n_clusters=number_clusters, linkage='average'))
        algos.append(AgglomerativeClustering(n_clusters=number_clusters, linkage='complete'))
        algos.append(AgglomerativeClustering(n_clusters=number_clusters, linkage='single'))
        
        algos.append(SOM(m=number_clusters, n=1,dim=ndim, random_state = 0))
    
        algos.append(DBSCAN())
        algos.append(OPTICS())
        algos.append(HDBSCAN(min_cluster_size=number_clusters))
        algos.append(HDBSCAN(min_cluster_size=number_clusters,cluster_selection_method='leaf'))

        #algos.append(SpectralClustering(n_clusters=number_clusters, affinity= 'poly', random_state=0))
        #algos.append(AffinityPropagation(random_state=0))

    return algos
