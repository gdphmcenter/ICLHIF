#coding:utf-8
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist,squareform


def ICLHIF():

    with open('Features/ICLHIF_pre_label.csv',mode='w+') as f:
        all_pre_label =[]
        for (a,b,c,d,e) in [(),()]:   #填入CL提取后的特征

            data = pd.read_csv("Features/0147_16dim.csv", sep=',', header=None, skiprows=a,  nrows=b)  
            X_train = data.iloc[:,1:17]
            y_train = data[0]
            test_data = pd.read_csv("Features/0147_16dim.csv", sep=',', header=None, skiprows=c, nrows=d)   
            all_X_test = test_data.iloc[:,1:17]
            all_y_test = test_data[0]

            dists = pdist(X_train,metric='euclidean')
            dist_Matrix = squareform(dists)
            Matrix = np.sort(dist_Matrix,0)
            min_dist_all = Matrix[1]
            dist_temp = pd.Series(min_dist_all)
            threshold = dist_temp.mean() + 2 * dist_temp.std()

            min_dists = []
            for i in range(e):
                X_test = all_X_test.iloc[i]
                X_test = np.array(X_test)          
                X_test = X_test.reshape(1,-1) 
                dist_list = np.sqrt(np.sum(np.asarray(X_test - X_train)**2, axis=1))
                dist_min = min(dist_list)
                min_dists.append(dist_min)

            pre_label = []        
            for j in min_dists:
                if j < threshold:
                    dist_min_bool = 0
                    pre_label.append(dist_min_bool)
                else:
                    dist_min_bool = 1
                    pre_label.append(dist_min_bool)
            all_pre_label.append(pre_label)
        np.savetxt(f, all_pre_label, fmt="%s",delimiter=",")
    
    path = 'Features/ICLHIF_pre_label.csv'
    f = open(path,'r')
    a = f.readlines()
    f = open(path,'w')
    for line_b in a:
        line_a = line_b.lstrip('[')
        if ']' in str(line_a):
            line_a = str(line_a).replace(']',' ')
            f.write(line_a)
    f.close()



def AD():
    all_pre_label = pd.read_csv('Features/ICLHIF_pre_label.csv',header=None)
    for i in range(all_pre_label.shape[0]):
        pre_label = all_pre_label.iloc[i,:]
        df1 = pre_label.value_counts(normalize=True)
        print(df1)


def ND():
    all_pre_label = pd.read_csv('Features/ICLHIF_pre_label.csv',header=None)
    for i in range(all_pre_label.shape[0]):
        pre_label = all_pre_label.iloc[i,:]
        df2 = pre_label.rolling(3).sum()
        df2 = df2.value_counts(normalize=True)
        print(df2)




if __name__ == "__main__":
   
    ICLHIF()
    AD()
    ND()


    
    




