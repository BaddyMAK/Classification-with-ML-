# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:21:30 2017

@author: Badiaa
"""

import pandas as pd
import numpy as np
from sklearn import tree

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
#%%

lis= ["age", "bp", "sg", "al", "su", "rbc", "pc",  "pcc", "ba", "bgr", "bu", 
      "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "htn", "dm", "cad",
      "appet", "pe", "ane", "class"]
xx = pd.read_csv('chronic_kidney_disease.arff', sep=',', skiprows=29, header=None, na_values=['?', '\t?'],
                  names = lis)
key_list =["normal", "abnormal", "present", "notpresent", "yes", "no", "poor", "good", "ckd", "notckd","ckd\t","\tyes", "\tnotckd",  "\tno" , ]
key_val=[0,1,0,1,0,1,0,1,1,0,1,1,0,0]
x= xx.copy()
x=x.replace(key_list,key_val)
data_frame=x.dropna()
data_frame.info()

print(data_frame.describe().T)
target=data_frame.loc[:, 'class']  # the class to be estimated 
target_names=['notckd', 'ckd'] # the categorical features of the class to be estimated
data=data_frame.drop( 'class', axis =1) # the original dataframe without the last column (the class to be estimated)
clf=tree.DecisionTreeClassifier("entropy") # implementation of the hierarchical/decision trees
clf=clf.fit(data, target)
feat_names = lis
feat_names.remove('class')

dotfile = open("Tree.dot",'w')
dot_data=tree.export_graphviz(clf, out_file=dotfile,feature_names= lis,
    class_names=target_names, filled= True, rounded= True, special_characters=True)
dotfile.close()
#%% features importances 
print('features importance')
for name, importance in zip(feat_names, clf.feature_importances_):
    print(name, importance)

plt.figure()    
plt.plot(clf.feature_importances_,'b*')
plt.xlabel('Feature number')
plt.ylabel('Information Gain')
plt.title('Feature Importances (case1: Remove)')
plt.show()  

np.set_printoptions(precision=2)
a=clf.feature_importances_
print (a)
cc=data.iloc[[1]]

print(clf.predict_log_proba(cc))


#%%