#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import pandas as pd
import pandas_profiling
from yellowbrick.target import ClassBalance
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz # display the tree within a Jupyter notebook
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from ipywidgets import interactive, IntSlider, FloatSlider, interact
import ipywidgets
from IPython.display import Image
from subprocess import call
import matplotlib.image as mpimg
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.classifier import ROCAUC
plt.style.use("ggplot")
warnings.simplefilter("ignore")
plt.rcParams['figure.figsize'] = (12,8)
hr=pd.read_csv("F:/COLLEGE/ML Project/Employee Turnover/Turnover.csv")
hr.profile_report(title="Data Report")

pd.crosstab(hr.department, hr.quit).plot(kind='bar')
plt.title("Turnover Frequency on Salary Bracket")
plt.xlabel('Department')
plt.ylabel('Frequency of Turnover')
cat_vars = ['department','salary']
for var in cat_vars:
     cat_list=pd.get_dummies(hr[var], prefix=var)
     hr=hr.join(cat_list)
print(hr.head())
hr.drop(columns=['department','salary'], axis=1, inplace=True)

#balance class
visualizer=ClassBalance(labels=['stayed','left']).fit(hr.quit)
visualizer.show()
X=hr.loc[:, hr.columns!='quit']
y=hr.quit
X_train, X_test, y_train,y_test= train_test_split(X,y,random_state=0,test_size=0.2,stratify=y)
@interact
def plot_tree(crit=['gini','entropy'],
              split=['best','random'],
              depth=IntSlider(min=1,max=30,value=2, continuous_update=False),
              min_split=IntSlider(min=2,max=5,value=2, continuous_update=False),
              min_leaf=IntSlider(min=1,max=5,value=1, continuous_update=False)):
     estimator=DecisionTreeClassifier(random_state=0,
                                      criterion=crit,
                                      splitter=split,
                                      max_depth=depth,
                                      min_samples_split=min_split,
                                      min_samples_leaf=min_leaf)

     estimator.fit(X_train, y_train)
     print('Decision Tree Training Accuracy: {:.3f}'.format(accuracy_score(y_train,estimator.predict(X_train))))
     print('Decision Tree Training Accuracy: {:.3f}'.format(accuracy_score(y_test,estimator.predict(X_test))))
     graph=Source(tree.export_graphviz(estimator,
                                       out_file=None,
                                       feature_names=X_train.columns,
                                       class_names=['stayed','quit'],
                                       filled=True))

     display(Image(data=graph.pipe(format='png')))
     graph.view()

dt=DecisionTreeClassifier(class_weight=None,criterion='gini',max_depth=3,
                          max_features=None,max_leaf_nodes=None,
                          min_impurity_decrease=0.0,min_impurity_split=None,
                          min_samples_leaf=1,min_samples_split=2,
                          min_weight_fraction_leaf=0.0,presort=False,random_state=0,
                          splitter='best')

viz=FeatureImportances(dt)
viz.fit(X_train,y_train)

viz.show();
visualizer=ROCAUC(dt, classes=['stayed','quit'])
visualizer.fit(X_train,y_train)
visualizer.score(X_test,y_test)
visualizer.poof();





















plt.show()
