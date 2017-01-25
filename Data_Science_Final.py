
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import plotly.plotly as py
from plotly.graph_objs import *
import cufflinks as cf
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Ridge , RidgeClassifier
from sklearn.svm import SVC

from sklearn import neighbors 
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[47]:

auto = pd.read_csv('Auto.csv')


# In[48]:

auto.year.value_counts()


# In[49]:

auto.head(2)


# # A little Preprocessing: Inpute the 5 missing HP values

# #### First, seperate the missing rows from rest of the data

# In[50]:


n = len(auto.year.value_counts())

bad_hp_year = []
hp_years = [[] for _ in range(n)]
mpg_years = [[] for _ in range(n)]
i = 0

for index, row in auto.iterrows():
    year = row['year'] - 70
    
    try:
        hp_years[year].append(int(row['horsepower']))
    except ValueError:
        bad_hp_year.append(i)
        
    mpg_years[year].append(int(row['mpg']))
    i += 1


# #### Below, originial later in the code, was moved to the begining - changes values from str -> floats

# In[51]:

auto.cylinders = auto.cylinders.astype(float)
auto.displacement = auto.displacement.astype(float)
## auto.horsepower = auto.horsepower.astype(float)  MUST drop missing values before converting to float
auto.weight = auto.weight.astype(float)
auto.acceleration = auto.acceleration.astype(float)
auto.year = auto.year.astype(float)
auto.origin = auto.origin.astype(float)
auto.mpg = auto.mpg.astype(float)

missing = auto.iloc[bad_hp_year]
missing = missing.drop('horsepower',1)
car_names = missing['Make-Model']
missing = missing.drop('Make-Model',1)

auto = auto.drop(auto.index[bad_hp_year])
auto.horsepower = auto.horsepower.astype(float)


# In[52]:

missing  ## Test set with missing HP values removed


# In[ ]:




# In[53]:

## Imputing the ????? missing values in HorsePower

X_train = pd.DataFrame(auto[['cylinders','displacement','mpg','weight','acceleration','year','origin']])
y_train = pd.DataFrame(auto[['horsepower']])

X_test = pd.DataFrame(missing)

knn = neighbors.KNeighborsRegressor()

y_predict = knn.fit(X_train, y_train)
y_predict = y_predict.predict(X_test)


# In[54]:

missing['horsepower'] = y_predict
missing['Make-Model'] = car_names


# In[57]:

plt.scatter(X_train['mpg'], y_train, c='k', label='data')
plt.plot( X_test['mpg'],y_predict, c='g', label='prediction')


# In[56]:

missing


# ### As you can see, Ford Mavering has 94 Horsepower!!  Not saying that Maverick could have had an incredible year in 1974, but it is unlikely.  The Mav was made for 6 years total, with an HP range 72-88, we will manually inpute 87 hp for 1974, taking to account its large knn prediction.
# 

# In[12]:

missing.loc[missing.index[0],'horsepower'] = 87
print missing


# In[13]:

auto = auto.append(missing)


# In[14]:

part1 = auto.copy()

American_hp = []
American_year = []
German_hp = []
German_year = []
Japanese_hp = []
Japanese_year =[]

for i in range(0,len(part1)):
    if part1.origin[i] == 1:
        American_hp.append(part1.horsepower[i])
        American_year.append(part1.year[i])
    elif part1.origin[i] == 2:
        German_hp.append(part1.horsepower[i])
        German_year.append(part1.year[i])
    elif part1.origin[i] == 3:
        Japanese_hp.append(part1.horsepower[i])
        Japanese_year.append(part1.year[i])
    i+=1


# In[42]:

print np.mean(American_hp)
print np.mean(German_hp)
print np.mean(Japanese_hp)


# # Number 1: Data Visualization.  HP comparison for each country.  

# In[44]:

year = [range(1970,1983)]

trace1 = Scatter(
    x=American_year, y=American_hp,
    mode = 'markers',
    marker = dict(
        size = 16,
        color = 'Blue',
        showscale=True
        ),
    name = 'American HP',
    text=['American HP averaged 118.6'],
    textposition='top'
)

trace2 = Scatter(
    x=German_year, y=German_hp,
    mode = 'markers',
    marker = dict(
        size = 16,
        color = 'Black',
        showscale=True
        ),
    name='German HP',
    text=['German HP averaged 80.2'],
    textposition='top'
)

trace3 = Scatter(
    x=Japanese_year, y=Japanese_hp,
    mode = 'markers',
    marker = dict(
        size = 16,
        color = 'Red',
        showscale=True
        ),
    name='Japanese HP',
    text=['Japanese HP averaged 10.8'],
    textposition='top'
    
)



data = Data([trace1, trace2, trace3])
layout = Layout(
    title='Cars from around the world',
    updatemenus=list([
        dict(
            x=1970,
            y=1,
            yanchor='top',
            buttons=list([
                dict(
                    args=['visible', [True, True, True]],
                    label='All',
                    method='restyle'
                ),
                dict(
                    args=['visible', [True, False, False]],
                    label='American',
                    method='restyle'
                ),
                dict(
                    args=['visible', [False, True, False]],
                    label='German',
                    method='restyle'
                ),
                dict(
                    args=['visible', [False, False, True]],
                    label='Japanese',
                    method='restyle'
                ),
            ]),
        )
    ]),
)
fig = Figure(data=data, layout=layout)
py.iplot(fig)


# ## (Assuming the dataset is an accurate sampling of vehicles) The 60s and 70s was the era, of the American Sports car.  Detroit was throwing out low riding, bulky, fast cars as fast as the people could buy them, going to show no suprise in the higher average of HP.  
# 
# ## However, where Americans had in HP, they gave up in weight and MPG, where Japan and Germany wins.  
# 
# ## An interesting observation is the dip in HP (and rise in MPG) in 1974.  Brought on by the fuel crisis of 1973, where Nixon also lowered the national speed limit to 55 mph.  (Don't need much HP when you can't go fast..)

# In[16]:

avg_hp_years = []
avg_mpg_years = []
for i in range(0,len(hp_years)):
    avg = sum(hp_years[i])  /  float(len(hp_years[i]))
    avg_hp_years.append(avg)
    
    avg = sum(mpg_years[i])  /  float(len(mpg_years[i]))
    avg_mpg_years.append(avg)


# In[17]:

year_range = []
for i in range(1970,1983):
    year_range.append(i)
year_range


# In[18]:

data = {'MPG':avg_mpg_years, 'HP':avg_hp_years}
df = pd.DataFrame(data = data, index = year_range)


# ### Not a part of #1, quick plot to see if MPG and HP are correlated.

# In[21]:

df.iplot(secondary_y=['MPG', 'HP'], kind='bar', y = '' , title = 'HP vs MPG all Makes and MOdels',
         xTitle = 'year',yTitle = 'HP & MPG')


# ## 2.  Cross validation.  70:30 Training test, --->>  280 training, 117 test

# In[22]:

## Function calculates MSE of data frame

def MSE (results):
    actual = results[results.columns[0]]  #  <-- actual column
    results = results[results.columns[1]] #  <-- prediction column
    
    length = len(results)
    se = []
    for i in range(0,length):
        error = (float(actual[i]) - float(results[i]))**2
        se.append(error)
    
    se = sum(se)
    se = se**(.5)
    return se    


# In[23]:

r_mse = []
rf_mse = []
knn_mse = []
                                ## Not very pythonic, however perform shuffle-split cross validation 10 times for KNN, RR & RF
for i in range(0,10):
    auto = auto.reindex(np.random.permutation(auto.index))


    X_train = pd.DataFrame(auto[['cylinders','displacement','horsepower','weight','acceleration','year','origin']][:280])
    y_train = pd.DataFrame(auto[['mpg']][:280])

    X_test = pd.DataFrame(auto[['cylinders','displacement','horsepower','weight','acceleration','year','origin']][280:])


    clf = RandomForestRegressor()
    clf.fit(X_train,y_train)
    rf_predict = clf.predict(X_test)

    knn = neighbors.KNeighborsRegressor()
    knn.fit(X_train, y_train)
    knn_predict = knn.predict(X_test)
    knn_predict = [float(i) for i in knn_predict]

    ridge = Ridge()
    ridge.fit(X_train, y_train)
    ridge_predict = ridge.predict(X_test)
    ridge_predict = [float(i) for i in ridge_predict]

    actual = auto.mpg[280:]
    actual = (actual.values.tolist())

    rf_results = {'actual': actual, 'prediction' : rf_predict}
    rf_results = pd.DataFrame(rf_results)

    knn_results = {'actual': actual, 'prediction' : knn_predict}
    knn_results = pd.DataFrame(knn_results)

    ridge_results = {'actual': actual, 'prediction' : ridge_predict}
    ridge_results = pd.DataFrame(ridge_results)
    
    rf_mse.append(MSE(rf_results))
    r_mse.append(MSE(ridge_results))
    knn_mse.append(MSE(knn_results))


# In[24]:

print 'The trial Mean Squared Error For our 70:30 Random Forrest Regression test set is ', np.mean(rf_mse),'!!!'
print 'The trial Mean Squared Error For our 70:30 KNN Regression test set is ',  np.mean(knn_mse),'!!!'
print 'The trial Mean Squared Error For our 70:30 Ridge Regression test set is ', np.mean(r_mse),'!!!'


# ### Random Forrest produces the lowest MSE, therefore our MPG predictor should use randome forrest!

# In[25]:

instances = clf.feature_importances_


# In[26]:

titles = auto.columns
instances
titles = titles.drop('mpg',1)
titles = titles.drop('Make-Model',1)

d1 = {}
for i in range(0,len(instances)):
    a = titles[i]
    d1[a] = instances[i]
df1 = pd.DataFrame(data = d1, index =[0])


#df = pd.DataFrame( 'acceleratioin' = instances[0],'cylinders','displacement','horsepower','origin','weight','year')


# #### Plot showing 'importance' of each attribute for predicting MPG

# In[27]:

df1.iplot(kind = 'bar', title = 'Attribute Importance with respect to predicting MPG',
          xTitle = 'Attributes',yTitle ='Importance')


# In[28]:

auto['target'] = 0


# In[29]:

for i in range(0,len(auto)):
    if auto.acceleration[i] > 15:
        auto.loc[i, 'target'] = 1
    elif auto.acceleration[i] <= 15:
        auto.loc[i, 'target'] = 0
    else:
        print "didnt work"


# # 3.  Target Prediction.

# #### Function that takes count of wrongly predicted values

# In[30]:

def class_accuracy (results):
    actual = results[results.columns[0]]  #  <-- actual column
    results = results[results.columns[1]] #  <-- prediction column
    
    length = len(results)
    count = 0
    for i in range(0,length):
        if float(actual[i]) == float(results[i]):
            count += 1
    return count    


# In[31]:

r_class_error = []
rf_class_error = []
knn_class_error = []
svm_class_error = []
                                ## Method Below not very pythonic, however perform shuffle-split cross validation 10 times for KNN, RR & RF
for i in range(0,10):
    auto = auto.reindex(np.random.permutation(auto.index))


    X_train = pd.DataFrame(auto[['cylinders','displacement','horsepower','weight','acceleration','year','origin','mpg']][:280])
    y_train = pd.DataFrame(auto[['target']][:280])

    X_test = pd.DataFrame(auto[['cylinders','displacement','horsepower','weight','acceleration','year','origin','mpg']][280:])


    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)
    rf_predict = clf.predict(X_test)

    knn = neighbors.KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, y_train)
    knn_predict = knn.predict(X_test)
    knn_predict = [float(i) for i in knn_predict]

    ridge = RidgeClassifier()
    ridge.fit(X_train, y_train)
    ridge_predict = ridge.predict(X_test)
    ridge_predict = [float(i) for i in ridge_predict]
    
    svm = SVC()
    svm.fit(X_train,y_train)
    svm_predict = svm.predict(X_test)
    svm_predict = [float(i) for i in svm_predict]

    actual = auto.target[280:]
    actual = (actual.values.tolist())

    rf_results = {'actual': actual, 'prediction' : rf_predict}
    rf_results = pd.DataFrame(rf_results)

    knn_results = {'actual': actual, 'prediction' : knn_predict}
    knn_results = pd.DataFrame(knn_results)

    ridge_results = {'actual': actual, 'prediction' : ridge_predict}
    ridge_results = pd.DataFrame(ridge_results)
    
    svm_results = {'actual': actual, 'prediction' : svm_predict}
    svm_results = pd.DataFrame(svm_results)
    
    rf_class_error.append(class_accuracy(rf_results))
    r_class_error.append(class_accuracy(ridge_results))
    knn_class_error.append(class_accuracy(knn_results))
    svm_class_error.append(class_accuracy(svm_results))


# In[32]:

print 'The trial accuracy for classifying with Random Forrest Regression test set is ',(1- np.mean(rf_class_error)/len(auto))*100,'!!!'
print 'The trial accuracy for classifying with KNN Regression test set is ',(1- np.mean(knn_class_error)/len(auto))*100,'!!!'
print 'The trial accuracy for classifying with Ridge Regression test set is ',(1- np.mean(r_class_error)/len(auto))*100,'!!!'
print 'The trial accuracy for classifying with SVC test set is ',(1- np.mean(svm_class_error)/len(auto))*100,'!!!'


# ## The Best Classifying Method was found to be SVC with RBF kernel, degree =3

# In[33]:

from sklearn.metrics import classification_report
y_true = svm_results.actual
y_pred = svm_results.prediction
#target_names = ['target']
print(classification_report(y_true, y_pred))



# # HOWEVER, it cheated and didn't predict any 0 targets... smh

# In[34]:

y_true = knn_results.actual
y_pred = knn_results.prediction
#target_names = ['target']
print(classification_report(y_true, y_pred))


# ### ^ KNN did a much better job of overall predicting.  (78% accuracy)

# In[37]:

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 
from sklearn.metrics import auc
from sklearn import metrics



# In[38]:

fpr, tpr, thresholds = roc_curve(knn_results.actual,knn_results.prediction)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % metrics.roc_auc_score(knn_results.actual,knn_results.prediction))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for KNN')
plt.legend(loc="lower right")
plt.show()


# In[39]:

fpr, tpr, thresholds = roc_curve(svm_results.actual,svm_results.prediction)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % metrics.roc_auc_score(svm_results.actual,svm_results.prediction))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for SVM')
plt.legend(loc="lower right")
plt.show()


# ## As previously noted, Suppport Vector Classification cheated into making the accuracy look high.  KNN classifier with 15 neighbors produced the best target prediction for this project.  
