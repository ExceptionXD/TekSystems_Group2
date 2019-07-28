#Malicious Url Detector

import pandas as pd
import seaborn as sns
import numpy as np
from PIL import Image
import urllib
import requests
import matplotlib.pyplot as plt


url_data_bad = pd.read_csv("data2.csv", header = None, names = ["url", "class"])
url_data_bad.head()

url_data_bad['class'] = url_data_bad['class'].map({'bad':1})
url_data_bad.head(20)


url_data_bad['class'].unique()

url_data_bad_head = url_data_bad.head(3000)

url_data_bad_head.tail(10)

url_data_good = pd.read_csv("URLS.txt", header=None, names = ["url", "class"])

data = pd.concat([url_data_bad_head,url_data_good])

data['class'].unique()

data.shape

data[data['class']==0]

data[data['class']==1].head()

data_arr = np.array(data)



def get_tokens(input):
    tokens_by_slash=str(input.encode('utf-8')).split('/')
    
    all_tokens=[]
  
    for i in tokens_by_slash:
        tokens=str(i).split('-')
        tokens_by_dot=[]
    
        for j in range(0,len(tokens)):
            temp_tokens=str(tokens).split('.')
            tokens_by_dot=tokens_by_dot+temp_tokens
        all_tokens=all_tokens + tokens + tokens_by_dot
        
        # removes redundancy
        all_tokens = list(set(all_tokens))
        
        # .com is not required to be added to our features
        if 'com' in all_tokens:
            all_tokens.remove('com')
  
    return all_tokens

test = [te[1] for te in data_arr]
test

train = [tr[0] for tr in data_arr]
train

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(tokenizer=get_tokens)

train_vect = vect.fit_transform(train)
train_vect.todense()


# ## Splitting the data into train and test 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train_vect,test,test_size=0.3)


# ## Logistic Regression

from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression(random_state=0)
lreg.fit(x_train,y_train)
model_1 = lreg.score(x_test,y_test)
model_1

# ## K Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_test, y_test)
model_2 = classifier.score(x_test, y_test)
model_2

# ## Decision Trees
from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier(criterion='entropy', random_state=1)
dct.fit(x_train,y_train)
model_3 = dct.score(x_test,y_test)
model_3

# ## Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
model_4 = rf.score(x_test,y_test)
model_4

