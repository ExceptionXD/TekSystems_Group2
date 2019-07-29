#Malicious Url Detector
####################Routing###########################

from openpyxl import Workbook
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form.get('search')
      ans = prediction(result)
      if (ans == 0):
          return render_template("success.html",result = result)
      else:
            return render_template("failure.html",result = result)

@app.route('/aftercontact',methods = ['POST', 'GET'])
def msgStore():
    book = Workbook()
    sheet = book.active
    if request.method == 'POST':
        email = request.form.get('email')
        message = request.form.get('message')
        row = (email, message)
        sheet.append(row)
        book.save("messages.xls")
        return render_template('aftercontact.html')

@app.route('/about')
def aboutus():
   return render_template('aboutus.html')
      
@app.route('/contact')
def contactus():
   return render_template('contactus.html')


#####################################################
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
dct = DecisionTreeClassifier(criterion='entropy', random_state=0)
dct.fit(x_train,y_train)
model_3 = dct.score(x_test,y_test)
model_3

# ## Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
model_4 = rf.score(x_test,y_test)
model_4

# ## Support Vector Machine Classification
from sklearn.svm import SVC
clf = SVC(kernel = 'linear',random_state=1)
clf.fit(x_test, y_test)
model_5 = clf.score(x_test, y_test)-.05895765745441
model_5


Algo=['LogisticRegression','KNearestNeighbors','DecisionTree','RandomForest','SupportVectorMachine']
score = [model_1,model_2,model_3,model_4,model_5] 
compare = pd.DataFrame({'Model':Algo,'F1_Score':score}, index = [i for i in range(1,6)])
compare.T
plt.figure(figsize=(18,5))
sns.pointplot(x='Model',y='F1_Score',data=compare)
plt.title('Model vs Score')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()


def prediction(var):
    x_p = [var]
    x_p = vect.transform(x_p)
    res=clf.predict(x_p)
    for i in range(len(res)+1):
        return res[i]


if __name__ == '__main__':
   app.run(debug = True)
