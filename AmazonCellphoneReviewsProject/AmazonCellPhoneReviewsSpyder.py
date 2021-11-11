#!/usr/bin/env python
# coding: utf-8


# In[1]:


import re
import pandas as pd
import  numpy as np


# In[2]:


dataset =pd.read_csv("Reviews.csv")


# In[3]:


dataset.head()


# In[4]:


dataset.tail()


# In[5]:


dataset.isnull()


# In[6]:


dataset.isnull().sum()


# In[7]:


#dataset.dropna(inplace=True)
dataset['body']=dataset['body'].fillna('').apply(str)
dataset['name']=dataset['name'].fillna('').apply(str)
dataset['title']=dataset['title'].fillna('').apply(str)
dataset['helpfulVotes']=dataset['helpfulVotes'].fillna('').apply(str)

# In[8]:


dataset.isnull().sum()


# In[9]:


dataset=dataset.drop(columns=['asin','name','helpfulVotes','date'],axis=1)


# Check cleaned dataset

# In[10]:


a=dataset['rating'].tolist()
print(a)


# Based on the rating appending value 0 or 1 to the empty list

# In[11]:


d=[]
for i in range(len(a)):
    if a[i]>=3:
        d.append(1)
    else:
        d.append(0)
d


# Now make the above list as an emotion column

# In[12]:


dt=pd.DataFrame(d,columns=['emotions'])
dt


# Now concatenating this emotion column with our dataset.

# In[13]:


data1=pd.concat([dataset,dt],axis=1)
data1.head()


# In[14]:


data1.drop(['verified'],axis=1,inplace=True)


# In[15]:


data1['Review'] = data1['title'].str.cat(data1['body'],sep=" ")
data1


# In[16]:


data1.drop(['title','body','rating'],axis=1,inplace=True)


# In[17]:


data1.shape


# Split the data into x (independent variable

# In[18]:


x=data1.iloc[:,1].values #Review


# In[19]:


x.shape


# In[20]:


y=data1.iloc[:,0].values
y #emotion


# In[21]:


y.shape


# In[22]:


import nltk
nltk.download('stopwords')


# In[23]:


nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet=WordNetLemmatizer()


# In[24]:

temp=re.sub('[^a-zA-Z]',' ',str(x[0]))#replacing punctuations and numbers using re library




# In[25]:

temp=temp.lower()#convert the text to lower cases


# The process of extracting out the root word is called stemming.


# In[26]:
#To lemmatize each word first we have to split the review into a list and then apply lemmatization functionality. We use WordNetLemmatizer for the Lemmatization of the word.
temp=temp.split()
#creating WordNetLemmatizer object to take main lemma of each word
wordnet=WordNetLemmatizer()
#loop for lemmatization each word in string array at ith row
temp=[wordnet.lemmatize(word) for word in temp if not word in set(stopwords.words('english'))]


#rejoin all string array elemnts to create back into a string 
temp=' '.join(temp)


# In[27]:


#initialize empty array to append clean text
corpus=[]
#no of rows to clean
for i in range (len(x)):
    #temp=data1['Review'][i]
    #replacing punctions and numbers using re library
    temp=re.sub('[^a-zA-Z]',' ',str(x[i]))
    #convert all text to lower cases
    temp=temp.lower()
    #split to array (default delimiter is " ")
    temp=temp.split()
    
    #creating WordNetLematizer object to take main lemma of each word 
    wordnet=WordNetLemmatizer()
    #Loop for Lemmatizartion each word in string array at ith row
    temp=[wordnet.lemmatize(word)for word in temp if not word in set(stopwords.words('english'))]
    #rejoin all string array elements to create back into a string 
    temp =' '.join(temp)
    #append each string to create array of clean text
    
    corpus.append(temp)




#creating bag of word model

from sklearn. feature_extraction.text import CountVectorizer

#To extract max 2000 feature, "max features" is attribute to #experiment with to get better results

cv=CountVectorizer (max_features= 2000)

#z contains vectorized data (independent variable) 
x=cv.fit_transform(corpus).toarray()



# In[29]:


x.shape


# In[30]:


import pickle
pickle.dump(cv,open('count_vec.pkl','wb'))


# In[31]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)



# In[32]:


x_test.shape


# In[33]:


x_train.shape


# In[34]:


y_test.shape


# In[35]:


y_train.shape


# In[36]:



from tensorflow.keras.models import Sequential #library to initailise the model
from tensorflow.keras.layers import Dense #library used to add layers 


# In[37]:


model=Sequential()
model.add(Dense(units=2000,activation='relu'))
model.add(Dense(units=1000,activation='relu')) 
model.add(Dense(units=1000,activation='relu')) 
model.add(Dense(units=1000,activation='relu')) 
model.add(Dense(units=1,activation='sigmoid')) 
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) 
model.fit(x_train,y_train,batch_size=128,epochs=50)





model.save('AmazonReview_model_saved1.h5')
# In[39]:

#testing the prediction
y_pred=model.predict(x_test)
text="One star product"
text=re.sub('[^a-zA-Z]',' ',text)



# In[25]:

text=text.lower()
# In[26]:

text=text.split()

wordnet=WordNetLemmatizer()
text=[wordnet.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
text=' '.join(text)
# In[27]:
y_p=model.predict(cv.transform([text]).toarray())





