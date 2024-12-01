#!/usr/bin/env python
# coding: utf-8

# #### Problem statement - There are times when a user writes Good, Nice App or any other positive text, in the review and gives 1-star rating. Your goal is to identify the reviews where the semantics of review text does not match rating. 
# 
# #### Your goal is to identify such ratings where review text is good, but rating is negative- so that the support team can point this to users. 
# 
# Deploy it using - Flask/Streamlit etc and share the live link. 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('chrome_reviews.csv')


# In[3]:


data.head()


# In[4]:


data['Star'].unique()


# In[5]:


data.columns


# In[6]:


data.isnull().sum()


# In[7]:


data.shape


# In[8]:


data = data.drop(columns=['ID','Review URL','Developer Reply', 'Version', 'Review Date', 'App ID'])


# In[9]:


data['Text'].sample(10)


# In[10]:


data['Text'] = data['Text'].astype(str)


# In[11]:


data["Text"]


# In[12]:


df = data[['Text','Star']]
df.head()


# In[13]:


## droping nan values
df.dropna()
df.info()


# In[14]:


#import natural language tool kit
import nltk
import re #regular expressions module

nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# In[15]:


#declaring porter stemmer
port = PorterStemmer()
def text_cleaner (text): #function to clean text
    cleaned= re.sub('[^a-zA-Z]', " ", text) 
    cleaned= cleaned.lower()
    cleaned = cleaned.split()
    cleaned= [port.stem (word) for word in cleaned if word not in stopwords.words("english")]
    cleaned= ' '.join(cleaned)
    return cleaned


# In[16]:


df["Cleaned_Text"] = df["Text"].apply(lambda x: text_cleaner(str(x))) #declare cleaned text feature
df["Length"] = df["Text"].apply(lambda x:len(str(x))) #declare length feature
df.head()


# In[17]:


from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def sentiment_vader(text, sid):
    ss = sid.polarity_scores(text)
    ss.pop('compound')
    return max(ss, key=ss.get)


# In[18]:


def sentiment_textblob(text):
        x = TextBlob(text).sentiment.polarity
        
        if x<0:
            return 'negative'
        elif x==0:
            return 'neutral'
        else:
            return 'positive'

def plot_sentiment_barchart(text, method='TextBlob'):
    if method == 'TextBlob':
        sentiment = text.map(lambda x: sentiment_textblob(x))
    elif method == 'Vader':
        nltk.download('vader_lexicon')
        sid = SentimentIntensityAnalyzer()
        sentiment = text.map(lambda x: sentiment_vader(x, sid=sid))
    else:
        raise ValueError('Textblob or Vader')
    
    plt.bar(sentiment.value_counts().index,
            sentiment.value_counts())


# In[19]:


nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


# In[20]:


df["Score"] = df["Cleaned_Text"].apply(lambda review:sid.polarity_scores(review))


# In[21]:


df["Compound_Score"]  = df['Score'].apply(lambda score_dict: score_dict['compound'])


# In[22]:


df["Result"] = df["Compound_Score"].apply(lambda c: 'positive' if c > 0 else ('negative' if c < 0 else 'neutral'))
df.head()


# In[23]:


Suggestion = []
for row in df["Star"] :
    if row >= 3 :
         Suggestion.append("Correct rating")
    else :
         Suggestion.append("Check rating given")
            
df["Suggestion"] = Suggestion
df.head()


# ### model building

# In[24]:


from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
corpus = []
for i in range(len(data)):
    review=re.sub('[^a-z-A-Z]', ' ',df['Text'][i])
    review=review.lower()
    review=review.split()
    
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review= ' '.join(review)
    corpus.append(review)


# In[25]:


from sklearn.preprocessing import LabelEncoder
lab_enc = LabelEncoder()
df['Suggestion'] = lab_enc.fit_transform(df['Suggestion'])


# In[28]:


X = df.drop(columns=['Suggestion'])
Y = df['Suggestion']


# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v=TfidfVectorizer(max_features=2,ngram_range=(1,2))
X= tfidf_v.fit_transform(corpus).toarray()


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)


# In[33]:


tfidf_v.get_params()


# In[34]:


import warnings
warnings.simplefilter("ignore")


# In[36]:


count_data=pd.DataFrame(X_train, columns=tfidf_v.get_feature_names())


# In[37]:


import matplotlib.pyplot as plt


# In[38]:


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()


# In[39]:


import numpy as np
import itertools
from sklearn.metrics import accuracy_score,confusion_matrix


# In[41]:


def plot_confusion_matrix(cm, classes, normalize=False,
                         title='Confusion matrix', 
                         cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[42]:


classifier.fit(X_train,Y_train)
pred=classifier.predict(X_test)
score=accuracy_score(Y_test, pred)*100
print("accuracy:   %0.3f"% score)
cm=confusion_matrix(Y_test, pred)
plot_confusion_matrix(cm, classes=["Correct stars","Wrong star"])


# In[43]:


import pickle
filename= 'model.pkl'
pickle.dump(classifier, open(filename, 'wb'))


# In[23]:


df_focus = df[(df.Result == "positive")]
df_focus.head()


# In[25]:


df_focus.Suggestion.value_counts()


# In[26]:


df_focus.sample(10)


# In[ ]:


keyword = ['good','nice','thank you','best','awesome','helpful']


# In[ ]:


final = df_focus[(df_focus["Suggestion"] == "Focus Needed")]
final = final[final["Cleaned_Text"].isin(keyword)]
final.drop(final.iloc[:, 3:7], inplace = True, axis = 1)
display(final.head())
print(f"There are about {len(final.Suggestion)} reviews that are positive but have a bad rating")


# ### part 1 ques 2 completed
