#!/usr/bin/env python
# coding: utf-8

# ### Importing necessary libraries 

# In[1]:


import time
import warnings
import itertools
import numpy as np
import pandas as pd
from IPython.display import display 
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


start = time.time()


# In[3]:


warnings.filterwarnings("ignore")


# ### Importing the DataFrame 

# In[4]:


df = pd.read_csv("./clean_and_structured_news.csv")
df.head()


# ### Removing the duplicates 

# In[5]:


class RemoveDuplicates:
    def __init__(self, df: pd.DataFrame) -> None:
        self.__df = df
        self.__vectorizer = TfidfVectorizer(analyzer='word',
                                            ngram_range=(2, 5),
                                            stop_words='english',
                                            max_features=5000)

    def __combine_text(self) -> list[str]:
        """
        this function takes 'structured_data' column and converts it into list of string
        :return: list of string of sentences
        """
        self.__corpus = '.'.join(self.__df['structured_data'].to_list())
        return self.__corpus.split(".")

    def train(self) -> TfidfVectorizer:
        """
        this function, fits and transform tfidf vectorizer using the data under 'structured_data' column
        :return: vectors generated on the basis of data present under 'structured_data' column
        """
        self.__corpus = self.__combine_text()  # corpus: collection of texts
        self.__vectorizer.fit_transform(self.__corpus)
        # self.__only_idf = self.__vectorizer.idf_
        return self.__vectorizer

    def __get_sentence_embeddings(self, vectorizer, row_no: int, no_of_sentences: int) -> list:
        """
        this function returns indices of sentence that are repeated many times throughout the DataFrame
        :param row_no: indicates the DataFrame's index on which processing is to be done
        :param no_of_sentences: indicates the number of sentences present in DataFrame's index specified above
        :return: indices of sentences within the DataFrame's index that are repeated many times
        """
        self.__vectorizer = vectorizer
        structured_data = self.__df['structured_data'][row_no].split(".")
        cleaned_data = self.__df['clean_data'][row_no].split(".")
        
        X_test = [[np.exp(np.sqrt(self.__vectorizer.transform(structured_data)[i].toarray())).mean(), i, row_no, cleaned_data[i]] for i in range(no_of_sentences)]
        X_test.sort()
        return X_test

    def embeddings_for_sentence(self, vectorizer) -> list:
        """
        this function returns the sentence that are removed from the data.
        :return: sentences removed from the data.
        """
        __row = []
        for i in range(self.__df.shape[0]):
            structured_data = self.__df['structured_data'][i].split(".")
            cleaned_data = self.__df['clean_data'][i].split(".")
            no_of_sentences = len(structured_data)
            sentence_embeddings = self.__get_sentence_embeddings(vectorizer, i, no_of_sentences)
            __row.append(sentence_embeddings)
        return __row
    
    def __compute_threshold(self) -> float:
        """
        this function computes appropriate threshold
        :return: returns the calculated threshold
        """
        if "sentence_embeddings" in self.__df.columns:
            lst = list(itertools.chain(*self.__df['sentence_embeddings'].tolist()))
            sentence_embeddings = [i for (i, j, k, l) in lst]
            sentence_embeddings.sort()
            thresh = np.percentile(sentence_embeddings, 15, interpolation = 'midpoint')
            Q1 = np.percentile(sentence_embeddings, 25, interpolation = 'midpoint') # 0.00019836157232631794
            Q2 = np.percentile(sentence_embeddings, 50, interpolation = 'midpoint') # 0.0002630437164689158
            Q3 = np.percentile(sentence_embeddings, 75, interpolation = 'midpoint') # 0.00012697415854370304
            IQR = Q3 - Q1 # 0.00012697415854370304
            #sns.boxplot(sentence_embeddings)
            #print(f"Q1 25 percentile of sentence_embeddings: {Q1}")
            #print(f"Q1 50 percentile of sentence_embeddings: {Q2}")
            #print(f"Q1 75 percentile of sentence_embeddings: {Q3}")
            #print(f"IQR of sentence_embeddings: {IQR}")
            threshold = Q1 - 1.5 * IQR # 7.900334510763384e-06
            return Q1
    
    def duplicate_sentence_indices(self) -> list:
        """
        this function gives the indices of the duplicate sentences
        :return: returns the list duplicate sentence indices
        """
        __all_rows = []
        threshold = self.__compute_threshold()
        for i in self.__df['sentence_embeddings']:
            __row = []
            for [j, k, l, m] in i:
                if 0 < j < threshold:
                    __row.append([j, k, l, m])
            __all_rows.append(__row)
        return __all_rows
    
    def duplicate_sentences(self) -> list:
        """
        this function returns list of duplicate sentences
        :return: returns duplicate sentence's list
        """
        __all_rows = []
        for i in range(self.__df.shape[0]):
            cleaned_data = self.__df['clean_data'][i].split(".")
            duplicate_sentence_indices = self.__df['duplicate_sentence_indices'][i]
            __rows = [cleaned_data[k] for [j, k, l, m] in duplicate_sentence_indices]
            __all_rows.append('.'.join(__rows))
        return __all_rows


# In[6]:


rd = RemoveDuplicates(df)


# In[7]:


get_ipython().run_cell_magic('time', '', 'vectorizer = rd.train()')


# In[8]:


get_ipython().run_cell_magic('time', '', "df['sentence_embeddings'] = rd.embeddings_for_sentence(vectorizer)")


# In[9]:


get_ipython().run_cell_magic('time', '', "df['duplicate_sentence_indices'] = rd.duplicate_sentence_indices()")


# In[10]:


get_ipython().run_cell_magic('time', '', "df['duplicate_sentences'] = rd.duplicate_sentences()")


# In[11]:


df.head()


# In[12]:


def compute_threshold(series: pd.Series) -> float:
    lst = list(itertools.chain(*series.tolist()))
    sentence_embeddings = [i for (i,j,k,l) in lst]
    sentence_embeddings.sort()
    """Q1 = np.percentile(sentence_embeddings, 25, interpolation = 'midpoint')
    Q2 = np.percentile(sentence_embeddings, 50, interpolation = 'midpoint')
    Q3 = np.percentile(sentence_embeddings, 75, interpolation = 'midpoint')
    IQR = Q3 - Q1 
    sns.boxplot(sentence_embeddings)
    print(f"Q1 25 percentile of sentence_embeddings: {Q1}")
    print(f"Q1 50 percentile of sentence_embeddings: {Q2}")
    print(f"Q1 75 percentile of sentence_embeddings: {Q3}")
    print(f"IQR of sentence_embeddings: {IQR}")
    threshold = Q1 - 1.5 * IQR # 0.0001785034309020805 """
    res = []
    for m in range(1, 26):
        thres = np.percentile(sentence_embeddings, m, interpolation = 'midpoint')
        removed_sentences_indices = []
        for (i,j,k,l) in lst:
            if i < thres:
                removed_sentences_indices.append([i, j, k, l])
                removed_sentences_indices.sort()
        res.append([removed_sentences_indices, len(removed_sentences_indices), m])
    return res


# In[13]:


to_find_avg = []
res = compute_threshold(df['sentence_embeddings'])
for i in range(1, len(res)):
    to_find_avg.append(res[i][1])
    print(f"When {i}th percentile is set as threshold, no. of duplicate sentences found are = {res[i-1][1]}")


# In[14]:


def create_dataframe(data: list[str], tdidf: list[float], row: list[int], offset: list[int]) -> pd.DataFrame:
    result = pd.DataFrame()
    result['data'] = data
    result['tfidf_value'] = tfidf
    result['row_no'] = row
    result['sentence_offset'] = offset
    return result


# In[15]:


percentile = int(input("Enter percentile: "))
data, tfidf, row, offset = [], [], [], []
for [i,j,k] in res:
    if k == percentile:
        for l in i:
            data.append(l[3])
            tfidf.append(l[0])
            row.append(l[2])
            offset.append(l[1])
result = create_dataframe(data, tfidf, row, offset)
result.to_csv('./using_exp_sqrt_mean.csv', index=False)
display(result)


# In[16]:


result['tfidf_value'][1]


# In[17]:


result.sort_values(by='sentence_offset', ascending=False)


# In[18]:


for i in range(100):
    print(f"{i+1}. {result['data'][i]}")


# In[19]:


print(f"Total time taken in complete program execution: {(int)((time.time()-start)//60)} mins {(int)((time.time()-start)%60)} secs")


# In[ ]:




