#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import nltk
import re

from nltk.tokenize import WhitespaceTokenizer


# In[2]:


df = pd.read_csv('D:\Test_data.csv')


# In[3]:


df['text'] = df['text'].str.lower()


# In[4]:


tokenizer = WhitespaceTokenizer()


# In[5]:


# Токенизируем текст в отдельном столбце "tokens"
df['tokens'] = df['text'].apply(tokenizer.tokenize)


# In[6]:


# Определим списки ключевых слов для классификации фраз
greeting_phrases = ['здравствуйте', 'день', 'добрый']
introducing_phrases = ['зовут']
farewell_phrases = ['свидания', 'хорошего', 'доброго']
company_phrases = ['зовут', 'компания']
name_phrases = ['зовут']


# In[7]:


def classification(tokens):
    if len(set(greeting_phrases).intersection(set(tokens))) >= 1:
        return 'Greeting=True'
    if len(set(introducing_phrases).intersection(set(tokens))) >= 1:
        return 'Introducing=True'
    if len(set(farewell_phrases).intersection(set(tokens))) >= 1:
        return 'Farewell=True'
    else:
        return ""


# In[8]:


def company(tokens):
    if len(set(company_phrases).intersection(set(tokens))) >= 2:
        return tokens[tokens.index('компания')+1]
    else:
        return ''


# In[9]:


def name(tokens):
    if len(set(name_phrases).intersection(set(tokens))) == 1:
        return tokens[tokens.index('зовут')+1]
    else:
        return ''


# In[10]:


# Добавим столбец "insigth", где зафиксируем наши наблюдения
df['insight'] = df['tokens'].apply(lambda tokens: classification(tokens))


# In[11]:


# Добавим столбец "company_name" с наименованием компании
df['company_name'] = df['tokens'].apply(lambda tokens: company(tokens))


# In[12]:


# Добавим столбец, где указываются имена менеджеров
df['manager_name'] = df['tokens'].apply(lambda tokens: name(tokens))


# In[13]:


df.head()


# In[14]:


df.to_csv('test_data.csv')


# In[ ]:




