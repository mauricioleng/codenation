#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import (KBinsDiscretizer, StandardScaler)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import (CountVectorizer,TfidfVectorizer)
from sklearn.datasets import fetch_20newsgroups


# In[4]:


# Algumas configurações para o matplotlib.

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[5]:


countries = pd.read_csv("countries.csv", decimal = ",")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
countries.Country = countries.Country.str.strip()
countries.Region = countries.Region.str.strip()


# In[6]:


#Questao 1
q1 = list(countries.Region.unique())
q1.sort()


# In[7]:


#Questao 2
disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')


# In[15]:


disc.fit(countries.Pop_density.values.reshape(-1,1))
Pop_disc = disc.transform(countries.Pop_density.values.reshape(-1,1))


# In[17]:


quart =  np.quantile(Pop_disc, 0.9)


# In[28]:


Pop_disc[Pop_disc > quart].size


# In[7]:


#Questao 3
'''Como o one_hot_encoder cria uma coluna pra cada valor existente, basta checar a quantidade de valores unicos existentes 
mais um de dados missing'''
int(countries.Region.nunique() + countries.Climate.nunique()+1)


# In[34]:


#Questao 4
num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])


# In[35]:


num_pipe.fit(countries.select_dtypes([np.number]))


# In[43]:


# num_pipe.transform([test_country[2:]])[0][countries.columns.get_loc('Arable')].round(3)


# In[48]:


#Questao 5
countries.Net_migration.plot(kind='box');


# In[60]:


q1 = countries.Net_migration.quantile(0.25)
q3 = countries.Net_migration.quantile(0.75)
iq = q3-q1

sup_limit = q3 + 1.5 * iq
inf_limit = q1 - 1.5 *iq

out_inf = (countries.Net_migration < inf_limit).sum()
out_sup = (countries.Net_migration > sup_limit).sum()
total = countries.Net_migration.size


# In[62]:


print(total, out_inf, out_sup)


# In[64]:


#Questao 6
categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[68]:


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(newsgroup.data)


# In[73]:


X[:,vectorizer.vocabulary_['phone']].sum()


# In[77]:


#Questao 7
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroup.data)
X[:,vectorizer.vocabulary_['phone']].sum()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[6]:


def q1():
    a1 = list(countries.Region.unique())
    a1.sort()
    return a1


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[7]:


def q2():
    disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    disc.fit(countries.Pop_density.values.reshape(-1,1))
    Pop_disc = disc.transform(countries.Pop_density.values.reshape(-1,1))
    quart =  np.quantile(Pop_disc, 0.9)
    return int(Pop_disc[Pop_disc > quart].size)


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[9]:


def q3():
    return int(countries.Region.nunique() + countries.Climate.nunique()+1)


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[36]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[11]:


def q4():
    num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
    num_pipe.fit(countries.select_dtypes([np.number]))
    return float(num_pipe.transform([test_country[2:]])[0][9].round(3))


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[12]:


def q5():
    q1 = countries.Net_migration.quantile(0.25)
    q3 = countries.Net_migration.quantile(0.75)
    iq = q3-q1

    sup_limit = q3 + 1.5 * iq
    inf_limit = q1 - 1.5 *iq

    out_inf = (countries.Net_migration < inf_limit).sum()
    out_sup = (countries.Net_migration > sup_limit).sum()
    return (int(out_inf), int(out_sup), False)


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[76]:


def q6():
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(newsgroup.data)
    return int(X[:,vectorizer.vocabulary_['phone']].sum())


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[14]:


def q7():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(newsgroup.data)
    return float(round(X[:,vectorizer.vocabulary_['phone']].sum(),3))

