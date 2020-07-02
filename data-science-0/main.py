#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


n_obs_col = (black_friday.shape[0],black_friday.shape[1])
n_obs_col


# In[4]:


black_friday.head()


# In[5]:


crit1 = black_friday['Gender'] == 'F'
crit2 = black_friday['Age'] == '26-35'
q2a = int(black_friday[crit1 & crit2].shape[0])


# In[6]:


q3a = int(black_friday['User_ID'].nunique())


# In[7]:


q4a = int(black_friday.dtypes.nunique())


# In[8]:


q5a = float(black_friday.isnull().sum().max()/black_friday.shape[0])


# In[9]:


q6a = int(black_friday.isnull().sum().max())


# In[10]:


q7a = list(dict(black_friday.Product_Category_3.value_counts()).keys())[0]


# In[11]:


max_value = black_friday.Purchase.max()
min_value = black_friday.Purchase.min()
black_friday['Purchase_Norm'] = (black_friday.Purchase - min_value) / (max_value - min_value)


# In[12]:


q8a = float(black_friday.Purchase_Norm.mean())


# In[13]:


m = black_friday.Purchase.mean()
std = black_friday.Purchase.std()
black_friday['Purchase_Std'] = (black_friday.Purchase - m) / (std)


# In[14]:


critn_1 = black_friday.Purchase_Std <= 1
critn_2 = black_friday.Purchase_Std >= -1
q9a = int(len(black_friday.Purchase_Std[critn_1 & critn_2]))


# In[6]:


q10a = (black_friday.Product_Category_2.isnull() & black_friday.Product_Category_3.notnull()).sum()==0


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[ ]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return n_obs_col
    pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[ ]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return q2a
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[ ]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return q3a
    pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[ ]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return q4a
    pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[ ]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return q5a
    pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[ ]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return q6a
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[ ]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return q7a
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[ ]:


def q8():
    # Retorne aqui o resultado da questão 8.
    return q8a
    pass


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[ ]:


def q9():
    # Retorne aqui o resultado da questão 9.
    return q9a
    pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[ ]:


def q10():
    # Retorne aqui o resultado da questão 10.
    return q10a
    pass

