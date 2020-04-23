# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

from IPython.core.interactiveshell import InteractiveShell
# Used to get multiple outputs per cell
InteractiveShell.ast_node_interactivity = "all"

# %% [markdown]
# ### Support Functions
# #### Clean data of stop words and punctuation

# %%
def clean_data(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words() and word.isalpha()]
    print (tokens_without_sw)
    return tokens_without_sw

# %% [markdown]
# ### Main implementation
# #### Load Datasets

# %%
tweets_dataset = pd.read_csv('tweets_info.csv') 
user_dataset = pd.read_csv('user_info.csv') 
tweets_dataset.sort_values(by='Likes count', ascending = False)
user_dataset

# %% [markdown]
# #### Clean tweets

# %%
tweets_dataset['Clean Tweet'] = tweets_dataset['Tweet'].apply(clean_data)
print(tweets_dataset)