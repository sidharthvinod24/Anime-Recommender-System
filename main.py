import streamlit as st
import pandas as pd

from sentence_transformers import SentenceTransformer, util
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader
import torch
import scipy
import json
import pandas as pd
import gzip
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

animes = pd.read_csv("data/animelist.csv")
f = gzip.GzipFile("model/similar_title_embeddings.npy.gz", "r")

def recommend(anime):
    embeddings_recommender = np.load(f)

st.image("https://wallpaper.dog/large/20468918.jpg")
st.write("""
    # Anime Recommendation by SBERT
    Made by Sidharth Vinod
"""
)
st.subheader("Enter your Anime")
selected_anime = st.selectbox(
'So, which Anime did you enjoy?',
(animes['Title'].values))

st.multiselect('Choose the Medium Type', ['TV', 'Movie', 'ONA'])

