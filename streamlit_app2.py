import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('tips.csv')
st.dataframe(df.head(6))

st.markdown('---')
