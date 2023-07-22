#$ pipreqs /home/project/location
#Successfully saved requirements file in /home/felisaav/streamlit/requirements.txt

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('tips.csv')
st.dataframe(df.head(6))

st.markdown('---')

st.header('Matplotlib')

sex_dist=df['sex'].value_counts()
st.dataframe(sex_dist)

with st.container():
  fig,ax=plt.subplots()
  ax.pie(sex_dist, autopct='%0.2f%%',labels=['Male','Female'])
  st.pyplot(fig)

  fig,ax=plt.subplots()
  ax.bar(sex_dist.index,sex_dist)
  st.pyplot(fig)

st.markdown('---')

with st.container():
  c1,c2=st.columns(2)

  with c1:
    fig,ax=plt.subplots()
    ax.pie(sex_dist,autopct='%0.2f%%',labels=['Male','Female'])
    st.pyplot(fig)

  with c2:
    fig,ax=plt.subplots()
    ax.bar(sex_dist.index,sex_dist)
    st.pyplot(fig)

  with st.expander('Click to see the data'):
    st.dataframe(sex_dist)
