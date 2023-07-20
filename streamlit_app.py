import streamlit as st
import pandas as pd
import numpy as np

st.write('I hate windows')
st.write(pd.DataFrame({
  'first_column':[1,2,3,4],
  'second_column':[10,20,30,40]
}))

data=pd.DataFrame({
  'first_column':[5,6,7,8],
  'second_column':[15,25,35,45]
})

st.write(data)


