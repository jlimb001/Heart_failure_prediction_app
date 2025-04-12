import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items
import numpy as np
import plotly.express as px
import streamlit as st
import pickle
# from streamlit_extras.add_vertical_space import add_vertical_space

### Default best practice structure when you have multiple cols:
# Define streamlit_element
# streamlit_element = st.columns, st.sidebar, st.tabs, etc. (see Layouts & Containers in docs)
# with streamlit_element:
    # Write some functions here that define what's inside
    # plotly_fig = px.line(data, x='col1', y='col2')
    # st.plotly_chart(plotly_fig)
    # st.markdown("Some markdown formatted text.")
    
st.header('Customer Churn Analysis')
st.markdown(' ##### Exploratory Data Analysis')

data = pd.read_csv("https://raw.githubusercontent.com/sabinagio/data-analytics/main/data/customer_churn.csv")
st.write(data)

num_cols = data.select_dtypes('number') 

col1, col2 = st.columns(2)

with col1:
    selection = st.selectbox('', data.columns)
    if selection in num_cols:
        st.plotly_chart(px.histogram(data, x=selection))
    else:
        st.plotly_chart(px.histogram(data, y=selection))


