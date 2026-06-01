import streamlit as st
from sklearn import linear_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import plotly.express as px

st.set_page_config(layout="wide")

df=pd.read_csv("teslay.csv")

col5,col6,col7,col8= st.columns(4)
with col5:
    st.metric("Total Colors",df["color"].nunique())
with col6:
    st.metric("Average Prices Of Tesla Y Model",round(df["price"].mean(),2))  #2 mtlb decimal point
with col7:
    st.metric("Maximum Prices Of  Tesla Y Model",round(df["price"].max(),2))
with col8:
    st.metric("Minimum Prices Of  Tesla Y Model",round(df["price"].min(),2))

col3,col4=st.columns(2)

with col3:
    st.title("Total Value Count Color-wise")
    df2=df["color"].value_counts()
    st.dataframe(df2)
with col4:
     
    st.title("YEAR-WISE (model) PRICE TREND!")

    date_year=df.groupby("year")["price"].mean().reset_index()

    fig=px.line(
        date_year,
        x="year",
        y="price",
        title="Price Trend"
    )
    st.plotly_chart(fig)

col1,col2=st.columns(2)

with col1:
    st.subheader("RAW DATA SET")
    st.dataframe(df)

with col2:
        
        st.subheader("Color Wise Dataset")

        fig_price= px.bar(
            df,
            x="color",
            y="price",
            text="color",
            title="COLOR WISE PRICES",
            color="color"
        )
        st.plotly_chart(fig_price)