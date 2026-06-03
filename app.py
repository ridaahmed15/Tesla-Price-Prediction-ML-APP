import streamlit as st
from sklearn import linear_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu

import plotly.express as px

st.set_page_config(layout="wide")

select= option_menu(
    menu_title=None,
    options=["Home","Predict Price","About Tesla Y"],
    orientation="horizontal"
)

df=pd.read_csv("teslay.csv")


if select=="Home":
    st.title("Tesla Y model Price Analysis for 2024 & 2025")

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

if select=="Predict Price":
    st.title("Predict The Price Of Tesla Y Model")

    le=LabelEncoder()

    df['color']=le.fit_transform(df['color'])

    x=df[["year","km","color"]]
    y=df.price


    #st.dataframe(x)  #ccomment krdie kiu k   labelencoder ki need ni hui
    #st.dataframe(y)

    model=linear_model.LinearRegression()
    model.fit(x,y)   #model ko fit krk predictions deta  hai

    col9,col10=st.columns(2)

    with col9:
        st.info("By Selecting Year, Color & Km's")

        year=st.selectbox("Enter Year:",[2020,2021,2022,2023,2024,2025,2026,2027,2028])
        km=int(st.number_input("Enter Kilometers",min_value=6000))
        color=st.selectbox("Select Color",le.classes_)
        pred=st.button("Predict")

    with col10:
        col_ch=le.transform([color])[0] #color change
        predicted_price=model.predict([[year,km,col_ch]])
        score=model.score(x,y)         #pehle hi upr accuracy batata rahyga
        accuracy=int(score*100)
        st.metric("Accuracy Of Current Model As Per Recent Data",accuracy,"%")
        if pred:
            if color:
                col_ch=le.transform([color])[0] #color change
                predicted_price=model.predict([[year,km,col_ch]])  #yahan predict krny k bd b wahi cheez hgi price k sth
                st.balloons()
                score=model.score(x,y)
                accuracy=int(score*100)

                st.subheader("How Accurate The Predicted Price Is!")
                st.info(f"{accuracy}%")
                st.subheader("Here Is Your Predicted Price For Tesla Y")
                predic=int(predicted_price[0])   #2 k decimal me round krwa k predic k variable me dal dia or phr nechy print krwadia
                st.info(predic)

            #JO USER KA DATA HGA WOH CSV FILE ME SAVE HJYGA

            existing_data=pd.read_csv("teslay.csv")
            new_data=pd.DataFrame({"year":[year],"km":[km],"color":[color],"price":[predic]})
            updated_data=pd.concat([existing_data,new_data])
            updated_data.to_csv("teslay.csv",index=False)

if select=="About Tesla Y":
    st.set_page_config(layout="wide")
    st.image("tesla.jpg", use_container_width=True)

    col11,col12=st.columns(2)
    with col11:

        st.title("ABOUT US")
        st.write("The Tesla Model Y is more than just an electric SUV; it represents a journey toward a future where the driving experience is sustainable, intelligent, and fully connected. Our mission is to lead the transport revolution, blending cutting-edge innovation with daily practicality. With unparalleled range, sports-car-like performance, and a minimalist design, the Model Y is crafted for those who refuse to compromise on their journey.")
        st.write("Our goal is to provide accurate, insightful, and up-to-date content that helps individuals make informed decisions about electric vehicles. From vehicle features and performance insights to industry developments and ownership resources, we are committed to delivering information that is both valuable and easy to understand.We believe that the future of transportation is sustainable, intelligent, and connected. As electric vehicle technology continues to transform the automotive landscape, we aim to bridge the gap between innovation and everyday drivers by providing clear, reliable guidance and expert perspectives.We believe that the future of transportation is sustainable, intelligent, and connected. As electric vehicle technology continues to transform the automotive landscape, we aim to bridge the gap between innovation and everyday drivers by providing clear, reliable guidance and expert perspectives.Whether you are researching your next vehicle, exploring the benefits of electric mobility, or staying informed about the latest advancements, our platform is dedicated to supporting your journey with professionalism, integrity, and a passion for the future of transportation.")
    
    with col12:
        st.image("teslaz.png",use_container_width=True)

    st.subheader("WHY CHOOSE US?")
    st.write("We are committed to bridging the gap between innovation and everyday drivers by providing clear, reliable guidance and expert perspectives. Our goal is to deliver accurate, insightful, and up-to-date content that helps you make informed decisions about electric vehicles. Whether you are researching your next vehicle, exploring the benefits of electric mobility, or staying informed about the latest advancements, our platform is dedicated to supporting your journey with professionalism, integrity, and a passion for the future of transportation.")
    st.subheader("FEATURE-FOCUSED:")
    st.write("Tesla Model Y is designed to prioritize both family needs and the passion for driving:")
    st.write("✨ The minimalist interior and panoramic glass roof provide an airy, modern feel with flexible seating options.")
    st.write("🛡️ Advanced safety features, including autopilot capabilities, ensure a secure and stress-free driving experience.")
    st.write("⚡ Electric torque delivers instant acceleration and smooth handling, providing confidence on every turn.")
    st.write("📱 With a large touchscreen display and fast-charging capabilities, the Model Y acts as a smart, mobile technology hub.")

    st.image("teslascar.jpg", use_container_width=True)






