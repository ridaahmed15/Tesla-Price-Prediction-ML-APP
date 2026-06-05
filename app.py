import streamlit as st
from sklearn import linear_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu

import plotly.express as px

st.set_page_config(layout="wide")

select= option_menu(
    menu_title=None,
    options=["Home","Predict Price","About Us","About Dev"],
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

if select=="About Us":
    st.set_page_config(layout="wide")
    st.image("tesla.jpg", use_container_width=True)

    col11,col12=st.columns(2)
    with col11:

        st.title("ABOUT US")
        st.write("The Tesla Model Y is more than just an electric SUV; it represents a journey toward a future where the driving experience is sustainable, intelligent, and fully connected. Our mission is to lead the transport revolution, blending cutting-edge innovation with daily practicality. With unparalleled range, sports-car-like performance, and a minimalist design, the Model Y is crafted for those who refuse to compromise on their journey.")
        st.write("Our goal is to provide accurate, insightful, and up-to-date content that helps individuals make informed decisions about electric vehicles. From vehicle features and performance insights to industry developments and ownership resources, we are committed to delivering information that is both valuable and easy to understand.We believe that the future of transportation is sustainable, intelligent, and connected. As electric vehicle technology continues to transform the automotive landscape, we aim to bridge the gap between innovation and everyday drivers by providing clear, reliable guidance and expert perspectives.We believe that the future of transportation is sustainable, intelligent, and connected. ")
    
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

    st.write("On the performance front, the vehicle remains highly efficient, with the Long Range All-Wheel Drive variant capable of reaching an EPA-estimated 327–329 miles on a single charge. Those who prioritize speed can enjoy the Performance trim, which is capable of accelerating from 0 to 60 mph in as little as 3.3 seconds. The vehicle maintains a strong safety standard by including Tesla’s camera-based suite, featuring adaptive cruise control, lane-keeping assist, and automated emergency braking as standard equipment on all models. Additionally, the option for Full Self-Driving (Supervised) is available, now with more flexible access through monthly subscription plans.")

    st.write("The 2026 Tesla Model Y introduces a new software feature known as Comfort Braking, which modulates brake pedal input for more refined, linear deceleration. Connectivity has been improved through Ultra Wideband technology, enhancing phone key performance, while the Wi-Fi hotspot now offers faster download speeds and greater range. Structural refinements to the body shell have increased stiffness, which, when combined with aerodynamic improvements from the new front splitter and rear diffuser, has increased the EPA-estimated range by over 20 miles.")

    st.write("To manage cabin temperature, the panoramic glass roof now reflects 26% more sunlight, and a factory-designed retractable sunshade is available as a new accessory. For increased utility, the rear seats now fold flat electrically, and the cargo area includes a magnetic, folding load-bay cover that stores in a dedicated under-floor cubby. Finally, the navigation system has been updated to show lifelike 3D renderings of vehicles at Supercharger sites, including specific icons to distinguish between different models like the Cybertruck and Model X.")
    
    col13,col14,col15=st.columns(3)

    with col13:
        st.success("☀️ Design & Utility")

    with col14:
        st.success("📱 Technology Hub")

    with col15:
        st.success("✨ Seating Comfort")

if select=="About Dev":
    st.title("RIDA AHMED")

    st.write("I am a Full-Stack Python Developer and Machine Learning App Developer with a deep passion for building scalable, data-driven solutions. My work focuses on bridging the gap between complex algorithms and user-friendly interfaces, ensuring that technology serves a practical, real-world purpose.")

    st.header("About This Project")
    st.write("This application is a testament to my commitment to MLOps and robust software engineering. To build this, I leveraged a powerful stack of tools:")
    st.write("Python 🐍: The core logic behind the data processing and predictive engine.")
    st.write("Scikit-Learn🧠: Used to power the machine learning model for accurate price predictions.")

    st.write("Pandas 📊: For efficient data manipulation and cleaning.")
    st.write("Plotly Express 📈: To create interactive, dynamic visualizations that react to real-time data.")

    st.write("Streamlit & Streamlit_option_menu 🌐: To provide a seamless, high-performance web interface.")

    st.write("I believe that a model is only as strong as its data. By implementing custom validation pipelines, I have ensured that this application remains reliable, filtering out abnormal inputs to maintain high-quality datasets. I am always exploring new ways to make AI more accessible and efficient.")
    st.write("I’m currently focused on growing as an early-career professional and am always eager to collaborate on meaningful projects. Feel free to reach out through my professional channels:")

    st.markdown("#### 🔗 CONNECT WITH ME!")

    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/ridaahmed-dev/)"
    )

    st.markdown(

        "[GITHUB](https://github.com/ridaahmed15)"


        )
    
    st.divider()











