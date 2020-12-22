import streamlit as st
import base64
import joblib

import requests

from datetime import datetime
import pytz

import re
import nltk
from nltk.corpus import stopwords # importing 'stopwords' to this notebook
from nltk.stem.porter import PorterStemmer ## stemming of words
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import plotly.express as px

# Loading required pkl files
model = joblib.load("pkl_files/Direct_news_impact/Apple_stock_behaviour.pkl")
cv_model= joblib.load("pkl_files/Direct_news_impact/Count_Vec_model.pkl")

html_temp = """
    <div style=padding:10px">
    <h1 style="color:white;text-align:left;">APPL Daily </h1>
    </div>
    <div>
    <h3 style="color:white;text-align:left;"> Click on the buttons below to know Closing price behavior predictions:</h3>
    <h3 style="color:white;text-align:left;"> 1. Based on effect of popular news of the day(the date in New York at the time you are using this app) related to Apple, Mac, iPhone, IOS</h3>
    <h3 style="color:white;text-align:left;"> 2. Based on effect of sentiments of Popular news of the day on AAPL</h3>
    <h3 style="color:white;text-align:left;"> 3. Future(15 days) interactive graph forecast based on historic stock trends</h3>
    <h3 style="color:white;text-align:left;"> You can analyze all the three predictions and plan your next move accordingly.</h3> 
    <h3 style="color:white;text-align:left;"> But remember the fact, "Stock Markets are unpredictable at many times". We hope this will help you get profits from AAPL:)</h3>
    <h4 style="color:white;text-align:left;"> Due to heavy computations the app might run a bit slow.</h4>
    """
html_temp1= """
    <div>
    <h2 style="color:yellow;text-align:left;">APPL might rise</h3>
    </div>
    """
html_temp2= """
    <div>
    <h2 style="color:yellow;text-align:left;">APPL might fall</h3>
    </div>
    """

st.markdown(html_temp,unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg("Web_app_graphics/Theme.jpg")


# Cleanig all the news(stacked together)
def nlp_preprocessing(news_input):
    news= re.sub('[^a-zA-Z]', ' ', news_input)
    news= news.lower()
    news= news.split()
    news=[word for word in news if not word in set(stopwords.words('english'))]
    pe = PorterStemmer()
    news=[pe.stem(word) for word in news if not word in set(stopwords.words('english'))]
    news= ' '.join(news)
    return([news])


def fetch_keyword_news(your_keyword_list, date, apiKey):
    all_news= []
    for i in range(len(your_keyword_list)):
        url = ('http://newsapi.org/v2/everything?'
               'q=' + str(your_keyword_list[i]) + '&'
               'from='+str(date)+'&'
               'language=en&'
               'sortBy=popularity&'
               'apiKey=' + str(apiKey))
        response = requests.get(url).json()
        article= response["articles"]
        
        for i in range(len(article)):
            all_news.append(article[i]["content"])
            
    return(all_news)


def decision(list_of_out_classes):
    z= 0.0  # will store number of "zero" class
    o= 0.0  # will store number of "one" class
    for i in range(len(list_of_out_classes)):
        if list_of_out_classes[i]==0:
            z= z+1
        else:
            o= o+1

    if o>z: # If probabity of occuring "1" is more
        st.markdown(html_temp1,unsafe_allow_html=True)
        
    else: # If probablity of ocuring "0" is more or exactly equal
        st.markdown(html_temp2,unsafe_allow_html=True)

def into_the_future(future):
    forecast_point = pd.read_csv("Dataframes_generated/data_forecast.csv")
    # future = int(input())
    if future>25:
        future=25

    today_index = forecast_point[forecast_point["Date"] == str(datetime.now(pytz.timezone('US/Eastern')).date())].index.item()
    post_index = today_index + future
    all_index = np.arange(today_index, post_index)

    x = []
    y = []
    for i in all_index:
        x = np.append(x, forecast_point.iloc[i, 1])
        y = np.append(y, forecast_point.iloc[i, 2])

    future_df = pd.DataFrame(list(zip(x, y)), columns=["Date", "Close"])
    fig = px.line(future_df, x="Date", y="Close", title="Forecast of Apple Stock using Neural Prophet")
    st.plotly_chart(fig)
        

apiKey_list=['20d469827dbb4eb384d22490ea5df888', '75d16a33351a44969f3a5ac41eb7cf20','6496b9cb73c34054a8b58a3dee86c672',
'5e6b9203fe4247369e70351f0ab2b1b3','3907a8165aec4be89b2e12f3a5ad541a','79bbb20ec53e4d1b85c2caca76402488',
'388eff313e1a4d399d55ebb19d4db4cd','a0936894b7904a03a4c35ca6627ebc33','c15a4b03480c4081bd3d184bc8559f23',
'c05beec776fa4b1fbcc46bdad8efa951','be98dcb51dd64998ad08a6dd2c5f9e80','376c9dfc704748279df3e6f30a751a1e',
'46660f56bd6e45f986fea91dc87b1fc1','5dbf8944da394e4ca003b7fea5b736c5','31ba8f79f57d41c8b03d3334760154b3',
'723db6ceb2e8465daffa882be629d6fb','1a88fcc99b0b41de902fcdbc45bd4a97','06d039549c914c78a46d2c0c137b7f7c',
'1b47a4f26fc949c4ad280f9bfb81cd5d'] # List of different API Keys

# ---------------------------------------------------------------------------------------------------------------------------------

if st.button("Fetch News and Predict"):
    weekno = datetime.now(pytz.timezone('US/Eastern')).date().weekday()

    if weekno<5:
        random.shuffle(apiKey_list)
        
        input_list= fetch_keyword_news(["Apple", "Mac", "iPhone", "Google"], datetime.now(pytz.timezone('US/Eastern')).date(), apiKey_list[3]) # List which will contain all the latest news inputs

        filtered_list= list(filter(None, input_list)) 

        out_class_list=[] # List which will store output classes

        for i in range(len(filtered_list)):
            inp_preprocessed= nlp_preprocessing(filtered_list[i])
            inp= cv_model.transform(inp_preprocessed)
            out_class_list.append(model.predict(inp))

        html_temp3= """
        <div>
        <h3 style="color:white;text-align:left;">Some Top News:</h3>
        </div>
        """
        st.markdown(html_temp3,unsafe_allow_html=True)
        for i in range(0, 5):
            news= filtered_list[i]
            html_view_news_temp='<p style="background-color:white">%s</p>' % str(news)
            st.markdown(html_view_news_temp,unsafe_allow_html=True)

        html_temp4= """
        <div>
        <h1 style="color:white;text-align:left;">Prediction:</h1>
        </div>
        """
        st.markdown(html_temp4,unsafe_allow_html=True)

        decision(out_class_list)

    else:
        html_temp5= """
        <div>
        <h1 style="color:white;text-align:left;">Weekend: Market is Closed</h1>
        </div>
        """
        st.markdown(html_temp5,unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------------------------------------------------
# Loading required pkl files
sen_cv_model= joblib.load("pkl_files/News_sentiment_impact/cv_model.pkl")
sen_feature_scaler= joblib.load("pkl_files/News_sentiment_impact/Feature_scaler.pkl")
sen_get_sentiment_proba= joblib.load("pkl_files/News_sentiment_impact/Txt_sentiment.pkl")
sen_predict_stock_behave= joblib.load("pkl_files/News_sentiment_impact/Historic_Apple_News_Sentiment.pkl")

if st.button("Prediction based on News Sentiment"):
    weekno = datetime.now(pytz.timezone('US/Eastern')).date().weekday()

    if weekno<5:
        random.shuffle(apiKey_list)    
        random.shuffle(apiKey_list)
            
        input_list= fetch_keyword_news(["Apple", "Mac", "iPhone", "Google"], datetime.now(pytz.timezone('US/Eastern')).date(), apiKey_list[3]) # List which will contain all the latest news inputs

        filtered_list= list(filter(None, input_list)) 

        out_class_list=[] # List which will store output classes

        for i in range(len(filtered_list)):
            inp_preprocessed= nlp_preprocessing(filtered_list[i])
            inp= sen_cv_model.transform(inp_preprocessed)
            sen_classes= sen_get_sentiment_proba.predict_proba(inp)
            scaled= sen_feature_scaler.transform(sen_classes)
            out_class_list.append(sen_predict_stock_behave.predict(scaled))

        decision(out_class_list)
    else:
        html_temp6= """
        <div>
        <h1 style="color:white;text-align:left;">Weekend: Market is Closed</h1>
        </div>
        """
        st.markdown(html_temp6,unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------------------------------------------------
      
if st.button("Predict based on stock trends"):
    html_temp7= """
        <div>
        <h1 style="color:white;text-align:left;">Forecast Graph till 25th April 2022:</h1>
        </div>
        """
    st.markdown(html_temp7,unsafe_allow_html=True)
    st.image("Web_app_graphics/Closing_forecast.jpeg")
    # future_days= st.slider('Slide to select the number of days you want to forecast. Range-(5, 25)', min_value= 5, max_value= 25, step=1)
    html_temp8= """
        <div>
        <h1 style="color:white;text-align:left;">Forecast of next 15 days from now:...</h1>
        </div>
        """
    st.markdown(html_temp8,unsafe_allow_html=True)
    into_the_future(16)


html_temp9= """
    <div>
    <h3 style="color:white;text-align:left;">Incase you get a "KeyError" or "ConnectionError", please refresh and try again.</h3>
    </div>
    """
st.markdown(html_temp9,unsafe_allow_html=True)