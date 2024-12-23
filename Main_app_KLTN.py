#!/usr/bin/env python
# coding: utf-8
import streamlit as st

import pandas as pd
import numpy as np
import os
import random
import base64

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import LSTM, Flatten, Dense, Masking, GRU
from tensorflow.keras.models import Sequential
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error, root_mean_squared_error

from datetime import datetime,timedelta
import pytz
from vnstock import *

import warnings
warnings.filterwarnings('ignore')



seed_value = 32

os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

st.set_page_config(page_title="Dự báo giá cổ phiếu bằng mô hình học sâu",page_icon="📊")

def set_seed(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def create_multistep_dataset(data, time_step, predict_step):
    X, y = [], []
    for i in range(time_step, len(data) - predict_step + 1):
        X.append(data[i-time_step:i, 0])  # Lấy 100 ngày trước đó làm đầu vào
        y.append(data[i:i+predict_step, 0])  # Lấy 1 ngày tiếp theo làm đầu ra
    return np.array(X), np.array(y)

def get_today_vietnam():
    # Đặt múi giờ Việt Nam
    vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    # Lấy thời gian hiện tại tại Việt Nam
    now = datetime.now(vietnam_tz)
    # Trả về ngày hôm nay dưới định dạng chuỗi YYYY-MM-DD
    return now.strftime('%Y-%m-%d')

def add_business_day(date):
    while True:
        date += timedelta(days=1)  # Cộng thêm 1 ngày
        if date.weekday() < 5:  # Kiểm tra nếu là ngày trong tuần (Thứ Hai đến Thứ Sáu)
            break
    return date



def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        background-color: rgba(255, 255, 255, 0.9); /* Điều chỉnh độ mờ ở đây */
        background-blend-mode: overlay;
    }}
    .custom-title {{
        color: #F05454;
    }}
    .stMarkdown, .stText {{
        color: #30475E !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

add_bg_from_local('background.png')

st.header(":red[Dự đoán giá cổ phiếu bằng mô hình LSTM - GRU]")

list=listing_companies()
list=list[(list['organTypeCode']=='DN')&((list['comGroupCode']=='HOSE') | (list['comGroupCode']=='HNX'))]
mcp=list.ticker.to_list()
mcp.sort()

stock = st.selectbox(
    ":red[Chọn cổ phiếu bạn muốn dự đoán]",
    mcp
)

if stock is not None:
    st.success(f"Bạn đã chọn cổ phiếu : "+stock)

    st.write('Bạn có muốn dự báo cổ phiếu này ?')

    if st.button("Nhấn nút để bắt đầu tính toán"):
        st.success("Đang tính toán để dự đoán giá cổ phiếu")
        start_date='2015-01-01'
        end_date=get_today_vietnam()
        df=stock_historical_data(stock,start_date,end_date)
        print_date=df.tail(1).time.values[0]
        

        
        df.set_index('time',inplace=True)
        df.index=pd.to_datetime(df.index,format='%d/%m/%Y')
        df=df.sort_values('time')
        pos_df=df.reset_index()
        
        train_data = df[['close']].values

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data = scaler.fit_transform(train_data)

        time_step = 20
        predict_step = 1
        X_train, y_train = create_multistep_dataset(train_data, time_step, predict_step)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        model = Sequential([
                            LSTM(units=128, return_sequences=True, input_shape=(time_step, 1)),  # LSTM layer đầu tiên
                            GRU(units=64, return_sequences=False),
                            Dense(units=64),
                            Dense(units=predict_step)
                            ])
        model.compile(optimizer='adam', loss='mean_absolute_error')

        model.fit(X_train, y_train, batch_size=32, epochs=100,shuffle=True)
        
        yhat_train = model.predict(X_train)
                
        yhat_train = scaler.inverse_transform(yhat_train)
        y_train=scaler.inverse_transform(y_train)
        
        R2 = round(r2_score(y_train, yhat_train), 1)
        RMSE=round(root_mean_squared_error(y_train, yhat_train),-2)
        MAE=round(root_mean_absolute_error(y_train, yhat_train),-2)

        if RMSE>MAE:
            bias=RMSE
        else:
            bias=MAE

        st.write('Mô hình có độ chính xác là :',R2*100,'%')

        test_data=train_data[-time_step:]
        
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)
        
        y_test = model.predict(test_data)

        y_test = scaler.inverse_transform(y_test)

        y_test=round(y_test[0,0],-2)
        
        print_date = add_business_day(print_date)
        print_date=print_date.strftime('%d-%m-%Y')
        st.write("Giá đóng cửa của ngày ",print_date ," sẽ nằm trong khoảng từ : ",y_test-bias," đến ",y_test+bias)





