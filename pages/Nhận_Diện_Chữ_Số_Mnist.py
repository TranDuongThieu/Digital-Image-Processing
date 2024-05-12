import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import model_from_json
import numpy as np
import random
import cv2

st.markdown('<div>'
                '<h1 style="color: white;">Nhận diện chữ số Mnist</h1>'
            '</div>', unsafe_allow_html=True)

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/free-vector/gradient-grainy-texture_23-2148981502.jpg?w=1380&t=st=1701576249~exp=1701576849~hmac=c31e813f8bff0b5218da38e2815b2218f516bc79e9c21fc9b281f3c0dd5a557b");
    background-size: 100% 100%;
}
[data-testid="stHeader"]{
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
    right:2rem;
}
[data-testid="stSidebar"] > div:first-child {
    background-image: url("https://img.freepik.com/free-photo/abstract-textured-backgound_1258-30489.jpg?w=740&t=st=1701580457~exp=1701581057~hmac=25da83b2b1b6cf489fbc5a0ee890c79ff26156910c8b2e6a218c0a12bf939422");
    background-position: center;
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)

def tao_anh_ngau_nhien():
    image = np.zeros((10*28, 10*28), np.uint8)
    data = np.zeros((100,28,28,1), np.uint8)

    for i in range(0, 100):
        n = random.randint(0, 9999)
        sample = st.session_state.X_test[n]
        data[i] = st.session_state.X_test[n]
        x = i // 10
        y = i % 10
        image[x*28:(x+1)*28,y*28:(y+1)*28] = sample[:,:,0]    
    return image, data

if 'is_load' not in st.session_state:
    # load model
    model_architecture = 'D:/3year/XLAS/streamlit_final/streamlit_final/nhan_dang_chu_so_mnist_streamlit/digit_config.json'
    model_weights = 'D:/3year/XLAS/streamlit_final/streamlit_final/nhan_dang_chu_so_mnist_streamlit/digit_weight.h5'
    model = model_from_json(open(model_architecture).read())
    model.load_weights(model_weights)

    OPTIMIZER = tf.keras.optimizers.Adam()
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
                metrics=["accuracy"])
    st.session_state.model = model

    # load data
    (_,_), (X_test, y_test) = datasets.mnist.load_data()
    X_test = X_test.reshape((10000, 28, 28, 1))
    st.session_state.X_test = X_test

    st.session_state.is_load = True
    print('Lần đầu load model và data')
else:
    print('Đã load model và data rồi')

if st.button('Tạo ảnh'):
    image, data = tao_anh_ngau_nhien()
    st.session_state.image = image
    st.session_state.data = data

if 'image' in st.session_state:
    image = st.session_state.image
    st.image(image)

    if st.button('Nhận dạng'):
        data = st.session_state.data
        data = data/255.0
        data = data.astype('float32')
        ket_qua = st.session_state.model.predict(data)
        dem = 0
        s = ''
        for x in ket_qua:
            s = s + '%d ' % (np.argmax(x))
            dem = dem + 1
            if (dem % 10 == 0) and (dem < 100):
                s = s + '\n'   
        st.text(s)
