import streamlit as st
from PIL import Image
import numpy as np
import cv2
from keras.models import load_model

st.title('Cats And Dogs Classification')

model = load_model('D:\ANSON\CV_level2\models\classify_cat_and_dog.h5')

image = st.file_uploader('Chọn Ảnh Chó Hoặc Mèo')
if image is not None:
    
    image_test = Image.open(image)
    st.image(image_test)
    im2arr = np.array(image_test)
    im2arr = np.resize(im2arr, (128,128,3))
    # arr2im = Image.fromarray(im2arr)

    # im2arr = np.expand_dims(im2arr, axis=0)
    im2arr = im2arr.reshape(-1,128,128,3)

    clicked = st.button('Recognize')
    if clicked:
        predict = np.argmax(model.predict(im2arr), axis=1)

        st.header('Kết Quả')
        if predict == 1:
            audio_file = open('D:\ANSON\CV_level2\ys_practice\\04-Convolution_neural_networks\Tieng-Cho-sua-remix-www_nhacchuongvui_com (mp3cut.net).mp3', 'rb')
        else:
            audio_file = open('D:\ANSON\CV_level2\ys_practice\\04-Convolution_neural_networks\\nhac-chuong-meo-meo-meo-meo-meo-tran-duc-bo (mp3cut.net).mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')