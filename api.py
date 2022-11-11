import streamlit as st 
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

def main():

    model1 = tf.keras.models.load_model('mobilenetv2_party_classifier.hdf5') ## Load trained model
    model2 = tf.keras.models.load_model('resnet50_party_classifier.hdf5')

    class_names = ['Aam Aadmi Party', 'All India Anna Dravida Munnetra', 'All India Forward Bloc', 'All India Majlis-E-Ittehadul Muslimeen', 
                  'All India NR Congress', 'All India United Democratic Front', 'All Jharkand Students Union', 'Bahujan Samaj Party', 
                  'Bharatiya Janata Party', 'Biju Janata Dal', 'Bodoland Peoples Front', 'Communist Party of India', 
                  'Communist Party of India Liberation Marxist', 'Communist Party of India Marxist', 'Desiya Murpokku Dravida Kazhagam', 
                  'Dravida Munnetra Kazhagam', 'Goa Forward Party', 'Indian National Congress', 'Indian National Lok Dal', 
                  'Indian Union Muslim League', 'Indigenous People Front of Tripura', 'Jammu Kashmir National Conference', 
                  'Jammu Kashmir Peoples Democratic', 'Janata Congress Chattisgarh', 'Janata Dal Secular', 'Janata Dal United', 
                  'Jannayak Janta Party', 'Lok Jansakthi Party', 'Maharashtra Navnirman Sena', 'Mizoram National Front', 
                  'Mizoram Peoples Conference', 'Naga Peoples Front', 'Nationalist Congress Party', 'Nationalist Democratic Progressive Party', 
                  'Pattali Makkal Katchi', 'Peoples Democratic Front', 'Peoples Party of Arunachal', 'Rashtriya Janata Dal', 'Rashtriya Loktantrik Party',
                  'Rastriya Lok Dal', 'Revolutionary Socialist Party', 'Samajwadi Party', 'Shirmoni Akali Dal', 'Shiv Sena', 'Sikkim Democratic Front', 
                  'Sikkim Krantikari Morcha', 'Telangana Rastra Samithi', 'Telugu Desam Party', 'Yuvajana Sramika Rythu Congress Party'] 
                  
 # Party 

    st.header("Party Symbol Classifier for VVPAT counting")
    st.markdown("This app demonstrates classification of votes. Upload images only in the given format.")

    uploaded_file = st.file_uploader('Upload party symbol image: ',type=['jpg','jpeg'])

    if uploaded_file:
        # bytes_data = uploaded_file.getvalue()
        # image = Image.frombuffer("RGB",data=bytes_data, size=(180, 180))
        print(uploaded_file)
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        if len(image_array.shape) == 2:
            image_array = np.repeat(image_array[...,np.newaxis],3,-1)
        image = cv2.resize(image_array, (180, 180))
        st.image(image, caption='Uploaded Party Symbol.', use_column_width=True)
        #col1, col2 = st.columns(2)
        #with col1:
        label1 = class_names[np.argmax(model1.predict(np.expand_dims(image, 0)))]
        label2 = class_names[np.argmax(model2.predict(np.expand_dims(image, 0)))]
        st.write("Prediction of MobilnetV2 model:\n", label1)
        st.write("Prediction of Resnet50 model:\n", label2)
        #with col2:

if __name__ == "__main__":
    main()