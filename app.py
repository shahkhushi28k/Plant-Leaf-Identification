import base64
from PIL import Image
import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu
import pandas as pd

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('models/efficientnetb3-Plant Village Disease-99.26.h5')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size = (224,224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #converting single image to batch
    prediction = model.predict(input_arr)
    return np.argmax(prediction) #return index of max element



    

main_bg = "bg2.jpeg"
main_bg_ext = "jpeg"

side_bg = "bg.jpeg"
side_bg_ext = "jpeg"





#sidebar

with st.sidebar:
    
    app_mode = option_menu('Plant Disease Detector',
                          
                          ['Home',
                           'About',
                           'Prediction',
                          ],
                          icons=['house','info','tree'],
                          default_index=0)
                           
#--------------------------------------------------------------->>Home<<---------------------------------------------------------------------








selected_option = None



#Home Page
if app_mode == 'Home':


    #st.title('RootiFy')
    css = """
    <style>
    @keyframes slide-in {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(0); }
    }
    .slide-in {
        animation: slide-in 2s ease-in-out;
    }
    </style>
    """

    # Display the CSS styles
    st.write(css, unsafe_allow_html=True)

    # Create animated "Rootify" text
    st.write("<div class='slide-in'><h1>Rootify</h1></div>", unsafe_allow_html=True)
    st.divider()
    image_path = 'home.gif'
    st.image(image_path,width=700,use_column_width='always')
    st.divider()
    st.subheader('Welcome to Rootify: Your Plant Health Companion')
    st.write('\n')
    st.markdown('<div style="text-align: justify;">Rootify is your ultimate solution for plant disease detection and care. In a world where plants are not just greenery but cherished members of your life, their health is our top priority. Rootify is here to make sure your plants thrive.</div>', unsafe_allow_html=True)
    st.write('\n')
    st.markdown('<div style="text-align: justify;">At Rootify, we understand the deep connection you share with your plants. Whether you are an experienced gardener, a passionate plant lover, or a curious beginner in the world of botany, our application simplifies the process of identifying, preventing, and managing plant diseases, ensuring your plants remain robust and vibrant.</div>', unsafe_allow_html=True)
    st.write('\n')
    st.subheader('Caring for Your Green Companions')
    st.write('\n')
    st.markdown('<div style="text-align: justify;"><b>We believe that every plant has a story, and we are here to help you ensure that story is one of vitality and growth. With our cutting-edge technology and user-friendly interface, we are dedicated to making plant care accessible, enjoyable, and rewarding for all. Your plants deserve the best, and Rootify is your dedicated companion on the journey to healthier and happier green companions.<b></div>', unsafe_allow_html=True)
    st.write('\n')
    features = [
    "**Disease Detection**: Our advanced algorithms swiftly and accurately identify plant diseases. Just snap a photo, and we'll provide a diagnosis along with recommended treatments.",
    "**Plant Library**: Explore our extensive plant species library, complete with care guides, growth stages, and disease profiles. Learn about your plants and how to nurture them to perfection.",
    "**Community Support**: Join a community of like-minded plant enthusiasts. Share your plant stories, seek advice, and help others with your own experiences in plant care and disease management."
    ]

    # Display the features as a list
    st.write("**Key Features**")
    for feature in features:
        st.write(f"- {feature}")
    st.write('\n')
    image_path = 'working.gif'
    st.image(image_path,width=500,use_column_width='always')
    st.write('Our Promise')
    st.write('\n')
    st.markdown('<div style="text-align: justify;"><b>At Rootify, we are committed to your plants well-being. We believe that every plant has the potential to flourish, and we are here to help you unlock that potential. Together, we will ensure your green companions enjoy long, healthy, and vibrant lives..<b></div>', unsafe_allow_html=True)
    st.write('\n')
    st.markdown('<div style="text-align: justify;"><b>Welcome to Rootify, where we celebrate the beauty of nature and the joy of nurturing it. Join us on this green journey, and lets watch your plants thrive and flourish together.<b></div>', unsafe_allow_html=True)


#About Page
if app_mode == 'About':
    st.header('Welcome to About Page..!')
    st.subheader('Case Study')
    image_path = 'about.png'
    st.image(image_path,width=500)
    st.markdown('<div style="text-align: justify;">Human society needs to increase food production by an estimated 70% by 2050 to feed an expected population size that is predicted to be over 9 billion people. Currently, infectious diseases reduce the potential yield by an average of 40% with many farmers in the developing world experiencing yield losses as high as 100%. The widespread distribution of smartphones among crop growers around the world with an expected 5 billion smartphones by 2020 offers the potential of turning the smartphone into a valuable tool for diverse communities growing food. One potential application is the development of mobile disease diagnostics through machine learning and crowdsourcing. Here we announce the release of over 50,000 expertly curated images on healthy and infected leaves of crops plants through the existing online platform PlantVillage. We describe both the data and the platform. These data are the beginning of an on-going, crowdsourcing effort to enable computer vision approaches to help solve the problem of yield losses in crop plants due to infectious diseases.</div>', unsafe_allow_html=True)
    st.write('\n')
    st.markdown('We can Predict the following Plants: ')




    # Sample data with plant names and image paths
    data = {
        'Plant Name': [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ],
        'Image Path': [
            'input/Apple___Apple_scab.jpeg', 'input/Apple___Black_rot.jpeg',
            'input/Apple___Cedar_apple_rust.jpeg', 'input/Apple___healthy.jpeg',
            'input/Blueberry___healthy.jpeg', 'input/Cherry_(including_sour)___healthy.jpeg',
            'input/Cherry_(including_sour)___Powdery_mildew.jpeg', 'input/Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot.jpeg',
            'input/Corn_(maize)___Common_rust_.jpeg', 'input/Corn_(maize)___healthy.jpg',
            'input/Corn_(maize)___Northern_Leaf_Blight.jpg', 'input/Grape___Black_rot.jpeg',
            'input/Grape___Esca_(Black_Measles).jpeg', 'input/Grape___healthy.jpeg',
            'input/Grape___Leaf_blight_(Isariopsis_Leaf_Spot).jpeg', 'input/Orange___Haunglongbing_(Citrus_greening).jpeg',
            'input/Peach___Bacterial_spot.jpeg', 'input/Peach___healthy.jpeg',
            'input/Pepper,_bell___Bacterial_spot.jpeg', 'input\Pepper,_bell___healthy.jpeg',
            'input/Potato___Early_blight.jpeg', 'input/Potato___healthy.jpeg',
            'input/Potato___Late_blight.jpeg', 'input/Raspberry___healthy.jpeg',
            'input/Soybean___healthy.jpeg', 'input/Squash___Powdery_mildew.jpeg',
            'input/Strawberry___healthy.jpeg', 'input/Strawberry___Leaf_scorch.jpeg',
            'input/Tomato___Bacterial_spot.jpeg', 'input/Tomato___Early_blight.jpeg',
            'input/Tomato___healthy.jpeg', 'input/Tomato___Late_blight.jpeg',
            'input/Tomato___Leaf_Mold.jpeg', 'input/Tomato___Septoria_leaf_spot.jpeg',
            'input/Tomato___Spider_mites Two-spotted_spider_mite.jpeg', 'input/Tomato___Target_Spot.jpeg',
            'input/Tomato___Tomato_mosaic_virus.jpeg', 'input/Tomato___Tomato_Yellow_Leaf_Curl_Virus.jpeg'
        ]
    }

    df = pd.DataFrame(data)

# Display images in a 4x10 matrix with names

    num_rows = 10
    num_cols = 4
    total_images = len(df)

    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col in cols:
            index = row * num_cols + cols.index(col)
            if index < total_images:
                image = Image.open(df['Image Path'][index])
                col.image(image, caption=df['Plant Name'][index], use_column_width=True)
    
   


    
   

#Prediction Page
if app_mode == 'Prediction':

    st.header('Welcome to Prediction Page..!')
    test_image = st.file_uploader('Upload Image')
    if(test_image is not None):
        st.image(test_image,width=4,use_column_width='always')
        
    #Predcit Button
    if(st.button('Predict')):
        #st.write('Prediction is :')
        result_index = model_prediction(test_image)
        #Reading Labels
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        st.success("Model is Predicting , it's a {}".format(label[result_index]))

        