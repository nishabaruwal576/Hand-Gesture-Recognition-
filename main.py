# Modules
import streamlit as st
from pyrebase import pyrebase
from PIL import Image , ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os
import h5py

# Configuration Key Database ma store garna

firebaseConfig = {
  'apiKey': "AIzaSyAQzQcKWeJsmRjCD5LlfzDenYb4WexS0YA",
  'authDomain': "asl-recognition-system.firebaseapp.com",
  'databaseURL': "https://asl-recognition-system-default-rtdb.europe-west1.firebasedatabase.app",
  'projectId': "asl-recognition-system",
  'storageBucket': "asl-recognition-system.appspot.com",
  'messagingSenderId': "476723922042",
  'appId': "1:476723922042:web:fe4033bb285da6e812a7c3",
  'measurementId': "G-FQDPG13LKZ"
}

# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Database
db = firebase.database()
storage = firebase.storage()
st.title('ASL Recognition System')
st.sidebar.title('User Authentication')
st.image('welcome.png')

# Authentication

choice = st.sidebar.selectbox('Login/Signup',['Login','Sign Up'])

# User ko Input 

email = st.sidebar.text_input('Please enter your email address')
password = st.sidebar.text_input('Please enter your password',type = 'password')

# Sign Up Block

if choice == 'Sign Up':
	handle = st.sidebar.text_input('Please enter your Username')
	submit = st.sidebar.button('Create My Account')

	if submit:
		user = auth.create_user_with_email_and_password(email,password)
		st.success('Your account has been created successfully')

		# Sign in
		user = auth.sign_in_with_email_and_password(email,password)
		db.child(user['localId']).child("Name").set(handle) 
		db.child(user['localId']).child("ID").set(user['localId']) 
		st.title('Welcome ' +  handle)
		st.info('Login via dropdown Login Option')

#Login Block
if choice == 'Login':
	login = st.sidebar.checkbox('Login')
	if login:
		user = auth.sign_in_with_email_and_password(email,password)
		st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
		bio = st.radio('Pages',['Home','Detection'])
#Home Page
		if bio == 'Home':
			st.header('ASL Recognition System')
			st.image("asl.webp")
			st.write(''' This project named "Sign Language Recognition system using CNN" is a part of the assessment submitted by Nisha Baruwal Chhetri (041902900042) to Sunway International Business School
			for the completion of her ongoing Bachelors degree of Computer Science. 
			''')
		# Detection Page
		elif bio == 'Detection':
			st.title('Detect Image')
			def main():
				file_uploaded = st.file_uploader("Choose the file", type=['jpg','png','jpeg'])
				if file_uploaded is not None:
					image = Image.open(file_uploaded)
					figure = plt.figure()
					plt.imshow(image)
					plt.axis('off')
					result = predict_class(image)
					st.write(result)
					st.pyplot(figure)
			def predict_class(image):
				classifier_model = tf.keras.models.load_model(r'/Users/nishabaruwal/Documents/Nisha Folder Final/saved_model/model.hdf5')
				shape = ((224,224,3))
				model = tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape=shape)])
				test_image = image.resize((224,224))
				test_image = preprocessing.image.img_to_array(test_image)
				test_image = test_image/255.0
				test_image = np.expand_dims(test_image, axis = 0)
				class_names = ['0',
				               '1',
                               '2',
                               '3',
                               '4',
                               '5',
                               '6',
                               '7',
                               '8',
                               '9',
                               'a',
                               'b',
                               'c',
                               'd',
                               'e',
                               'f',
                               'g',
                               'h',
                               'i',
                               'j',
                               'k',
                               'l',
                               'm',
                               'n',
                               'o',
                               'p',
                               'q',
                               'r',
                               's',
                               't',
                               'u',
                               'v',
                               'w',
                               'x',
                               'y',
                               'z'
				]
				predictions = model.predict(test_image)
				scores = tf.nn.softmax(predictions[0])
				scores = scores.numpy()
				image_class = class_names[np.argmax(scores)]
				result = "The image uploaded is: {}".format(image_class)
				return result
			if __name__ =="__main__":
				main()
    
    

		
		
		

		
	


								
								