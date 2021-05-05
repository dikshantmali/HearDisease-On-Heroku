
# -*- coding: utf-8 -*-
"""

@author: Dikshant Mali
"""

import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('practice_decision_model.pkl', 'rb')) 
# Feature Scaling
dataset = pd.read_csv('CLASSIFICATION DATASET.csv')
# Extracting independent variable:
X = dataset.iloc[:, 0 : 14]
X = X.fillna(method = 'ffill')

# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X.Gender = labelencoder_X.fit_transform(X.Gender)
X.Geography = labelencoder_X.fit_transform(X.Geography)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


def predict_note_authentication(age,cp,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal):
  output= model.predict(sc.transform([[age,cp,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal]]))
  print("The HeartDisease", output)
  if output==[0]:
    prediction="The person has Type 0 HeartDisease"
  elif output==[1]:
    prediction="The person has Type 1 HeartDisease"
  elif output==[2]:
    prediction="The person has Type 2 HeartDisease"
  elif output==[3]:
    prediction="The person has Type 3 HeartDisease"
  else:
    prediction="The person has Type 4 HeartDisease"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:#fa8072" >
   <div class="clearfix">           
   <div class="col-lg-12">
   <center><p style="font-size:40px;color:black;margin-top:10px;">The First Heroku Deployment Testing Project</p></center> 
   <center><p style="font-size:30px;color:black;margin-top:10px;">Department of Computer Engineering PIET,Jaipur</p></center> 
   <center><p style="font-size:25px;color:black;margin-top:10px;"Machine Learning Lab Experiment For Practice Of Heroku</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Heart Disease Prediction using Entropy Based Decision Tree Classification")
    age = st.number_input("Enter The age From 1 To 60",10,60)
    cp = st.number_input("Enter The cp value From 1 To 4",1,4)
    trestbps = st.number_input("Enter The trestbps From 100 To 200",100,200)
    chol = st.number_input("Enter The chol From 100 To 350",100,350)
    fbs = st.number_input("Enter The fbs either 0 or 1",0,2)
    Gender = st.number_input('Insert Gender Value Male:1 Female:0')
    Geography = st.number_input('Insert Geography Location Value France:0 Spain:1')
    restecg = st.number_input("Enter The For RestECG From 1 to 180",0,180)
    thalach = st.number_input("Enter The Value For thalach from 1 To 180",0,180)
    exang = st.number_input("Enter The exang Form 0 to 5",0,5)
    oldpeak = st.number_input("Enter The oldpeak 0 to 5",0,5)
    slope = st.number_input("Enter The slope 0 to 5",0,5)
    ca = st.number_input("Enter The ca value 0 to 9",0,9)
    thal = st.number_input("Enter The thal value 0 to 10",1,10)
    if st.button("Predict"):
      result=predict_note_authentication(age,cp,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal)
      st.success('{} '.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Dikshant Mali")
      st.subheader("Student , Poornima Institute Of Engineering And Technology")

if __name__=='__main__':
  main()