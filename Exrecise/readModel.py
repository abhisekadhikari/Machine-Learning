import numbers
import pickle
from statistics import mode
import numpy as np
# from sklearn.linear_model import LinearRegression
import streamlit as st

st.title('auto car insurance')
number = st.number_input("Enter Amount")
model = pickle.load(open('model.pkl', 'rb'))
result = model.predict([[number]])
if st.button("Check"):
    st.title(f"The Amount of your insurance is: {result}$")