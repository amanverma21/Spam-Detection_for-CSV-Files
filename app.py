import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def predict(message):
    model = pickle.load(open('model.pkl', 'rb'))
    cv = pickle.load(open('count.pkl', 'rb'))
    X_cV = cv.transform([message]).toarray()
    return "".join(model.predict(X_cV))


st.title('SMS Spam Predictor')
st.header('Using multiple files')
uploaded_file = st.file_uploader('Upload a CSV file', type='csv')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    df['prediction'] = df['v2'].apply(predict)
    csv = df.to_csv(index=False)
    st.download_button(label='Result_CSV', data=csv, file_name='predictions.csv', mime='text/csv')
st.text("")
st.text("")
st.header('Using Single Input')
x=st.text_input('Enter Message: ')

if st.button('Predict'):
    y=predict(x)
    st.title('message is : {}'.format(y))