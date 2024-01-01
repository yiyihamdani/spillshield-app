import pickle
import numpy as np
import streamlit as st

from until import classify

#set title 
st.title('Klasifikasi Tumpahan Minyak')

#set header
st.header('Unggah Gambar')

#upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

#load classifer
model - load_model('./model/copymodel.h5')

"load class names
with open('./model/labels-ril.txt', 'r') as f:
  class_names = [a[:-1].split('')[1] for a in f.readlines()]
  f.close()

#display image
if file is not None:
  image = Image.open(file).convert('RGB')
  st.image(image, use_column_width-True)

  #classify image
  class_name, conf_score - classify(image, model, class_name)

  #write classification
  st.write("## {}".format(class_name))
  st.write("### Prediksi: {}%". format(int(conf_score *1000) / 10))
