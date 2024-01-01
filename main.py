import pickle
import numpy as np
import streamlit as st
from PIL import Image

from util import classify

import pandas as pd
import jcopdl
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from jcopdl.callback import Callback, set_config

# Set title
st.title('Klasifikasi Tumpahan Minyak')

# Set header
st.header('Unggah Gambar')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model = load_model('./copymodel.h5')

# Load class names
with open('./model/labels-ril.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

# Display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Classify image
    class_name, conf_score = classify(image, model, class_names)

    # Write classification
    st.write("## {}".format(class_name))
    st.write("### Prediksi: {}%".format(int(conf_score * 1000) / 10))
