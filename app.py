import fastbook
from fastbook import *
from fastai.vision.widgets import *
import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model.pkl"
model = load_learner(filename)
# otwieramy wcześniej wytrenowany model


def main():
    st.set_page_config(page_title="Jaka to żaba?")
    overview = st.container()
    st.image("https://cdn.mos.cms.futurecdn.net/39CUYMP8vJqHAYGVzUghBX-970-80.jpg")
    params = st.container()
    prediction = st.container()

    with overview:
    	st.title("Co to za gatunek żaby?")

    with params:
        frog_uploader = st.file_uploader('Zdjęcie żaby')

    data = [[frog_uploader]]
    frog = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
    	st.subheader("Jaki to gatunek żaby?")
    	st.subheader((frog))
    	st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][frog][0] * 100))

if __name__ == "__main__":
    main()
