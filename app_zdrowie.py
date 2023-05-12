# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model_health.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model


def main():
    st.set_page_config(page_title="Przewidywanie Zdrowia")
    overview = st.container()
    st.image("https://images.newscientist.com/wp-content/uploads/2019/06/18153152/medicineshutterstock_1421041688.jpg?width=900")
    params = st.container()
    prediction = st.container()

    with overview:
    	st.title("Czy jesteś zdrowy?")

    with params:
    	age_slider = st.slider("Wiek [lata]", min_value=10, max_value=80, step=1)
    	height_slider = st.slider("Wzrost [cm]", min_value=159, max_value=200, step=1)
    	smpt_slider = st.slider("Liczba objawów", value=0, min_value=0, max_value=5, step=1)
    	comorb_slider = st.slider("Liczba chorób współistniejących", min_value=0, max_value=5, step=1)
    	meds_slider = st.slider("Liczba przyjmowanych leków", min_value=0, max_value=5, step=1)

    data = [[smpt_slider, age_slider, comorb_slider, height_slider, meds_slider]]
    health = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
    	st.subheader("Czy ta osoba jest zdrowa?")
    	st.subheader(("Tak" if health[0] == 0 else "Nie"))
    	st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][health][0] * 100))

if __name__ == "__main__":
    main()
