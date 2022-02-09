import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

scaler = pickle.load(open('scaleer.pk1','rb'))
svc = pickle.load(open('svc (1).sav','rb'))

def generate_prediction(concave_points_worst, concave_points_mean, radius_worst, concavity_mean, concavity_worst,
                        texture_worst, radius_se, symmetry_worst, compactness_mean, perimeter_worst, area_se):
    
    array = np.array([concave_points_worst, concave_points_mean, radius_worst, concavity_mean, concavity_worst,
                        texture_worst, radius_se, symmetry_worst, compactness_mean,perimeter_worst,area_se])
    scaled_array = scaler.transform(array.reshape(1,-1))

    pred = svc.predict(scaled_array)
    prob = svc.predict_proba(scaled_array)

    return pred, prob

def main():
    st.title('Classificação de câncer de mama')
    st.subheader('https://www.kaggle.com/uciml/breast-cancer-wisconsin-data')

    concave_points_worst = st.number_input('Concave Points Worst')
    concave_points_mean = st.number_input('Concave Points Mean')
    radius_worst = st.number_input('Radius Worst')
    concavity_mean = st.number_input('Concavity Mean')
    concavity_worst = st.number_input('Concavity Worst')
    texture_worst = st.number_input('Texture Worst')
    radius_se = st.number_input('Radius Se')
    symmetry_worst = st.number_input('Simmetry Worst')
    compactness_mean = st.number_input('Compacteness Mean')
    perimeter_worst = st.number_input('Perimeter Worst')
    area_se = st.number_input('Area Se')



    st.sidebar.image('logo_dados-e-saude.png', width=130)
    st.sidebar.subheader('by: Emilly :sunglasses:')

    botao = st.button('Gerar Predição')

    if botao:
        pred, prob = generate_prediction(concave_points_worst, concave_points_mean, radius_worst, concavity_mean, concavity_worst,
                        texture_worst, radius_se, symmetry_worst, compactness_mean, perimeter_worst,area_se)
        if pred == 1:
            st.write('Maligno')
        else:
            st.write('Benignoexit')

        st.write(f'Prababilidade: {round(prob[0][1],2)}')

if __name__ == '__main__':
    main()