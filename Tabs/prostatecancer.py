"""This modules contains data about prediction page"""

# Import necessary modules
import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle

with open('./models/prostate_model.pkl', 'rb') as file: 
    loaded_model = cloudpickle.load(file)

def predict(Model, features):
    predicted = Model.predict(np.array(features).reshape(1, -1))
    # print(predicted)
    return predicted

def app(df):
    """This function create the prediction page"""

    # Add title to the page
    st.title("Prostate Cancer DeTection")

    st.subheader("Enter Values:")

    rad = st.number_input("Radius", value=df["radius"].min(), min_value=df["radius"].min(), max_value=df["radius"].max())
    tex = st.number_input("Texture", value=df["texture"].min(), min_value=df["texture"].min(), max_value=df["texture"].max())
    per = st.number_input("Perimeter", value=df["perimeter"].min(), min_value=df["perimeter"].min(), max_value=df["perimeter"].max())
    are = st.number_input("Area", value=df["area"].min(), min_value=df["area"].min(), max_value=df["area"].max())
    smo = st.number_input("Smoothness", value=df["smoothness"].min(), min_value=df["smoothness"].min(), max_value=df["smoothness"].max())
    com = st.number_input("Compactness", value=df["compactness"].min(), min_value=df["compactness"].min(), max_value=df["compactness"].max())
    sym = st.number_input("Symmetry", value=df["symmetry"].min(), min_value=df["symmetry"].min(), max_value=df["symmetry"].max())
    fad = st.number_input("Fractal Dimension", value=df["fractal_dimension"].min(), min_value=df["fractal_dimension"].min(), max_value=df["fractal_dimension"].max())

    features = [rad,tex,per,are,smo,com,sym,fad]

    st.header("The values entered by user")
    st.cache_data()
    df3 = pd.DataFrame(features).transpose()
    df3.columns=['radius','texture','perimeter','area','smoothness','compactness','symmetry','fractal_dimension']
    st.dataframe(df3)

    
    # Create a button to predict
    if st.button("Detect"):
        prediction = predict(loaded_model, features)
        

        # Print the output according to the prediction
        if (prediction == 1):
            st.sidebar.info("Major parameters affecting Prostate Cancer are its Compactness and Area / Perimeter")
            st.warning("The person has Prostate Cancer!!")
            if (com > 0.09):
                st.write("Cause: Compactness Tumour is High",com)
            elif(are > 550):
                st.write("Cause: Area of Tumour is High",are)
            elif(per > 88):
                st.write("Cause: Perimeter of Tumour is High",per)
        else:
            st.success("The person is safe from Prostate Cancer")