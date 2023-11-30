import streamlit as st
from PIL import Image
import os
import cloudpickle
import matplotlib.pyplot as plt     
import numpy as np
import cv2
# import keras

image_size = 150


with open('./models/lung_model.pkl', 'rb') as file:
    loaded_model = cloudpickle.load(file)

with open('./models/lungclassification_model.pkl', 'rb') as file:
    classification_loaded_model = cloudpickle.load(file)


def cnn_predict_show_inner_workings(image, layers):
    out_p = (image/255) - 0.5
    outputs = []
    for i in layers:
      out_p = i.forward_prop(out_p)
      outputs.append(out_p)

    plots = []
    layer = 1
    for output in outputs[:-1:]:
      fig, axes = plt.subplots(1, output.shape[-1], figsize=(2*output.shape[-1],2))
      axes[0].set_ylabel(f"Layer {layer} output", rotation=90, size='large')
      for i in range(output.shape[-1]):
          ax = axes[i]
          ax.imshow(output[:,:,i], cmap='gray')
      layer += 1
    plt.tight_layout()
    plt.show()
    # print("Softmax layer output: ", outputs[-1])
    return np.argmax(outputs[-1])

    

def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        file_path = os.path.join("./images", uploaded_file.name)
        file_path = file_path.replace("\\", "/")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return file_path
    return None


def app():
    st.title("Lung Cancer Detection")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        file_path = save_uploaded_file(uploaded_file)

        if file_path is not None:
            st.success(f"Image saved successfully to {file_path}")
            img = cv2.imread(file_path)
            img_array = cv2.resize(img,(150,150))
            img_array = np.array(img_array) 
            img_array = img_array.reshape(1,150,150,3)
            
            grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_image = clahe.apply(grayscale_image)

            img = cv2.resize(clahe_image, (image_size,image_size))
            

    
            if st.button("Detect"):
                prediction = cnn_predict_show_inner_workings(img, loaded_model)
                if (prediction == 1):
                    st.warning("The person has Lung Cancer!!")
                    classification = classification_loaded_model.predict(img_array)
                    indicies = classification.argmax()
                    print(indicies)
                    if (indicies == 0):
                        st.warning("The type of Lung Cancer is Adenocarcinoma")
                    elif (indicies == 1):
                        st.warning("The type of Lung Cancer is Large Cell Carcinoma")
                    else :
                        st.warning("The type of Lung Cancer is Squamous Cell Carcinoma")
                else:
                    st.success("The person is safe from Lung Cancer")
        