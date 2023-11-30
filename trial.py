import joblib
import numpy as np
loaded_model = joblib.load("models/prostate.joblib")
rad = 9
tex = 20
per = 52
are = 202
smo = 0.07
com = .04
sym = .135
fad = .05
features = [rad,tex,per,are,smo,com,sym,fad]
def predict(Model, features):
    # Get model and model score
    predicted = Model.predict(np.array(features).reshape(1, -1))
    # if (prediction == 1):
    #         st.sidebar.info("Major parameters affecting Prostate Cancer are its Compactness and Area / Perimeter")
    #         st.warning("The person has Prostate Cancer!!")
    #         if (com > 0.09):
    #             st.write("Cause: Compactness Tumour is High",com)
    #         elif(are > 550):
    #             st.write("Cause: Area of Tumour is High",are)
    #         elif(per > 88):
    #             st.write("Cause: Perimeter of Tumour is High",per)
    #     else:
    #         st.success("The person is safe from Prostate Cancer")
    return predicted

prediction = predict(loaded_model,features)


print(prediction)