import joblib
import pandas as pd
import streamlit as st


models = joblib.load('kyphosis_models.pkl')


st.title("Kyphosis Prediction App")

# Input sections
st.header("User Details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age")
    Number = st.number_input("Number")
    Start = st.number_input("Start")
    

# Model Selection
st.header("Model Selection")
selected_model = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"])
if st.button("Predict"):
    user_data = pd.DataFrame({"Age": [age], "Number": [Number], "Start": [Start] 
                              })
    if selected_model == "Logistic Regression":
        prediction = models['logistic_regression'].predict(user_data)
    elif selected_model == "Decision Tree":
        prediction = models['decision_tree'].predict(user_data)
    elif selected_model == "Random Forest":
        prediction = models['random_forest'].predict(user_data)
    else:
            prediction = models['XGBoost'].predict(user_data)
    st.write(f"{selected_model} Kyphosis Prediction:", prediction)
    
st.markdown("---")
st.markdown("<center>Made by RedCherry</center>", unsafe_allow_html=True)