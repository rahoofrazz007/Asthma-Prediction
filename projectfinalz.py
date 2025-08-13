import streamlit as st

import joblib

gb_model=joblib.load("model.pkl")
l1=joblib.load("encoder1.pkl")
l2=joblib.load("encoder2.pkl")
l3=joblib.load("encoder3.pkl")
l4=joblib.load("encoder4.pkl")
l5=joblib.load("encoder5.pkl")
s=joblib.load("scaler.pkl")

st.title("synthetic_asthma_dataset2")
st.write("Enter Data Description")

Age=st.number_input("Enter Your Age")
Gender=st.selectbox('Enter Your Gender',['female','male'])
BMI=st.number_input('Enter BMI')
Smoking_Status=st.selectbox('Enter Smoking_Status',['Never','Former'])
Family_History=st.number_input('Ener Family_History')
Air_Pollution_Level=st.selectbox('Enter Air_Pollution_Level',['Low','Moderate','High'])
Physical_Activity_Level=st.selectbox("Enter Physical_Activity_Level",['Sedentary','Moderate','Active'])
Occupation_Type=st.selectbox("Enter Occupation_Type",['Indoor','Outdoor'])
Medication_Adherence=st.number_input("Enter Medication_Adherence")
Number_of_ER_Visits=st.number_input("Enter Number_of_ER_Visits")
Peak_Expiratory_Flow=st.number_input("Enter Peak_Expiratory_Flow")
FeNO_Level=st.number_input("Enter FeNO_Level")



Gender=l1.fit_transform(data[Gender])[0]
Smoking_Status=l2.fit_transform(data[Smoking_Status])[0]
Air_Pollution_Level=l3.fit_transform(data[Air_Pollution_Level])[0]
Physical_Activity_Level=l4.fit_transform(data[Physical_Activity_Level])[0]
Occupation_Type=l5.fit_transform(data[Occupation_Type])[0]
if st.button("predict"):
    result=model.predict(s.transform([[Age,Gender,BMI,Smoking_Status,Family_History,Air_Pollution_Level,Physical_Activity_Level,Occupation_Type,
                                       Medication_Adherence,Number_of_ER_Visits,Peak_Expiratory_Flow,FeNO_Level]]))[0]
    st.success('the output is {}'.format(result))
    

    

