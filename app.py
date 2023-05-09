import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import sklearn

# 页面内容设置
# 页面名称
st.set_page_config(page_title="Intubation", layout="wide")
# 标题
st.title('A simplified invasive mechanical ventilation prediction model for ICU patients')

st.markdown('_This is an online tool to predict  the need of invasive mechanical ventilation for the adult ICU patients.\
         Please adjust the value of each feature. After that, click on the Predict button at the bottom to see the prediction._')

st.markdown('## Input Data:')
# 隐藏底部水印
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            <![]()yle>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)



@st.cache
def predict_quality(model, df):
    y_pred = model.predict_proba(df)
    return y_pred[:, 1]


# 导入模型
model = joblib.load('cb_12fea.pkl')

st.sidebar.title("Features")

# 设置各项特征的输入范围和选项
age = st.sidebar.slider(label='age', min_value=18.00,
                                  max_value=120.00,
                                  value=70.00,
                                  step=0.01)


BMI = st.sidebar.slider(label='BMI', min_value=1.00,
                                max_value=100.00,
                                value=24.00,
                                step=0.01)

DBP = st.sidebar.slider(label='DBP', min_value=1,
                                   max_value=200,
                                   value=80,
                                   step=1)


GCS = st.sidebar.number_input(label='GCS Score', min_value=3,
                       max_value=15,
                       value=3,
                       step=1)



LOS = st.sidebar.number_input(label='LOS before ICU', min_value=0.00,
                            max_value=1000.00,
                            value=100.00,
                            step=0.01)

PaO2 = st.sidebar.number_input(label='PaO2', min_value=1.0,
                            max_value=1000.0,
                            value=1.0,
                            step=0.1)

SpO2 = st.sidebar.number_input(label='SpO2', min_value=0,
                            max_value=100,
                            value=50,
                            step=1)

RR = st.sidebar.number_input(label='RR', min_value=0,
                            max_value=100,
                            value=0,
                            step=1)

glucose = st.sidebar.number_input(label='glucose', min_value=1,
                            max_value=1000,
                            value=1,
                            step=1)

calcium = st.sidebar.number_input(label='calcium', min_value=0.0,
                            max_value=20.0,
                            value=0.0,
                            step=0.1)

PLT = st.sidebar.number_input(label='PLT', min_value=1,
                            max_value=1000,
                            value=50,
                            step=1)

RBC = st.sidebar.number_input(label='RBC', min_value=0.00,
                            max_value=10.00,
                            value=3.00,
                            step=0.01)

features = {'age': age,
            'BMI': BMI,
            'DBP': DBP,
            'GCS': GCS,
            'LOS before ICU': LOS,
            'PaO2': PaO2,
            'SpO2':SpO2,
            'RR': RR,
            'glucose':glucose,
            'calcium':calcium,
            'PLT': PLT,
            'RBC': RBC
}

features_df = pd.DataFrame([features])
#显示输入的特征
st.table(features_df)

#显示预测结果与shap解释图
if st.button('Predict'):
    prediction = predict_quality(model, features_df)
    st.write("the probability of mortality:")
    st.success(round(prediction[0], 3))


