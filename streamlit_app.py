import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

from ucimlrepo import fetch_ucirepo 

# fetch dataset 

@st.cache_data
def load_data():
    df = fetch_ucirepo(id=186) 
    data = pd.concat([df.data.features, df.data.targets], axis=1)
    return data

data = load_data()


model = joblib.load("wine_model.joblib")

st.set_page_config(layout='wide')
st.title('Wine Quality Prediction Dashboard',
         help="This app predicts the quality of white wine on a scale from 0 to 10 based on its chemical properties.")

# Initialize session state for saved predictions
if 'save_button' not in st.session_state:
    st.session_state['save_button'] = pd.DataFrame()
if 'input_data' not in st.session_state:
    st.session_state['input_data'] = None
if 'predicted' not in st.session_state:
    st.session_state['predicted'] = None


with st.sidebar:
    st.header("**Wine Features**")

    fixed_acidity = st.slider("Fixed Acidity (g/dm³)", min_value=3.50, max_value=14.50, step=0.01)
    volatile_acidity = st.slider("Volatile Acidity (g/dm³)", min_value=0.05, max_value=1.20, step=0.01)
    citric_acid = st.slider("Citric Acid (g/dm³)", min_value=0.00, max_value=1.70, step=0.01)
    residual_sugar = st.slider("Residual Sugar (g/dm³)", min_value=0.5, max_value=66.0, step=0.1)
    chlorides = st.slider("Chlorides (g/dm³)", min_value=0.005, max_value=0.400, step=0.001)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide (mg/dm³)", min_value=1.0, max_value=300.0, step=1.0)
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide (mg/dm³)", min_value=5.0, max_value=450.0, step=1.0)
    density = st.slider("Density (g/cm³)", min_value=0.95, max_value=1.05, step=0.0001, format="%.4f")
    ph = st.slider("pH", min_value=2.5, max_value=4.0, step=0.01)
    sulphates = st.slider("Sulphates (g/dm³)", min_value=0.1, max_value=1.2, step=0.01)
    alcohol = st.slider("Alcohol (% vol)", min_value=7.0, max_value=15.0, step=0.1)

    col1, col2 = st.columns([1,2])
    button = col1.button("Calculate", type="primary", width='stretch')
    if button:
        st.success('Prediction done!')

    
input_data = {'fixed_acidity': fixed_acidity,
            'volatile_acidity': volatile_acidity,
            'citric_acid': citric_acid,
            'residual_sugar': residual_sugar,
            'chlorides': chlorides,
            'free_sulfur_dioxide': free_sulfur_dioxide,
            'total_sulfur_dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': ph,
            'sulphates': sulphates,
            'alcohol': alcohol}

input_df = pd.DataFrame([input_data])


if button:
    prediction = model.predict(np.array(input_df))
    st.metric(label="The predicted quality of the wine is:", value=f"{prediction[0]:.2f}")

    st.session_state['input_data'] = input_df
    st.session_state['predicted'] = prediction[0]
else:
    st.metric(label="The predicted quality of the wine is:", value=None)
    st.info("Adjust the features in the sidebar and click 'Calculate' to see the predicted wine quality.")

if st.session_state['input_data'] is not None and st.session_state['predicted'] is not None:
    save_button = st.button("Save Results", type="secondary")
    if save_button:
        saved = st.session_state['input_data'].copy()
        saved['predicted_quality'] = st.session_state['predicted']
        st.session_state['save_button'] = pd.concat([st.session_state['save_button'], saved], ignore_index=True)
        st.session_state['input_data'] = None
        st.session_state['predicted'] = None

        st.rerun()



tab1, tab2 = st.tabs(["Dataset Overview", "Saved Predictions"])


with tab1:
    col1, col2 = st.columns(2, gap="medium")

    importance = model.feature_importances_
    columns = data.columns[:-1]

    feature_importances = pd.DataFrame({"feature": columns,
                                        "importance": importance}).sort_values(by="importance", ascending=True)
    
    col1.write("### Feature Importance")

    feature_chart = alt.Chart(feature_importances).mark_bar().encode(
        x=alt.X('importance:Q', title='Importance'),
        y=alt.Y('feature:N', sort='-x', title='Feature')).properties(height=350)
    col1.altair_chart(feature_chart)


    col2.write("### Quality Distribution")

    quality_counts = data['quality'].value_counts().reset_index()
 
    quality_disc_chart = alt.Chart(quality_counts).mark_bar().encode(
        x=alt.X('quality:O', axis=alt.Axis(labelAngle=0), title='Wine Quality'), 
        y=alt.Y('count:Q', title="Count")).properties(height=350)
    col2.altair_chart(quality_disc_chart)

    st.write("### Scatter plots of Features vs Quality")
    option = st.selectbox('Select a feature to plot against Quality:',
                          options=[col for col in data.columns if col != 'quality'])
    scatter = alt.Chart(data).mark_circle().encode(
        x=option,
        y=alt.Y('quality:Q', title='Quality'),
        color='quality')
    st.altair_chart(scatter)

    st.write("### Dataset Sample")
    st.dataframe(data.head(10))



with tab2:
    st.write("### Saved Predictions")
 
    if not st.session_state['save_button'].empty:
        st.dataframe(st.session_state['save_button'])
    else:
        st.info("No saved predictions yet. Use the 'Save Results' button after making a prediction to save your results here.")
