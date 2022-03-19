# Core Pkgs
import streamlit as st
import altair as alt

# EDA Pkgs
import pandas as pd
import numpy as np
from datetime import datetime

# Utils
import joblib
pipe_lr = joblib.load(
    open("App/models/Language_Detection_pipe_lr.pkl", "rb"))


# Fxn
def predict_lang(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


# Main Application
def main():
    st.title("Language Detection App - ")
    menu = ["Home"]

    with st.form(key='lang_detection_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        # Apply Fxn Here
        prediction = predict_lang(raw_text)
        probability = get_prediction_proba(raw_text)

        # add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            st.write("- "+prediction)
            st.write("Confidence : {}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            # st.write(probability)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            # st.write(proba_df.T)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["Language", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='Language', y='probability', color='Language')
            st.altair_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
