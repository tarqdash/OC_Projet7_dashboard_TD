import streamlit as st
import streamlit.components.v1 as components
st.set_option('deprecation.showPyplotGlobalUse', False)

import requests
import shap

from PIL import Image
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly

import sys
import os
import datetime

import shap

#################################
#################################
#################################
# Configuration of the streamlit page
st.set_page_config(page_title='PRET A DEPENSER DASHBOARD',
                   page_icon='random',
                   layout='centered',
                   initial_sidebar_state='auto')
# Title 
st.markdown("<h1 style='text-align: center; color: #E2383F;'><strong>💹 PRET A DEPENSER DASHBOARD</u></strong></h1>", unsafe_allow_html=True)
# Subtitle
st.markdown("<h4 style='text-align: center'><i>“Tarek DACHRAOUI - Data Science OC_Projet 7.”</i></h4>", unsafe_allow_html=True)
st.markdown("***")


# Display the logo in the sidebar
path = "./images/logo.png"
image = Image.open(path)
st.sidebar.image(image, width=250)

# local API (à remplacer par l'adresse de l'application déployée)
API_URL = "http://127.0.0.1:8000/api/"
# API_URL = "https://oc-api-FastAPI-td.herokuapp.com/api/"
#################################################################################
# LIST OF API REQUEST FUNCTIONS
#################################################################################

# Get list of SK_IDS (cached)
@st.cache
def get_sk_id_list():
    # URL of the sk_id API
    SK_IDS_API_URL = API_URL + "sk_ids/"
    # Requesting the API and saving the response
    response = requests.get(SK_IDS_API_URL)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    # Getting the values of SK_IDS from the content
    SK_IDS = pd.Series(content['data']).values
    return SK_IDS

# Get scoring of one applicant customer (cached)
@st.cache
def get_customer_scoring(selected_sk_id):
    # URL of the scoring API
    SCORING_API_URL = API_URL + "scoring_customer/?SK_ID_CURR=" + str(selected_sk_id)
    # Requesting the API and save the response
    response = requests.get(SCORING_API_URL)
    # convert from JSON format to Python dict
    content = json.loads(response.content.decode('utf-8'))
    # getting the values from the response
    client_score = content['score']
    optimal_threshold = content['optimal_threshold']
    
    return client_score, optimal_threshold

# Get Client Info (cached)
@st.cache
def get_client_info(selected_sk_id):
    # URL of the scoring API
    CLI_İNFO_API_URL = API_URL + "client_info/?SK_ID_CURR=" + str(selected_sk_id)
    # save the response of API request
    response = requests.get(CLI_İNFO_API_URL)
    # convert from JSON format to Python dict
    content = json.loads(response.content)
    selected_client_info = content['selected_client_info']
    return selected_client_info

# Get shap plot params of the customer (cached)
@st.cache
def get_shap_plot_params(selected_sk_id):
    # URL of the scoring API
    SHAP_PLOTS_API_URL = API_URL + "shap_plot_params/?SK_ID_CURR=" + str(selected_sk_id)
    # save the response of API request
    response = requests.get(SHAP_PLOTS_API_URL)
    # convert from JSON format to Python dict
    content = json.loads(response.content)
    # convert data to pd.DataFrame or pd.Series
    expected_value_1 = content['expected_value_1']
    shap_values_1 = pd.DataFrame(content['shap_values_1']).values
    selected_sample = pd.DataFrame(content['selected_sample'])
    selected_sample_val = selected_sample.values
    feature_names = selected_sample.columns
    
    return expected_value_1, shap_values_1, selected_sample, feature_names

# Get the feature descriptions (cached)
@st.cache
def get_feat_desc():
    # URL of the scoring API
    FEAT_DESC_API_URL = API_URL + "feat_desc/"
    # save the response of API request
    response = requests.get(FEAT_DESC_API_URL)
    # convert from JSON format to Python dict
    content = json.loads(response.content)
    feat_desc = pd.DataFrame(content['feat_desc'])
    
    return feat_desc

# Get the feature value of the selected client (cached)
@st.cache
def get_feat_val(selected_sk_id, selected_feat):
    # URL of the scoring API
    FEAT_VAL_API_URL = API_URL + "feat_val/?SK_ID_CURR=" + str(selected_sk_id) + "&FEAT_NAME=" + selected_feat
    # save the response of API request
    response = requests.get(FEAT_VAL_API_URL)
    # convert from JSON format to Python dict
    content = json.loads(response.content)
    feat_val = content['feat_val'][0]
    
    return feat_val

# Get the the NN of the selected client samples (cached)
@st.cache
def get_NN_samples(selected_sk_id):
    # URL of the scoring API
    NN_SAMP_API_URL = API_URL + "NN_samples/?SK_ID_CURR=" + str(selected_sk_id)
    # save the response of API request
    response = requests.get(NN_SAMP_API_URL)
    # convert from JSON format to Python dict
    content = json.loads(response.content)
    NN_samples = pd.DataFrame(content['NN_samples'])
    
    return NN_samples


# ------------------------------------------------
# Select the customer's ID
# ------------------------------------------------

SK_IDS = get_sk_id_list()
selected_sk_id = st.sidebar.selectbox("Please select a client ID", SK_IDS, key=1)


# ##################################################
# SCORING
# ##################################################

st.header("  Default Risk Score  ")
st.write('---------------------------------------------------------------------------------------')

#  Get client_score & optimal_threshold
client_score , optimal_threshold = get_customer_scoring(selected_sk_id)

gauge_col, padding, cli_info_col = st.columns((100, 8, 30))

with gauge_col:
    ## Gauge ##
    # https://plotly.com/python/gauge-charts/
    range_val = optimal_threshold*100
    gauge = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Default Risk Score gauge", 'font': {'size': 24}},
        value = client_score*100,
        mode = "gauge+number",
        gauge = {'axis': {'range': [None, 100]},
                 'steps' : [
                     {'range': [0, range_val], 'color': "#008177"},
                     {'range': [range_val, range_val + 5], 'color': "#00BAB3"},
                     {'range': [range_val + 5, range_val + 10], 'color': "#D4E88B"},
                     {'range': [range_val + 10, range_val + 15], 'color': "#F4EA9D"},
                     {'range': [range_val + 15, range_val + 20], 'color': "#FF9966"},
                     {'range': [range_val + 20, 100], 'color': "#E2383F"},
                 ],
                 'threshold': {
                     'line': {'color': "black", 'width': 10},
                     'thickness': 0.8,
                     'value': client_score*100},
                 'bar': {'color': "black", 'thickness' : 0.15}
                }
    ))

    gauge.update_layout(width=600, height=500,
                        margin=dict(l=50, r=50, b=100, t=100, pad=4))

    st.plotly_chart(gauge)

    if 0 < client_score <= optimal_threshold:
        trust_text = "EXCELLENT"
    elif optimal_threshold < client_score <= optimal_threshold + 0.05:
        trust_text = "GOOD"
    elif optimal_threshold + 0.05 < client_score <= optimal_threshold + 0.1:
        trust_text = "AVERAGE"
    elif optimal_threshold + 0.1 < client_score <= optimal_threshold + 0.15:
        trust_text = "LOW"
    else :
        trust_text = "WEAK"

    st.write('TRUST score for the selected client : **{}**'.format(trust_text))

with cli_info_col:
    # Get client info
    selected_client_info = get_client_info(selected_sk_id)

    # Préparation informations client sélectionné
    client_id = selected_client_info["SK_ID_CURR"]
    client_age = selected_client_info["YEARS_BIRTH"]
    client_employed = selected_client_info["YEARS_EMPLOYED"]
    client_work = selected_client_info["NAME_INCOME_TYPE"]
    client_income = selected_client_info["AMT_INCOME_TOTAL"]
    client_contract = selected_client_info["NAME_CONTRACT_TYPE"]
    client_status = selected_client_info["NAME_FAMILY_STATUS"]
    client_gender = selected_client_info["GENDER"]
    client_education = selected_client_info["NAME_EDUCATION_TYPE"]


    # Affichage d'informations sur le client sélectionné dans la sidebar
    st.header("📋 Client informations")
    st.write("**Client ID**", client_id)
    st.write("**Age**", int(client_age), "years")
    st.write("**Gender** :", client_gender)
    st.write("**Family status** :", client_status)
    st.write("**Education** :", client_education)
    st.write("**Years employed**", int(client_employed), "years")
    st.write("**Income type** :", client_work)
    st.write("**Income**", int(client_income), "$")
    st.write("**Contract type** :", client_contract)

st.write('---------------------------------------------------------------------------------------')
st.sidebar.subheader('📈 SHAP explainer')
# ------------------------------------------------
# Explain Prediction score
# ------------------------------------------------
if st.sidebar.checkbox('SHAP Prediction Explainer', key=3):

    st.header("  SHAP Force & Decision Plots  ")
    st.write('---------------------------------------------------------------------------------------')
    
    shap_plot_col, feat_desc_col = st.columns((5, 2))
    
    with shap_plot_col:
        # Get shap plot params
        expected_value_1, shap_values_1, selected_sample, feature_names = get_shap_plot_params(selected_sk_id)

        ind_fig = shap.force_plot(expected_value_1,
                                  shap_values_1,
                                  selected_sample,
                                  feature_names=feature_names,
                                  link='logit',
                                  plot_cmap=["#EF553B","#636EFA"])
        ind_fig_html = f"<head>{shap.getjs()}</head><body>{ind_fig.html()}</body>"
        components.html(ind_fig_html, height=120)
        
        fig = plt.figure(figsize=(14, 7))
        shap.decision_plot(expected_value_1, shap_values_1, feature_names=list(feature_names), link='logit')
        st.pyplot(fig)

    with feat_desc_col:
        #Feature descriptions
        feat_desc = get_feat_desc()
        selected_feat = st.selectbox("Please select a feature description", list(feature_names), key=10)
        
        #Get feat value for the selected client
        feat_val = get_feat_val(selected_sk_id, selected_feat)
        description = feat_desc[(feat_desc['Table']=='application_{train|test}.csv') & \
                             (feat_desc['Row']==selected_feat)]['Description']
        
        #write
        st.write('Feature Description : **{}**'.format(description.values))
        st.write('Feature value :',feat_val)


st.write('---------------------------------------------------------------------------------------')        

# Comparison with Similar clients
st.sidebar.header("📊 Comparison with Similar clients")
# ##################################################


if st.sidebar.checkbox("Stats for nearest neighbors", key=4):
    st.write("IN PROGRESS ...")
    
    #  Get nearest neighbors (50)
    NN_samples = get_NN_samples(selected_sk_id)
    st.write(NN_samples)
    
      
    # Age
    if st.sidebar.button("Age"):
        # Graph dans app principale
        if st.sidebar.checkbox("Show age infos / Hide", value = True):
            st.write("IN PROGRESS ...Age")
            boxplot_col, write_col = st.columns((10, 2))
            with boxplot_col:
                fig = plt.figure(figsize=(5, 4))
                sns.boxplot( y=NN_samples["YEARS_BIRTH"]);
                st.pyplot(fig)
            with write_col:
                st.write("**Client Age** :", int(client_age), "years")
    
    # YEARS EMPLOYED
    if st.sidebar.button("YEARS EMPLOYED"):
        # Graph dans app principale
        if st.sidebar.checkbox("Show years employed infos / Hide", value = True): 
            st.write("IN PROGRESS ...YEARS EMPLOYED")
            boxplot_col, write_col = st.columns((10, 2))
            with boxplot_col:
                # affichage des boxplots
                fig = plt.figure(figsize=(10, 4))
                sns.boxplot( y=NN_samples["YEARS_EMPLOYED"]);
                st.pyplot(fig)
            with write_col:
                st.write("**Client YEARS EMPLOYED**", int(client_employed), "years")
            
    # Work
    if st.sidebar.button("Work"):
        st.sidebar.write("**Work** :", client_work)
        # Graph dans app principale
        if st.sidebar.checkbox("Show Work infos / Hide", value = True): 
            st.write("IN PROGRESS ...Work")
            pie_col, write_col = st.columns((10, 2))
            with pie_col:
                #plot the pie graph
                fig = plt.figure(figsize=(14,10))
                NN_samples["NAME_INCOME_TYPE"].value_counts(normalize=True).plot.pie()
                st.pyplot(fig)
            with write_col:
                st.write("**Client Work**", client_work)
                
            

    if st.sidebar.button("Income"):
        # Graph dans app principale
        if st.sidebar.checkbox("Show income infos / Hide", value = True): 
            st.write("IN PROGRESS ...Income")
            boxplot_col, write_col = st.columns((10, 2))
            with boxplot_col:
                fig = plt.figure(figsize=(10, 4))
                sns.boxplot( y=NN_samples["AMT_INCOME_TOTAL"]);
                st.pyplot(fig)
            with write_col:
                st.write("**Income**", int(client_income), "$")
            
    
    if st.sidebar.button("Contract Type"):
        # Graph dans app principale
        if st.sidebar.checkbox("Show contract type infos / Hide", value = True):
            st.write("IN PROGRESS ...Contract Type")
            pie_col, write_col = st.columns((10, 2))
            with pie_col:
                #plot the pie graph
                fig = plt.figure(figsize=(14,10))
                NN_samples["NAME_CONTRACT_TYPE"].value_counts(normalize=True).plot.pie()
                st.pyplot(fig)
            with write_col:
                st.write("**Contract Type** :", client_contract)
            

    if st.sidebar.button("Family Status"):
        # Graph dans app principale
        if st.sidebar.checkbox("Show Family Status infos / Hide", value = True): 
            st.write("IN PROGRESS ...Family Status")
            pie_col, write_col = st.columns((10, 2))
            with pie_col:
                #plot the pie graph
                fig = plt.figure(figsize=(14,10))
                NN_samples["NAME_FAMILY_STATUS"].value_counts(normalize=True).plot.pie()
                st.pyplot(fig)
            with write_col:
                st.write("**Family Status**", client_status)
            
    
    if st.sidebar.button("Gender"):
        # Graph dans app principale
        if st.sidebar.checkbox("Show Gender infos / Hide", value = True): 
            st.write("IN PROGRESS ...Gender")
            pie_col, write_col = st.columns((10, 2))
            with pie_col:
                #plot the pie graph
                fig = plt.figure(figsize=(14,10))
                NN_samples["GENDER"].value_counts(normalize=True).plot.pie()
                st.pyplot(fig)
            with write_col:
                st.write("**Gender**", client_gender)
        
            
    
    if st.sidebar.button("Education Type"):
        # Graph dans app principale
        if st.sidebar.checkbox("Show Education infos / Hide", value = True):
            st.write("IN PROGRESS ...Education Type")
            pie_col, write_col = st.columns((10, 2))
            with pie_col:
                #plot the pie graph
                fig = plt.figure(figsize=(14,10))
                NN_samples["NAME_EDUCATION_TYPE"].value_counts(normalize=True).plot.pie()
                st.pyplot(fig)
            with write_col:
                st.write("**Education**", client_education)
    
    
st.write('---------------------------------------------------------------------------------------')
    
    
