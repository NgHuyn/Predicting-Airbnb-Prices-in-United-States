import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor
from pickle import load

metadata = {}
with open('metadata.txt', 'r') as fi:
    textdata = fi.read()
    textdata = textdata.split('\n')
    for item in textdata:
        if ('  ' in item):
            subitem = item.split('  ')
        else:
            subitem = item.split(' ')
        metadata[subitem[0]] = subitem[1:]

amenities_list = []
amenities_dict = {}
with open("amenities.txt", 'r') as fi:
    textdata = fi.read()
    textdata = textdata.split('\n')
    for item in textdata:
        subitem = item.split('@')
        if (len(subitem) == 2):
            amenities_list.append(subitem[1])
            amenities_dict[subitem[1]] = subitem[0]

scaler = load(open('scaler.pkl', 'rb'))
# print(amenities_list)

st.title("Airbnb price predictor")
with st.form('Fill in this form to get our prediction!'):
    st.text("Fill in this form to get our prediction!")
    st.markdown("**Host informations**")
    city = st.selectbox("Choose your city:", metadata['city'])
    col = st.columns([1, 1, 1, 1])
    with col[0]:
        host_response_time = st.selectbox("Response time", metadata['host_response_time'])
    with col[1]:
        host_response_rate = st.slider("Response rate", 0.0, 1.0, step=0.01)
    with col[2]:
        host_acceptance_rate = st.slider("Acceptance rate", 0.0, 1.0, step=0.01)
    st.markdown("""
    <style>
    [data-testid=column]:nth-of-type(4) [data-testid=stVerticalBlock]{
        gap: 0rem;
    }
    </style>
    """,unsafe_allow_html=True)
    with col[3]:
        host_is_superhost = st.checkbox("Super host")
        host_has_profile_pic = st.checkbox("Have profile pic")
        host_identity_verified = st.checkbox("Identity verified")
    col = st.columns([2, 3])
    with col[0]:
        host_total_listings_count = st.slider('Total listings count', 1.0, 15.0, step=1.0)
    with col[1]:
        host_verifications = st.multiselect("Host verifications", metadata['host_verifications'])
    
    st.markdown("**Accomodation informations**")
    col = st.columns([1, 1])
    with col[0]:
        latitude = st.number_input("Latitude", step=1e-13,format="%.13f")
    with col[1]:
        longitude = st.number_input("Longitude", step=1e-13,format="%.13f")
    col = st.columns([1, 1, 1, 1])
    with col[0]:
        room_type = st.selectbox("Room type", metadata['room_type'])
    with col[1]:
        accommodates = st.slider("Accommodates", 1.0, 16.0, step=1.0)
    with col[2]:
        bedrooms = st.slider("Bedrooms", 1.0, 16.0, step=1.0)
    with col[3]:
        beds = st.slider("Beds", 1.0, 16.0, step=1.0)

    amenities_ = st.multiselect("Amenities", amenities_list)
    col = st.columns([1, 1, 1])
    with col[0]:
        minimum_nights = st.slider("Minimum nights", 1, 100, step=1)
    with col[1]:
        maximum_nights = st.slider("Maximum nights", 1, 1000, step=1)
    with col[2]:
        has_availability = st.selectbox("Availability", [1, 0])
    
    col = st.columns([1, 1, 1, 1])
    with col[0]:
        availability_30 = st.slider("Availability in 30 days", 0, 30, step=1)
    with col[1]:
        availability_60 = st.slider("Availability in 60 days", 0, 60, step=1)
    with col[2]:
        availability_90 = st.slider("Availability in 90 days", 0, 90, step=1)
    with col[3]:
        availability_360 = st.slider("Availability in 360 days", 0, 360, step=1)

    st.markdown("**Reviews**")
    col = st.columns([1, 1, 1, 1])
    with col[0]:
        number_of_reviews = st.slider("Number of reviews", 1, 3000, step=1)
        review_scores_rating = st.slider("Review scores rating", 1.0, 5.0, step=0.01)
    with col[1]:
        review_scores_accuracy = st.slider("Review scores accuracy", 1.0, 5.0, step=0.01)
        review_scores_cleanliness = st.slider("Review scores cleanliness", 1.0, 5.0, step=0.01)
    with col[2]:
        review_scores_checkin = st.slider("Review scores checkin", 1.0, 5.0, step=0.01)
        review_scores_communication = st.slider("Review scores communcation", 1.0, 5.0, step=0.01)
    with col[3]:
        review_scores_location = st.slider("Review scores location", 1.0, 5.0, step=0.01)
        review_scores_value = st.slider("Review scores value", 1.0, 5.0, step=0.01)
    
    col = st.columns([1, 1])
    with col[0]:
        instant_bookable = st.selectbox("Instant bookable", [1, 0])
    with col[1]:
        reviews_per_month = st.slider("Review per month", 0.0, 100.0, step=0.01)

    submitted = st.form_submit_button("Predict")
    if submitted:
        one_hot_features = [host_response_time]
        for item in host_verifications:
            one_hot_features.append(item)
        for item in amenities_:
            one_hot_features.append(amenities_dict[item])
        if (host_is_superhost):
            host_is_superhost = 1.0
        else:
            host_is_superhost = 0.0
        if (host_has_profile_pic):
            host_has_profile_pic = 1.0
        else:
            host_has_profile_pic = 0.0
        if (host_identity_verified):
            host_identity_verified = 1.0
        else:
            host_identity_verified = 0.0
        df = {
            'host_response_rate' : [host_response_rate],
            'host_acceptance_rate' : [host_acceptance_rate],
            'host_is_superhost' : [host_is_superhost],
            'host_total_listings_count' : [host_total_listings_count],
            'host_has_profile_pic' : [host_has_profile_pic],
            'host_identity_verified' : [host_identity_verified],
            'latitude' : [latitude],
            'longitude' : [longitude],
            'accommodates' : [accommodates],
            'bedrooms' : [bedrooms],
            'beds' : [beds],
            'minimum_nights' : [minimum_nights],
            'maximum_nights' : [maximum_nights],
            'has_availability' : [has_availability],
            'availability_30' : [availability_30],
            'availability_60' : [availability_60],
            'availability_90' : [availability_90],
            'availability_365' : [availability_360],
            'number_of_reviews' : [number_of_reviews],
            'review_scores_rating' : [review_scores_rating],
            'review_scores_accuracy' : [review_scores_accuracy],
            'review_scores_cleanliness' : [review_scores_cleanliness],
            'review_scores_checkin' : [review_scores_checkin],
            'review_scores_communication' : [review_scores_communication],
            'review_scores_location' : [review_scores_location],
            'review_scores_value' : [review_scores_value],
            'instant_bookable' : [instant_bookable],
            'reviews_per_month' : [reviews_per_month],
        }
        with open('feature_list.txt') as fi:
            features = fi.read()
            features = features.split('\n')
            for feature in features:
                if feature in one_hot_features:
                    df[feature] = [1]
                else:
                    df[feature] = [0]
        df = pd.DataFrame(df)
        df = scaler.transform(df)

        model = CatBoostRegressor()
        model.load_model("model.cbm")
        price = model.predict(df)
        st.text(f"Predicted price: {price[0]} USD")
