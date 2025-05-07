# === Import Libraries ===
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# === Load Dataset ===
df = pd.read_csv("global-hotels.csv")

# === Data Preprocessing ===
df['Number_Reviews'] = df['Number_Reviews'].replace({',': ''}, regex=True).astype(int)
df['Room_Type_original'] = df['Room_Type']
mask_misplaced = df['Price'].str.contains('[a-zA-Z]', na=False) & df['Room_Type'].isna()
df.loc[mask_misplaced, 'Room_Type'] = df.loc[mask_misplaced, 'Price']
df.loc[mask_misplaced, 'Price'] = np.nan
df['Price'] = df['Price'].str.replace(r'[^\d.]', '', regex=True)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.drop(columns=['Room_Type_original'])

# === Handle Missing Values ===
df_knn = df.copy()
label_encoders = {}
categorical_cols = ['Hotel_Name', 'Rating', 'Room_Type', 'City', 'Country']

for col in categorical_cols:
    le = LabelEncoder()
    df_knn[col] = le.fit_transform(df_knn[col].astype(str))
    label_encoders[col] = le

imputer = KNNImputer(n_neighbors=5)
df_knn_imputed = pd.DataFrame(imputer.fit_transform(df_knn), columns=df_knn.columns)

for col in categorical_cols:
    df_knn_imputed[col] = label_encoders[col].inverse_transform(df_knn_imputed[col].astype(int))

# === Feature Preparation for TF-IDF ===
df_knn_imputed['combined_features'] = (
    df_knn_imputed['Room_Type'].astype(str) + ' ' +
    df_knn_imputed['Rating'].astype(str)
)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_knn_imputed['combined_features'])

# Combine with scaled numeric features if needed
# (but not used in this simplified version)
final_features = tfidf_matrix

# === Recommendation Function (Location → Room Type & Rating) ===
def recommend_by_pref_location_first(df_knn_imputed, tfidf, final_features,
                                     location_city, room_type, rating, top_n=5):
    # Filter by selected city
    df_filtered = df_knn_imputed[df_knn_imputed['City'] == location_city]

    if df_filtered.empty:
        return pd.DataFrame(columns=['Hotel_Name', 'City', 'Room_Type', 'Rating', 'Score', 'Number_Reviews', 'Price', 'sim_score'])

    # Preference text only includes room type and rating
    pref_text = f"{room_type} {rating}"
    pref_tfidf = tfidf.transform([pref_text])

    # Only use tfidf_matrix (no numerical features)
    sims = cosine_similarity(pref_tfidf, final_features[df_filtered.index]).flatten()

    df_filtered = df_filtered.copy()
    df_filtered['sim_score'] = sims
    df_sorted = df_filtered.sort_values(by='sim_score', ascending=False)

    return df_sorted.head(top_n)[['Hotel_Name', 'City', 'Room_Type', 'Rating', 'Score', 'Number_Reviews', 'Price', 'sim_score']]

# === Streamlit UI ===
st.title('Hotel Recommendation System')
st.write('Find the best hotels based on your location, room type, and rating')

# Input Fields
location_city = st.selectbox('Select City', sorted(df_knn_imputed['City'].unique()))
room_type = st.selectbox('Select Room Type', sorted(df_knn_imputed['Room_Type'].unique()))
rating = st.selectbox('Select Hotel Rating', sorted(df_knn_imputed['Rating'].unique()))

# Button
if st.button('Get Recommendations'):
    recommendations = recommend_by_pref_location_first(
        df_knn_imputed=df_knn_imputed,
        tfidf=tfidf,
        final_features=final_features,
        location_city=location_city,
        room_type=room_type,
        rating=rating,
        top_n=5
    )

    if recommendations.empty:
        st.warning("No recommendations found for the selected criteria.")
    else:
        st.write("Here are the top recommended hotels:")
        st.dataframe(recommendations)
