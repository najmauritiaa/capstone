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

# Load dataset
df = pd.read_csv("global-hotels.csv")

# Preprocessing
df['Number_Reviews'] = df['Number_Reviews'].replace({',': ''}, regex=True).astype(int)
df['Room_Type_original'] = df['Room_Type']
mask_misplaced = df['Price'].str.contains('[a-zA-Z]', na=False) & df['Room_Type'].isna()
df.loc[mask_misplaced, 'Room_Type'] = df.loc[mask_misplaced, 'Price']
df.loc[mask_misplaced, 'Price'] = np.nan
df['Price'] = df['Price'].str.replace(r'[^\d.]', '', regex=True)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.drop(columns=['Room_Type_original'])

# Handle missing values
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

# Normalize numeric features
scaler = MinMaxScaler()
numerical_features = df_knn_imputed[['Score', 'Number_Reviews', 'Price']]
numerical_scaled = scaler.fit_transform(numerical_features)

# Combine features into one column
df_knn_imputed['combined_features'] = (
    df_knn_imputed['Rating'].astype(str) + ' ' +
    df_knn_imputed['Score'].astype(str) + ' ' +
    df_knn_imputed['Number_Reviews'].astype(int).astype(str) + ' ' +
    df_knn_imputed['Room_Type'].astype(str) + ' ' +
    df_knn_imputed['City'].astype(str) + ' ' +
    df_knn_imputed['Country'].astype(str)
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_knn_imputed['combined_features'])

# Combine TF-IDF matrix with numeric features
final_features = hstack([tfidf_matrix, numerical_scaled])

# Recommendation Function
def recommend_by_pref(df_knn_imputed, tfidf, scaler, final_features,
                      city, room_type, rating, top_n=5):
    # Filter berdasarkan kota dan tipe kamar
    df_filtered = df_knn_imputed[
        (df_knn_imputed['City'] == city) &
        (df_knn_imputed['Room_Type'] == room_type)
    ]

    if df_filtered.empty:
        return pd.DataFrame(columns=[
            'Hotel_Name', 'City', 'Room_Type', 'Rating',
            'Score', 'Number_Reviews', 'Price', 'sim_score'
        ])

    # Bangun vektor preferensi pengguna dari rating
    pref_text = f"{rating}"
    pref_tfidf = tfidf.transform([pref_text])

    # Hitung rata-rata fitur numerik dari hotel hasil filter
    avg_score = df_filtered['Score'].mean()
    avg_reviews = df_filtered['Number_Reviews'].mean()
    avg_price = df_filtered['Price'].mean()
    pref_num = scaler.transform([[avg_score, avg_reviews, avg_price]])

    # Gabungkan fitur
    pref_vec = hstack([pref_tfidf, pref_num])

    # Konversi index label ke posisi integer agar bisa digunakan di final_features
    filtered_positions = df_filtered.index.to_numpy().tolist()
    filtered_positions = [df_knn_imputed.index.get_loc(idx) for idx in filtered_positions]

    # Hitung similarity
    sims = cosine_similarity(pref_vec, final_features[filtered_positions, :]).flatten()

    # Masukkan skor similarity ke dataframe
    df_filtered = df_filtered.copy()
    df_filtered['sim_score'] = sims

    # Urutkan berdasarkan similarity tertinggi
    return df_filtered.sort_values(by='sim_score', ascending=False).head(top_n)[
        ['Hotel_Name', 'City', 'Room_Type', 'Rating', 'Score', 'Number_Reviews', 'Price', 'sim_score']
    ]

# === Streamlit UI ===
st.title("Hotel Recommender ala Traveloka ✈️🏨")

city = st.selectbox('Pilih Kota', df_knn_imputed['City'].dropna().unique())
room_type = st.selectbox('Pilih Tipe Kamar', df_knn_imputed['Room_Type'].dropna().unique())
rating = st.selectbox('Pilih Rating Hotel yang Kamu Inginkan', ['Excellent', 'Very Good', 'Good', 'Average', 'Poor'])

if st.button('Tampilkan Rekomendasi'):
    recommendations = recommend_by_pref(
        df_knn_imputed=df_knn_imputed,
        tfidf=tfidf,
        scaler=scaler,
        final_features=final_features,
        city=city,
        room_type=room_type,
        rating=rating,
        top_n=5
    )

    if recommendations.empty:
        st.warning("Tidak ditemukan hotel yang cocok dengan preferensimu.")
    else:
        st.success("Berikut rekomendasi hotel terbaik untukmu:")
        st.dataframe(recommendations)
