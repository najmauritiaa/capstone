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
df = pd.read_csv("/Users/nikitaandrea/Downloads/global-hotels.csv")

# Preprocessing code (same as before)
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

# Recommendation function
def recommend_by_pref(df_knn_imputed, tfidf, scaler, final_features,
                      rating, score, number_reviews, price_range, top_n=5):
    pref_text = f"{rating}"
    pref_tfidf = tfidf.transform([pref_text])

    avg_price = sum(price_range) / 2
    pref_num = scaler.transform([[score, number_reviews, avg_price]])

    pref_vec = hstack([pref_tfidf, pref_num])

    sims = cosine_similarity(pref_vec, final_features).flatten()

    df_knn_imputed['sim_score'] = sims
    df_filtered = df_knn_imputed[df_knn_imputed['Price'].between(*price_range)]
    df_sorted = df_filtered.sort_values(by='sim_score', ascending=False)

    return df_sorted.head(top_n)[['Hotel_Name', 'City', 'Room_Type', 'Rating', 'Score', 'Number_Reviews', 'Price', 'sim_score']]

# Streamlit UI
st.title('Hotel Recommendation System')
st.write('Find the best hotels based on your preferences')

# Input Fields
rating = st.selectbox('Select Hotel Rating', ['Very Good', 'Good', 'Excellent', 'Average', 'Poor'])
score = st.slider('Hotel Score', 1.0, 10.0, 8.0)
number_reviews = st.slider('Number of Reviews', 0, 5000, 1000)
price_range = st.slider('Price Range', 0, 1000, (100, 300))

# Get recommendations
if st.button('Get Recommendations'):
    recommendations = recommend_by_pref(
        df_knn_imputed=df_knn_imputed,
        tfidf=tfidf,
        scaler=scaler,
        final_features=final_features,
        rating=rating,
        score=score,
        number_reviews=number_reviews,
        price_range=price_range,
        top_n=5
    )

    st.write("Here are the top 5 recommended hotels:")
    st.dataframe(recommendations)
