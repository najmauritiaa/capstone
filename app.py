import streamlit as st
import pandas as pd
import ast
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

st.set_page_config(page_title="Peta Hotel & Rekomendasi", layout="wide")

# ---------------------- LOAD DATA -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("indonesia_hotels.csv")
    df = df.dropna()
    df['list_fasilitas'] = df['list_fasilitas'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

df = load_data()

# ------------------ CONTENT BASED FILTER ----------------
def content_based_recommendation(df, hotel_name, top_n=5):
    mlb = MultiLabelBinarizer()
    fasilitas_encoded = mlb.fit_transform(df['list_fasilitas'].apply(lambda x: [f.lower() for f in x]))
    fitur_df = pd.DataFrame(fasilitas_encoded, columns=mlb.classes_, index=df.index)

    try:
        idx = df.index[df['Hotel Name'] == hotel_name][0]
    except IndexError:
        return pd.DataFrame()

    cosine_sim = cosine_similarity(fitur_df)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx]

    top_indices = [i[0] for i in sim_scores[:top_n]]
    return df.iloc[top_indices]

# ---------------------- UI ------------------------------
st.title("🗺️ Peta Hotel dan Rekomendasi Serupa")

# Inisialisasi Peta
m = folium.Map(location=[-2.5, 117.5], zoom_start=5)
marker_cluster = MarkerCluster().add_to(m)

# Mapping koordinat ke ID unik
coord_to_id = {}

for _, row in df.iterrows():
    if row['Hotel Rating'] != 'Belum ada rating':
        image_url = row['Hotel Image'] if 'Hotel Image' in row and pd.notna(row['Hotel Image']) else ""
        html_popup = f"""
            <div style="width:200px">
                <h4>{row['Hotel Name']}</h4>
                <p>⭐ Rating: {row['Hotel Rating']}</p>
                {'<img src="' + image_url + '" width="180">' if image_url else ''}
            </div>
        """
        iframe = folium.IFrame(html=html_popup, width=200, height=200)
        popup = folium.Popup(iframe, max_width=250)

        lat, lon = row['Lattitute'], row['Longitude']
        folium.Marker(
            location=[lat, lon],
            popup=popup,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)

        hotel_id = row['Unnamed: 0'] if 'Unnamed: 0' in row else row.name
        coord_to_id[(round(lat, 5), round(lon, 5))] = hotel_id

# Tampilkan Peta
map_data = st_folium(m, width=700, height=500)

# Tangani Klik
if map_data and map_data.get("last_clicked"):
    clicked_lat = round(map_data["last_clicked"]["lat"], 5)
    clicked_lon = round(map_data["last_clicked"]["lng"], 5)

    hotel_id = coord_to_id.get((clicked_lat, clicked_lon))

    if hotel_id is not None:
        clicked_row = df[df['Unnamed: 0'] == hotel_id].iloc[0] if 'Unnamed: 0' in df.columns else df.iloc[hotel_id]
        clicked_name = clicked_row['Hotel Name']

        st.subheader(f"📌 Detail Hotel: {clicked_name}")
        if pd.notna(clicked_row['Hotel Image']):
            st.image(clicked_row['Hotel Image'], width=400)
        st.write(f"📍 {clicked_row['City']} - {clicked_row['Provinsi']}")
        st.write(f"💰 Rp {int(clicked_row['Min'])} - Rp {int(clicked_row['Max'])}")
        st.write(f"⭐ Rating: {clicked_row['Hotel Rating']}")
        st.write("**Fasilitas:**", ", ".join(clicked_row['list_fasilitas']))
        st.markdown("---")

        st.subheader("🔁 Rekomendasi Hotel Serupa")
        rekomendasi = content_based_recommendation(df, clicked_name)

        if not rekomendasi.empty:
            for _, row in rekomendasi.iterrows():
                st.markdown(f"### 🏨 {row['Hotel Name']}")
                if pd.notna(row['Hotel Image']):
                    st.image(row['Hotel Image'], width=400)
                st.write(f"📍 {row['City']} - {row['Provinsi']}")
                st.write(f"💰 Rp {int(row['Min'])} - Rp {int(row['Max'])}")
                st.write(f"⭐ Rating: {row['Hotel Rating']}")
                st.write("**Fasilitas:**", ", ".join(row['list_fasilitas']))
                st.markdown("---")
        else:
            st.warning("Tidak ditemukan hotel mirip.")
