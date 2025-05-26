import streamlit as st
import pandas as pd
import ast
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

st.set_page_config(page_title="Rekomendasi Hotel", layout="wide")

# ---------------------- LOAD DATA -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("indonesia_hotels.csv")
    df = df.dropna()
    df['list_fasilitas'] = df['list_fasilitas'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df.reset_index(drop=True)

df = load_data()

# ------------------ CONTENT BASED FILTER ----------------
def content_based_recommendation(df, hotel_name, top_n=5):
    mlb = MultiLabelBinarizer()
    fasilitas_encoded = mlb.fit_transform(df['list_fasilitas'].apply(lambda x: [f.lower() for f in x]))
    fitur_df = pd.DataFrame(fasilitas_encoded, columns=mlb.classes_, index=df.index)

    try:
        idx = df[df['Hotel Name'].str.strip().str.lower() == hotel_name.strip().lower()].index[0]
    except IndexError:
        return pd.DataFrame()

    cosine_sim = cosine_similarity(fitur_df)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx]

    top_indices = [i[0] for i in sim_scores[:top_n]]
    return df.iloc[top_indices]

# -------------------- TAB LAYOUT ------------------------
tab1, tab2 = st.tabs(["🗺️ Peta Hotel", "🤖 Rekomendasi Hotel"])

# -------------------- TAB 1: MAP ------------------------
with tab1:
    st.header("🗺️ Peta Sebaran Hotel")

    m = folium.Map(location=[-2.5, 117.5], zoom_start=5)
    marker_cluster = MarkerCluster().add_to(m)

    coord_to_index = {}

    for idx, row in df.iterrows():
        if row['Hotel Rating'] != 'Belum ada rating':
            image_url = row['Hotel Image'] if pd.notna(row['Hotel Image']) else ""
            popup_html = f"""
                <div style="width:200px">
                    <h4>{row['Hotel Name']}</h4>
                    <p>⭐ Rating: {row['Hotel Rating']}</p>
                    {'<img src="' + image_url + '" width="180">' if image_url else ''}
                </div>
            """
            iframe = folium.IFrame(html=popup_html, width=200, height=200)
            popup = folium.Popup(iframe, max_width=250)

            lat, lon = row['Lattitute'], row['Longitude']
            folium.Marker(
                location=[lat, lon],
                popup=popup,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(marker_cluster)

            coord_to_index[(round(lat, 5), round(lon, 5))] = idx

    map_data = st_folium(m, width=700, height=500)

    if map_data and map_data.get("last_clicked"):
        clicked_lat = round(map_data["last_clicked"]["lat"], 5)
        clicked_lon = round(map_data["last_clicked"]["lng"], 5)
        clicked_idx = coord_to_index.get((clicked_lat, clicked_lon))

        if clicked_idx is not None:
            st.session_state.clicked_index = clicked_idx
            st.success(f"✅ Hotel dipilih: {df.loc[clicked_idx]['Hotel Name']}")

# -------------------- TAB 2: REKOMENDASI ----------------
with tab2:
    st.header("🤖 Rekomendasi Hotel Berdasarkan Pilihan")

    if "clicked_index" not in st.session_state:
        st.info("Klik salah satu marker di tab 'Peta Hotel' untuk melihat rekomendasi.")
    else:
        idx = st.session_state.clicked_index
        clicked_hotel = df.loc[idx]

        st.subheader(f"📌 Hotel: {clicked_hotel['Hotel Name']}")
        if pd.notna(clicked_hotel['Hotel Image']):
            st.image(clicked_hotel['Hotel Image'], width=400)
        st.write(f"📍 {clicked_hotel['City']} - {clicked_hotel['Provinsi']}")
        st.write(f"💰 Rp {int(clicked_hotel['Min'])} - Rp {int(clicked_hotel['Max'])}")
        st.write(f"⭐ Rating: {clicked_hotel['Hotel Rating']}")
        st.write("**Fasilitas:**", ", ".join(clicked_hotel['list_fasilitas']))
        st.markdown("---")

        st.subheader("🔁 Rekomendasi Hotel Serupa")
        rekomendasi = content_based_recommendation(df, clicked_hotel['Hotel Name'])

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
