import streamlit as st
import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ---------------------- CONFIG -----------------------
st.markdown(
    """
    <div style="background-color:#e0f7fa;padding:10px 20px;border-radius:8px">
        <h1 style="color:#006064;">🗺️ Jelajahi Hotel di Indonesia</h1>
    </div>
    """,
    unsafe_allow_html=True
)
# ---------------------- LOAD DATA -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv('indonesia_hotels.csv')
    df = df.dropna()
    df['list_fasilitas'] = df['list_fasilitas'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

df = load_data()

# ---------------------- FILTERING FUNCTION -----------------------
def content_based_recommendation(df, hotel_name, top_n=5):
    mlb = MultiLabelBinarizer()
    fasilitas_encoded = mlb.fit_transform(df['list_fasilitas'].apply(lambda x: [f.lower() for f in x]))
    fitur_df = pd.DataFrame(fasilitas_encoded, columns=mlb.classes_, index=df.index)

    cosine_sim = cosine_similarity(fitur_df)

    try:
        idx = df.index[df['Hotel Name'] == hotel_name][0]
    except IndexError:
        return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx]

    top_indices = [i[0] for i in sim_scores[:top_n]]
    return df.iloc[top_indices]

# ---------------------- MAIN INTERFACE -----------------------
st.header("🗺️ Jelajahi Hotel di Indonesia")

# Buat peta
m = folium.Map(location=[-2.5, 117.5], zoom_start=5)
marker_cluster = MarkerCluster().add_to(m)

for _, row in df.iterrows():
    if row['Hotel Rating'] != 'Belum ada rating':
        image_url = row['Hotel Image'] if 'Hotel Image' in row and pd.notna(row['Hotel Image']) else ""
        popup_content = f"""
            <div style="width:200px">
                <b>{row['Hotel Name']}</b><br>
                Rating: {row['Hotel Rating']}<br>
                {'<img src="' + image_url + '" width="160">' if image_url else ''}
            </div>
        """
        folium.Marker(
            location=[row['Lattitute'], row['Longitude']],
            popup=row['Hotel Name'],
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)

# Tampilkan peta
map_data = st_folium(m, width=800, height=500)

# ---------------------- HOTEL KLIK DETEKSI -----------------------
if map_data and map_data.get("last_object_clicked_popup"):
    selected_hotel_name = map_data["last_object_clicked_popup"]

    st.success(f"🏨 Hotel dipilih: **{selected_hotel_name}**")

    selected_hotel = df[df['Hotel Name'] == selected_hotel_name].iloc[0]

    st.subheader("📋 Detail Hotel")
    if pd.notna(selected_hotel['Hotel Image']):
        st.image(selected_hotel['Hotel Image'], width=400)
    st.markdown(f"**Nama:** {selected_hotel['Hotel Name']}")
    st.markdown(f"**Lokasi:** {selected_hotel['City']}, {selected_hotel['Provinsi']}")
    st.markdown(f"**Rating:** {selected_hotel['Hotel Rating']}")
    st.markdown(f"**Harga:** Rp {int(selected_hotel['Min'])} - Rp {int(selected_hotel['Max'])}")
    st.markdown(f"**Fasilitas:** {', '.join(selected_hotel['list_fasilitas'])}")

    # ---------------------- Rekomendasi Serupa -----------------------
    st.subheader("Rekomendasi Hotel Serupa")

    rekomendasi = content_based_recommendation(df, selected_hotel_name)

    for _, row in rekomendasi.iterrows():
        st.markdown(f"### 🏨 {row['Hotel Name']}")
        if pd.notna(row['Hotel Image']):
            st.image(row['Hotel Image'], width=400)
        st.write(f"📍 {row['City']} - {row['Provinsi']}")
        st.write(f"💰 Rp {int(row['Min'])} - Rp {int(row['Max'])}")
        st.write(f"⭐ Rating: {row['Hotel Rating']}")
        st.write("**Fasilitas:**", ", ".join(row['list_fasilitas']))
        st.markdown("---")
