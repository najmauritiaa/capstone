import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ---------------------- CONFIG -----------------------
st.set_page_config(layout="wide")
st.markdown(
    """
    <div style="background-color:#e0f7fa;padding:10px 20px;border-radius:8px">
        <h1 style="color:#006064;">🏨 Sistem Rekomendasi Hotel di Indonesia</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------- LOAD DATA -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv('indonesia_hotels.csv')
    df = df.dropna()

    # Tukar Min dan Max jika Min > Max
    df[['Min', 'Max']] = df.apply(
        lambda row: pd.Series([row['Max'], row['Min']]) if row['Min'] > row['Max'] else pd.Series([row['Min'], row['Max']]),
        axis=1
    )

    def parse_fasilitas(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                return [val]
        elif isinstance(val, list):
            return val
        else:
            return []

    df['list_fasilitas'] = df['list_fasilitas'].apply(parse_fasilitas)
    return df

df = load_data()

# ---------------------- CONTENT-BASED FUNCTION -----------------------
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

# ---------------------- TABS -----------------------
tab1, tab2, tab3 = st.tabs(["💡 Rekomendasi Berdasarkan Emosi", "🔁 Content-Based Rekomendasi", "🗺️ Peta Interaktif"])

# ---------------------- TAB 1 -----------------------
with tab1:
    st.subheader("Langkah 1: Jawab pertanyaan berikut ini")
    questions = [
        "Apa yang kamu rasakan saat ini?",
        "Apakah kamu merasa kesepian hari ini?",
        "Apakah kamu merasa stres atau kewalahan akhir-akhir ini?",
        "Apakah kamu ingin berbicara dengan seseorang saat ini?",
        "Apakah kamu merasa bosan dengan rutinitasmu?",
        "Apakah kamu ingin mencoba hal baru hari ini?",
        "Apakah kamu merasa dihargai oleh orang-orang di sekitarmu?",
        "Apakah kamu sedang merindukan seseorang secara romantis?",
        "Apakah kamu ingin menyendiri dan menjauh dari keramaian?",
        "Apakah kamu merasa tubuhmu lelah atau tidak bersemangat?",
        "Apakah kamu merasa senang ketika dikelilingi banyak orang?",
        "Apakah kamu ingin merasakan adrenalin atau sesuatu yang memacu semangat?",
        "Apakah kamu merasa butuh memahami dirimu lebih dalam?",
        "Apakah kamu merasa kurang terhubung dengan orang-orang terdekat?",
        "Kalau bisa memilih aktivitas sekarang, kamu ingin:"
    ]

    answer_options = {
        "Apa yang kamu rasakan saat ini?": ["senang", "cemas", "sedih"],
        "Kalau bisa memilih aktivitas sekarang, kamu ingin:": ["tidur", "menyendiri", "pacaran", "jalan-jalan"]
    }

    answers = {}
    for q in questions:
        opts = answer_options.get(q, ["ya", "tidak"])
        answers[q] = st.selectbox(q, opts)

    st.subheader("Langkah 2: Masukkan Budget")
    budget_min = st.number_input("Budget Minimum (Rp)", min_value=0, value=300000, step=50000)
    budget_max = st.number_input("Budget Maksimum (Rp)", min_value=0, value=800000, step=50000)

    if st.button("🎯 Cari Rekomendasi Hotel"):
        rules = {
            "Meditation": [("Apa yang kamu rasakan saat ini?", ["cemas", "sedih"]),
                            ("Apakah kamu merasa stres atau kewalahan akhir-akhir ini?", ["ya"]),
                            ("Apakah kamu merasa tubuhmu lelah atau tidak bersemangat?", ["ya"]),
                            ("Apakah kamu merasa butuh memahami dirimu lebih dalam?", ["ya"])],
            "Me-time": [("Apakah kamu ingin menyendiri dan menjauh dari keramaian?", ["ya"]),
                        ("Kalau bisa memilih aktivitas sekarang, kamu ingin:", ["menyendiri", "tidur"])],
            "Socialized": [("Apakah kamu merasa kesepian hari ini?", ["ya"]),
                           ("Apakah kamu ingin berbicara dengan seseorang saat ini?", ["ya"]),
                           ("Apakah kamu merasa senang ketika dikelilingi banyak orang?", ["ya"])],
            "Date": [("Apakah kamu sedang merindukan seseorang secara romantis?", ["ya"]),
                     ("Kalau bisa memilih aktivitas sekarang, kamu ingin:", ["pacaran"])],
            "Adventure": [("Apakah kamu merasa bosan dengan rutinitasmu?", ["ya"]),
                          ("Apakah kamu ingin mencoba hal baru hari ini?", ["ya"]),
                          ("Apakah kamu ingin merasakan adrenalin atau sesuatu yang memacu semangat?", ["ya"]),
                          ("Kalau bisa memilih aktivitas sekarang, kamu ingin:", ["jalan-jalan"])],
            "Support": [("Apakah kamu merasa kurang terhubung dengan orang-orang terdekat?", ["ya"]),
                        ("Apakah kamu merasa dihargai oleh orang-orang di sekitarmu?", ["tidak"])]
        }

        scores = {need: sum(1 for q, valid in conds if answers.get(q) in valid) for need, conds in rules.items()}
        dominant_need = max(scores, key=scores.get)
        st.success(f"✅ Kebutuhan Emosionalmu Saat Ini: {dominant_need}")

        need_to_facilities = {
            "Meditation": ['spa', 'pijat', 'yoga', 'perpustakaan', 'taman', 'gazebo', 'atmosfer tenang'],
            "Adventure": ['bersepeda', 'mendaki', 'memancing', 'kayak', 'menyelam', 'snorkeling', 'berkuda'],
            "Date": ['bar', 'private', 'romantis', 'restoran', 'teras', 'laut'],
            "Socialized": ['klub', 'karaoke', 'bar', 'game', 'anak', 'umum'],
            "Me-time": ['perpustakaan', 'spa', 'yoga', 'tv', 'teras', 'tenang'],
            "Support": ['layanan', 'resepsionis', 'laundry', 'keamanan', 'apotek', 'pelayanan']
        }

        needed_keywords = need_to_facilities.get(dominant_need, [])

        def count_matching_facilities(hotel_facilities, keywords):
            return sum(any(kw.lower() in f.lower() for f in hotel_facilities) for kw in keywords)

        filtered = df[(df['Min'] >= budget_min) & (df['Max'] <= budget_max)].copy()
        filtered['matching_score'] = filtered['list_fasilitas'].apply(lambda x: count_matching_facilities(x, needed_keywords))
        top_5 = filtered.sort_values(by='matching_score', ascending=False).head(5)

        st.subheader("🏨 Top 5 Hotel Berdasarkan Emosi & Budget")
        st.dataframe(top_5[['Hotel Name', 'City', 'Min', 'Max', 'list_fasilitas', 'matching_score']])

# ---------------------- TAB 2 -----------------------
with tab2:
    st.subheader("🔁 Content-Based Filtering dari Top 5")
    df['fasilitas_str'] = df['list_fasilitas'].apply(lambda x: ' '.join(map(str, x)))
    vectorizer = CountVectorizer()
    fasilitas_matrix = vectorizer.fit_transform(df['fasilitas_str'])
    facility_sim = cosine_similarity(fasilitas_matrix)

    df['avg_price'] = (df['Min'] + df['Max']) / 2
    scaler = MinMaxScaler()
    norm_price = scaler.fit_transform(df[['avg_price']])
    price_sim = 1 - np.abs(norm_price - norm_price.T)
    combined_sim = 0.7 * facility_sim + 0.3 * price_sim

    top_indices = df.sort_values(by='avg_price').head(5).index.tolist()
    sim_scores = np.mean(combined_sim[top_indices], axis=0)
    sorted_indices = np.argsort(sim_scores)[::-1]
    content_based_indices = [i for i in sorted_indices if i not in top_indices][:10]

    top_10_cb = df.iloc[content_based_indices].copy()
    top_10_cb['similarity_score'] = sim_scores[content_based_indices]

    st.dataframe(top_10_cb[['Hotel Name', 'City', 'Min', 'Max', 'list_fasilitas', 'similarity_score']])

# ---------------------- TAB 3 -----------------------
with tab3:
    st.subheader("🗺️ Peta Interaktif Hotel")
    m = folium.Map(location=[-2.5, 117.5], zoom_start=5)
    cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        if row['Hotel Rating'] != 'Belum ada rating':
            image_url = row['Hotel Image'] if pd.notna(row['Hotel Image']) else ""
            popup_html = f"""
                <div style='width:200px'>
                    <b>{row['Hotel Name']}</b><br>
                    Rating: {row['Hotel Rating']}<br>
                    {'<img src="' + image_url + '" width="160">' if image_url else ''}
                </div>
            """
            popup = folium.Popup(folium.IFrame(html=popup_html, width=200, height=200), max_width=250)
            folium.Marker(
                location=[row['Lattitute'], row['Longitude']],
                popup=row['Hotel Name'],
                icon=folium.Icon(color='blue')
            ).add_to(cluster)

    map_data = st_folium(m, width=800, height=500)

    if map_data and map_data.get("last_object_clicked_popup"):
        selected_hotel_name = map_data["last_object_clicked_popup"]
        st.success(f"🏨 Hotel dipilih: *{selected_hotel_name}*")
        hotel = df[df['Hotel Name'] == selected_hotel_name].iloc[0]

        st.subheader("📋 Detail Hotel")
        if pd.notna(hotel['Hotel Image']):
            st.image(hotel['Hotel Image'], width=400)
        st.markdown(f"*Nama:* {hotel['Hotel Name']}")
        st.markdown(f"*Lokasi:* {hotel['City']}, {hotel['Provinsi']}")
        st.markdown(f"*Rating:* {hotel['Hotel Rating']}")
        st.markdown(f"*Harga:* Rp {int(hotel['Min'])} - Rp {int(hotel['Max'])}")
        st.markdown(f"*Fasilitas:* {', '.join(hotel['list_fasilitas'])}")

        st.subheader("Rekomendasi Serupa dari Hotel Ini")
        map_recs = content_based_recommendation(df[df['Hotel Name'] != selected_hotel_name], selected_hotel_name)

        for _, row in map_recs.iterrows():
            st.markdown(f"### 🏨 {row['Hotel Name']}")
            if pd.notna(row['Hotel Image']):
                st.image(row['Hotel Image'], width=400)
            st.write(f"📍 {row['City']} - {row['Provinsi']}")
            st.write(f"💰 Rp {int(row['Min'])} - Rp {int(row['Max'])}")
            st.write(f"⭐ Rating: {row['Hotel Rating']}")
            st.write("*Fasilitas:*", ", ".join(row['list_fasilitas']))
            st.markdown("---")
