import streamlit as st
import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ---------------------- CONFIG -----------------------
st.markdown(
    """
    <div style="background-color:#e0f7fa;padding:10px 20px;border-radius:8px">
        <h1 style="color:#006064;">🏨 Sistem Rekomendasi Hotel di Indonesia</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("🏨 Aplikasi Rekomendasi Hotel di Indonesia")

st.markdown("---")
st.header("📌 Section 1: Rekomendasi Berdasarkan Emosi dan Budget")

# Load dataset
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

indonesia_hotels = load_data()

# ================= UI =================
st.title("💡 Hotel Recommender App (Indonesia)")
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
        "Meditation": [
            ("Apa yang kamu rasakan saat ini?", ["cemas", "sedih"]),
            ("Apakah kamu merasa stres atau kewalahan akhir-akhir ini?", ["ya"]),
            ("Apakah kamu merasa tubuhmu lelah atau tidak bersemangat?", ["ya"]),
            ("Apakah kamu merasa butuh memahami dirimu lebih dalam?", ["ya"])
        ],
        "Me-time": [
            ("Apakah kamu ingin menyendiri dan menjauh dari keramaian?", ["ya"]),
            ("Kalau bisa memilih aktivitas sekarang, kamu ingin:", ["menyendiri", "tidur"])
        ],
        "Socialized": [
            ("Apakah kamu merasa kesepian hari ini?", ["ya"]),
            ("Apakah kamu ingin berbicara dengan seseorang saat ini?", ["ya"]),
            ("Apakah kamu merasa senang ketika dikelilingi banyak orang?", ["ya"])
        ],
        "Date": [
            ("Apakah kamu sedang merindukan seseorang secara romantis?", ["ya"]),
            ("Kalau bisa memilih aktivitas sekarang, kamu ingin:", ["pacaran"])
        ],
        "Adventure": [
            ("Apakah kamu merasa bosan dengan rutinitasmu?", ["ya"]),
            ("Apakah kamu ingin mencoba hal baru hari ini?", ["ya"]),
            ("Apakah kamu ingin merasakan adrenalin atau sesuatu yang memacu semangat?", ["ya"]),
            ("Kalau bisa memilih aktivitas sekarang, kamu ingin:", ["jalan-jalan"])
        ],
        "Support": [
            ("Apakah kamu merasa kurang terhubung dengan orang-orang terdekat?", ["ya"]),
            ("Apakah kamu merasa dihargai oleh orang-orang di sekitarmu?", ["tidak"])
        ]
    }

    scores = {}
    for need, condition_list in rules.items():
        score = 0
        for q, valid_answers in condition_list:
            if answers.get(q) in valid_answers:
                score += 1
        scores[need] = score

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

    def count_matching_facilities(hotel_facilities, needed_keywords):
        return sum(any(kw.lower() in f.lower() for f in hotel_facilities) for kw in needed_keywords)

    filtered_hotels = indonesia_hotels[
        (indonesia_hotels['Min'] >= budget_min) & (indonesia_hotels['Max'] <= budget_max)
    ].copy()

    if not filtered_hotels.empty:
        filtered_hotels['matching_score'] = filtered_hotels['list_fasilitas'].apply(
            lambda x: count_matching_facilities(x, needed_keywords)
        )
        top_5 = filtered_hotels.sort_values(by='matching_score', ascending=False).head(5)
        st.subheader("🏨 Rekomendasi Hotel Berdasarkan Emosi & Budget (Top 5)")
        st.dataframe(top_5[['Hotel Name', 'City', 'Min', 'Max', 'list_fasilitas', 'matching_score']])
    else:
        st.warning("⚠ Tidak ditemukan hotel dalam budget tersebut. Menampilkan tanpa filter budget.")
        indonesia_hotels['matching_score'] = indonesia_hotels['list_fasilitas'].apply(
            lambda x: count_matching_facilities(x, needed_keywords)
        )
        top_5 = indonesia_hotels.sort_values(by='matching_score', ascending=False).head(5)
        st.dataframe(top_5[['Hotel Name', 'City', 'Min', 'Max', 'list_fasilitas', 'matching_score']])

    # ============ Content-based Filtering ============
    indonesia_hotels['fasilitas_str'] = indonesia_hotels['list_fasilitas'].apply(lambda x: ' '.join(map(str, x)))
    vectorizer = CountVectorizer()
    facility_matrix = vectorizer.fit_transform(indonesia_hotels['fasilitas_str'])
    facility_sim = cosine_similarity(facility_matrix)

    indonesia_hotels['avg_price'] = (indonesia_hotels['Min'] + indonesia_hotels['Max']) / 2
    scaler = MinMaxScaler()
    normalized_price = scaler.fit_transform(indonesia_hotels[['avg_price']])
    price_diff_matrix = np.abs(normalized_price - normalized_price.T)
    price_sim = 1 - price_diff_matrix

    combined_sim = 0.7 * facility_sim + 0.3 * price_sim
    top_5_indices = top_5.index.tolist()
    sim_scores = np.mean(combined_sim[top_5_indices], axis=0)

    sorted_indices = np.argsort(sim_scores)[::-1]
    content_based_indices = [i for i in sorted_indices if i not in top_5_indices][:10]

    top_10_cb = indonesia_hotels.iloc[content_based_indices].copy()
    top_10_cb['similarity_score'] = sim_scores[content_based_indices]

    st.subheader("🔁 Rekomendasi Hotel Serupa (Content-Based)")
    st.dataframe(top_10_cb[['Hotel Name', 'City', 'Min', 'Max', 'list_fasilitas', 'similarity_score']])

    st.markdown("---")
    st.header("📌 Section 2: Rekomendasi Berdasarkan Hotel Tertentu dan Peta")

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
st.header("🗺 Jelajahi Hotel di Indonesia")

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

    st.success(f"🏨 Hotel dipilih: *{selected_hotel_name}*")

    selected_hotel = df[df['Hotel Name'] == selected_hotel_name].iloc[0]

    st.subheader("📋 Detail Hotel")
    if pd.notna(selected_hotel['Hotel Image']):
        st.image(selected_hotel['Hotel Image'], width=400)
    st.markdown(f"*Nama:* {selected_hotel['Hotel Name']}")
    st.markdown(f"*Lokasi:* {selected_hotel['City']}, {selected_hotel['Provinsi']}")
    st.markdown(f"*Rating:* {selected_hotel['Hotel Rating']}")
    st.markdown(f"*Harga:* Rp {int(selected_hotel['Min'])} - Rp {int(selected_hotel['Max'])}")
    st.markdown(f"*Fasilitas:* {', '.join(selected_hotel['list_fasilitas'])}")

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
        st.write("*Fasilitas:*", ", ".join(row['list_fasilitas']))
        st.markdown("---")
