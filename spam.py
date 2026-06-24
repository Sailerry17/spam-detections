import streamlit as st
import pickle

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Spam Detection Dashboard By Adhitya Prahma",
    page_icon="📩",
    layout="centered"
)

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
<style>

.main {
    background-color: #f4f8ff;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #2563eb;
}

.subtitle {
    text-align: center;
    color: #64748b;
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 8px 24px rgba(37,99,235,0.08);
}

.result-spam {
    background: #fff1f2;
    border-left: 8px solid #ef4444;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}

.result-ham {
    background: #ecfdf5;
    border-left: 8px solid #22c55e;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}

.metric-value {
    font-size: 36px;
    font-weight: bold;
}

.footer {
    text-align:center;
    color:#94a3b8;
    margin-top:30px;
}

.stButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    height: 50px;
    font-weight: bold;
    border: none;
    width: 100%;
}

.stButton button:hover {
    background-color: #1d4ed8;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ==========================
# HEADER
# ==========================
st.markdown("""
<div class='title'>
📩 Spam Detection Dashboard
</div>

<div class='subtitle'>
Deteksi Pesan Spam Menggunakan Algoritma Naive Bayes
</div>
""", unsafe_allow_html=True)

# ==========================
# INPUT CARD
# ==========================
st.markdown("<div class='card'>", unsafe_allow_html=True)

pesan = st.text_area(
    "Masukkan Pesan",
    height=180,
    placeholder="Contoh: Congratulations! You have won a free iPhone. Click here now..."
)

deteksi = st.button("🔍 Analisis Pesan")

st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# PREDIKSI
# ==========================
if deteksi:

    if pesan.strip() == "":
        st.warning("Silakan masukkan pesan terlebih dahulu.")
    else:

        data = vectorizer.transform([pesan])

        prediksi = model.predict(data)[0]

        probabilitas = model.predict_proba(data)[0]

        confidence = max(probabilitas) * 100

        if prediksi == 1:

            st.markdown(f"""
            <div class="result-spam">
                <h2>🚨 SPAM DETECTED</h2>
                <div class="metric-value">{confidence:.2f}%</div>
                <p>Confidence Score</p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-ham">
                <h2>✅ SAFE MESSAGE</h2>
                <div class="metric-value">{confidence:.2f}%</div>
                <p>Confidence Score</p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Spam Probability",
                f"{probabilitas[1] * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Not Spam Probability",
                f"{probabilitas[0] * 100:.2f}%"
            )

# ==========================
# FOOTER
# ==========================
st.markdown("""
<div class='footer'>
Machine Learning Deployment • Naive Bayes Spam Detection<br>
Universitas Nusa Putra
</div>
""", unsafe_allow_html=True)
