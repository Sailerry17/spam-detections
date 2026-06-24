import streamlit as st
import pickle

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Spam Detection Dashboard",
    page_icon="📩",
    layout="centered"
)

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #111827;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 25px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

.result-spam {
    background: linear-gradient(135deg,#ef4444,#dc2626);
    color:white;
    padding:20px;
    border-radius:15px;
    text-align:center;
}

.result-ham {
    background: linear-gradient(135deg,#22c55e,#16a34a);
    color:white;
    padding:20px;
    border-radius:15px;
    text-align:center;
}

.metric {
    font-size:30px;
    font-weight:bold;
}

.footer {
    text-align:center;
    color:gray;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL
# ======================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ======================
# HEADER
# ======================
st.markdown(
    '<div class="title">📩 Spam Message Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Deployment using Naive Bayes</div>',
    unsafe_allow_html=True
)

# ======================
# INPUT CARD
# ======================
st.markdown('<div class="card">', unsafe_allow_html=True)

pesan = st.text_area(
    "Input Message",
    height=180,
    placeholder="Type or paste a message here..."
)

deteksi = st.button(
    "🔍 Analyze Message",
    use_container_width=True
)

st.markdown('</div>', unsafe_allow_html=True)

# ======================
# PREDICTION
# ======================
if deteksi:

    if pesan.strip() == "":
        st.warning("Please enter a message first.")
    else:

        X = vectorizer.transform([pesan])

        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]

        confidence = max(prob) * 100

        st.markdown("<br>", unsafe_allow_html=True)

        if pred == 1:

            st.markdown(f"""
            <div class="result-spam">
                <h2>🚨 SPAM DETECTED</h2>
                <div class="metric">{confidence:.2f}%</div>
                <p>Confidence Score</p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-ham">
                <h2>✅ NOT SPAM</h2>
                <div class="metric">{confidence:.2f}%</div>
                <p>Confidence Score</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Spam Probability",
                f"{prob[1]*100:.2f}%"
            )

        with col2:
            st.metric(
                "Not Spam Probability",
                f"{prob[0]*100:.2f}%"
            )

# ======================
# FOOTER
# ======================
st.markdown(
    """
    <div class="footer">
        Universitas Nusa Putra • Machine Learning • Deployment with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
