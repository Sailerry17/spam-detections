import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📩 Spam Message Detection")

st.write("Masukkan pesan yang ingin diperiksa")

message = st.text_area("Pesan")

if st.button("Prediksi"):

    if message.strip() == "":
        st.warning("Masukkan pesan terlebih dahulu")
    else:

        data = vectorizer.transform([message])

        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("🚨 SPAM")
        else:
            st.success("✅ BUKAN SPAM")