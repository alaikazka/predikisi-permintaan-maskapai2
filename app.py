# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load Model dan Tools
@st.cache_resource
def load_artifacts():
    model = joblib.load('airline_model.joblib')
    scaler = joblib.load('scaler.joblib')
    encoders = joblib.load('encoders.joblib')
    return model, scaler, encoders

try:
    model, scaler, encoders = load_artifacts()
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan Anda telah menjalankan 'train_model.py' dan mengupload file .joblib ke GitHub.")
    st.stop()

# Judul Aplikasi
st.title("✈️ Prediksi Permintaan Maskapai Penerbangan")
st.write("Aplikasi Early Warning System untuk memprediksi keberhasilan booking (Booking Complete) berdasarkan data pelanggan.")

# Form Input
with st.form("prediction_form"):
    st.header("Masukkan Detail Penerbangan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_passengers = st.number_input("Jumlah Penumpang", min_value=1, max_value=10, value=1)
        sales_channel = st.selectbox("Sales Channel", encoders['sales_channel'].classes_)
        trip_type = st.selectbox("Tipe Perjalanan", encoders['trip_type'].classes_)
        purchase_lead = st.number_input("Purchase Lead (Hari sebelum terbang)", min_value=0, value=30)
        length_of_stay = st.number_input("Lama Menginap (Hari)", min_value=0, value=5)
        flight_hour = st.slider("Jam Penerbangan", 0, 23, 12)
        
    with col2:
        # Mapping manual untuk flight day agar user mudah memilih
        day_options = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        flight_day_str = st.selectbox("Hari Penerbangan", day_options)
        
        # Mengambil daftar route dan origin yang valid
        route = st.selectbox("Rute (Route)", encoders['route'].classes_)
        booking_origin = st.selectbox("Asal Booking (Origin)", encoders['booking_origin'].classes_)
        
        flight_duration = st.number_input("Durasi Penerbangan (Jam)", min_value=0.0, value=5.0)
        
    st.subheader("Layanan Tambahan")
    c1, c2, c3 = st.columns(3)
    with c1:
        wants_extra_baggage = st.checkbox("Extra Baggage")
    with c2:
        wants_preferred_seat = st.checkbox("Preferred Seat")
    with c3:
        wants_in_flight_meals = st.checkbox("In-Flight Meals")

    submit_btn = st.form_submit_button("Prediksi Booking")

# Logika Prediksi
if submit_btn:
    # 1. Preprocessing Input Data
    day_mapping = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
    flight_day_mapped = day_mapping[flight_day_str]
    
    # Convert checkbox to int
    baggage = 1 if wants_extra_baggage else 0
    seat = 1 if wants_preferred_seat else 0
    meals = 1 if wants_in_flight_meals else 0
    
    # Encode Categorical
    try:
        sales_channel_enc = encoders['sales_channel'].transform([sales_channel])[0]
        trip_type_enc = encoders['trip_type'].transform([trip_type])[0]
        route_enc = encoders['route'].transform([route])[0]
        booking_origin_enc = encoders['booking_origin'].transform([booking_origin])[0]
    except ValueError:
        st.error("Input kategori tidak dikenali oleh model.")
        st.stop()
        
    # Buat DataFrame untuk input (urutan kolom harus sama persis dengan saat training)
    input_data = pd.DataFrame({
        'num_passengers': [num_passengers],
        'sales_channel': [sales_channel_enc],
        'trip_type': [trip_type_enc],
        'purchase_lead': [purchase_lead],
        'length_of_stay': [length_of_stay],
        'flight_hour': [flight_hour],
        'flight_day': [flight_day_mapped],
        'route': [route_enc],
        'booking_origin': [booking_origin_enc],
        'wants_extra_baggage': [baggage],
        'wants_preferred_seat': [seat],
        'wants_in_flight_meals': [meals],
        'flight_duration': [flight_duration]
    })
    
    # Scale Numerical Columns (purchase_lead, length_of_stay, flight_duration)
    # Perhatikan urutan kolom scaling harus sama dengan saat fit
    num_cols = ['purchase_lead', 'length_of_stay', 'flight_duration']
    input_data[num_cols] = scaler.transform(input_data[num_cols])
    
    # 2. Prediksi
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    
    # 3. Tampilkan Hasil
    st.divider()
    if prediction == 1:
        st.success(f"**Prediksi: Booking Complete (Berhasil)**")
        st.write(f"Probabilitas customer akan menyelesaikan booking: **{prob*100:.1f}%**")
    else:
        st.warning(f"**Prediksi: Booking Tidak Selesai**")
        st.write(f"Probabilitas customer akan menyelesaikan booking: **{prob*100:.1f}%**")
