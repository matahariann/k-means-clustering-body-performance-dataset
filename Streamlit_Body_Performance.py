import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Body Performance Analysis",
    page_icon="💪",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 20px;
        background-color: #FF4B4B;
        color: white;
        height: 3rem;
        font-size: 18px;
    }
    .title-container {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .result-container {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Check if model exists, if not create a simple one
model_path = 'Model_KMeans_Body_Performance.pkl'

# Function to create and save a simple model if the original doesn't exist
def create_simple_model():
    # Create a simple pipeline with encoder, scaler and KMeans
    encoders = {'gender': LabelEncoder().fit(['M', 'F'])}
    scaler = StandardScaler()
    
    # Create a simple dataset to fit the scaler
    sample_data = pd.DataFrame({
        'age': [30, 40, 25, 35],
        'gender': ['M', 'F', 'M', 'F'],
        'height_cm': [170, 160, 175, 165],
        'weight_kg': [70, 60, 75, 65],
        'body fat_%': [15, 20, 10, 25],
        'systolic': [120, 115, 125, 118],
        'gripForce': [40, 30, 45, 32],
        'sit and bend forward_cm': [25, 30, 22, 35],
        'sit-ups counts': [30, 20, 35, 25],
        'broad jump_cm': [200, 180, 220, 190]
    })
    
    # Transform categorical data
    for column in ['gender']:
        sample_data[column] = encoders[column].transform(sample_data[column])
    
    # Fit the scaler
    scaler.fit(sample_data)
    
    # Create and fit KMeans model
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(scaler.transform(sample_data))
    
    # Create pipeline dictionary
    pipeline = {
        'encoder': encoders,
        'scaler': scaler,
        'kmeans': kmeans
    }
    
    # Save the model
    joblib.dump(pipeline, model_path)
    return pipeline

# Try to load the model, if it fails create a simple one
try:
    pipeline = joblib.load(model_path)
    st.sidebar.success("Successfully loaded the trained model!")
except:
    st.sidebar.warning("Couldn't find the original model. Using a simplified model for demonstration purposes.")
    pipeline = create_simple_model()

# Extract components from the pipeline
encoder = pipeline['encoder']
scaler = pipeline['scaler']
kmeans = pipeline['kmeans']

# Title section
st.markdown("""
    <div class="title-container">
        <h1 style="color: black;">🏃‍♂️ Analisis Performa Tubuh</h1>
        <p style='font-size: 1.2rem; color: #666;'>Sistem analisis clustering untuk menentukan tingkat performa tubuh Anda</p>
    </div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["📝 Input Data", "ℹ️ Informasi"])

with tab1:
    # Data Pribadi section
    st.markdown("### Data Pribadi")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age = st.number_input('Umur (tahun)', 
                             min_value=0, 
                             max_value=100,
                             value=30,
                             step=1, 
                             format="%d",
                             help="Masukkan umur Anda dalam tahun")
    
    with col2:
        gender = st.selectbox('Jenis Kelamin',
                            options=['Male', 'Female'],
                            help="Pilih jenis kelamin Anda")
    
    with col3:
        height = st.number_input('Tinggi Badan (cm)',
                               min_value=0.0,
                               max_value=250.0,
                               value=170.0,
                               step=0.1,
                               help="Masukkan tinggi badan Anda dalam sentimeter")
    
    with col4:
        weight = st.number_input('Berat Badan (kg)',
                               min_value=0.0,
                               max_value=200.0,
                               value=65.0,
                               step=0.1,
                               help="Masukkan berat badan Anda dalam kilogram")

    # Pengukuran Fisik section
    st.markdown("### Pengukuran Fisik")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        body_fat = st.number_input('Persentase Lemak Tubuh (%)',
                                 min_value=0.0,
                                 max_value=100.0,
                                 value=20.0,
                                 step=0.1,
                                 help="Masukkan persentase lemak tubuh Anda")
        
        systolic = st.number_input('Tekanan Darah Sistolik (mmHg)',
                                 min_value=0,
                                 max_value=200,
                                 value=120,
                                 step=1,
                                 format="%d",
                                 help="Masukkan tekanan darah sistolik Anda")
    
    with col2:
        grip_force = st.number_input('Kekuatan Genggaman (kg)',
                                   min_value=0.0,
                                   max_value=100.0,
                                   value=35.0,
                                   step=0.1,
                                   help="Masukkan kekuatan genggaman Anda dalam kilogram")
        
        sit_and_bend_forward = st.number_input('Sit and Bend Forward (cm)',
                                             min_value=0.0,
                                             max_value=100.0,
                                             value=25.0,
                                             step=0.1,
                                             help="Masukkan hasil pengukuran sit and bend forward dalam sentimeter")
    
    with col3:
        broad_jump = st.number_input('Broad Jump (cm)',
                                   min_value=0.0,
                                   max_value=400.0,
                                   value=200.0,
                                   step=0.1,
                                   help="Masukkan hasil pengukuran broad jump dalam sentimeter")
        
        sit_ups = st.number_input('Jumlah Sit-Ups',
                                min_value=0,
                                max_value=100,
                                value=25,
                                step=1,
                                format="%d",
                                help="Masukkan jumlah sit-ups yang dapat Anda lakukan")


    st.markdown("""
        <style>
        .stButton>button {
            background-color: blue;
            color: white;
            border: 2px solid white;
        }
        .stButton>button:hover {
            background-color: darkblue;
            color: white;
            border: 2px solid white;
        }
        </style>
        """, unsafe_allow_html=True)

    if st.button('Analisis Performa'):
        with st.spinner('Menganalisis data...'):
            gender_code = 'M' if gender == 'Male' else 'F'
            data_baru = pd.DataFrame([{
                'age': age,
                'gender': gender_code,
                'height_cm': height,
                'weight_kg': weight,
                'body fat_%': body_fat,
                'systolic': systolic,
                'gripForce': grip_force,
                'sit and bend forward_cm': sit_and_bend_forward,
                'sit-ups counts': sit_ups,
                'broad jump_cm': broad_jump
            }])

            # Transform categorical data
            for column in ['gender']:
                data_baru[column] = encoder[column].transform(data_baru[[column]])

            # Scale data
            data_baru_scaled = scaler.transform(data_baru)

            # Predict cluster
            cluster_pred = kmeans.predict(data_baru_scaled)
            jarak_cluster = kmeans.transform(data_baru_scaled)

            # Display results
            st.markdown("""
                <div class="result-container">
                    <h2 style='color: black; text-align: center; margin-bottom: 2rem;'>Hasil Analisis</h2>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Jarak ke Cluster Performa Tinggi",
                    value=f"{jarak_cluster[0][0]:.4f}"
                )
            
            with col2:
                st.metric(
                    label="Jarak ke Cluster Performa Rendah",
                    value=f"{jarak_cluster[0][1]:.4f}"
                )

            # Final result
            if cluster_pred[0] == 0:
                st.success("🌟 Anda termasuk dalam kelompok dengan **PERFORMA TINGGI**")
                
                # Add recommendations for high performance
                st.markdown("""
                #### Rekomendasi untuk Mempertahankan Performa Tinggi:
                - Pertahankan rutinitas olahraga Anda saat ini
                - Pastikan asupan nutrisi seimbang dan cukup
                - Jaga kualitas istirahat dan tidur
                - Lakukan evaluasi performa secara berkala setiap 3-6 bulan
                """)
            else:
                st.warning("⚠️ Anda termasuk dalam kelompok dengan **PERFORMA RENDAH**")
                
                # Add recommendations for low performance
                st.markdown("""
                #### Rekomendasi untuk Meningkatkan Performa:
                - Mulai rutinitas olahraga teratur minimal 3x seminggu
                - Tingkatkan konsumsi protein dan nutrisi penting lainnya
                - Kurangi lemak jenuh dan makanan olahan
                - Latih kekuatan dengan latihan beban secara bertahap
                - Tingkatkan fleksibilitas dengan peregangan rutin
                - Konsultasikan dengan profesional kesehatan atau trainer untuk program yang sesuai
                """)

            st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("""
    ### Tentang Aplikasi
    Aplikasi ini menggunakan metode clustering untuk menganalisis performa tubuh berdasarkan berbagai parameter fisik. 
    
    #### Parameter yang Diukur:
    1. **Data Pribadi**
       - Umur
       - Jenis Kelamin
       - Tinggi Badan
       - Berat Badan
    
    2. **Pengukuran Fisik**
       - Persentase Lemak Tubuh
       - Tekanan Darah Sistolik
       - Kekuatan Genggaman
       - Fleksibilitas (Sit and Bend Forward)
       - Broad Jump
       - Jumlah Sit-Ups
    
    #### Interpretasi Hasil
    Sistem akan mengklasifikasikan performa tubuh Anda ke dalam dua kategori:
    - 🌟 **Performa Tinggi**: Menunjukkan kondisi fisik yang optimal
    - ⚠️ **Performa Rendah**: Mengindikasikan perlunya peningkatan kondisi fisik
    
    #### Tips Penggunaan
    - Pastikan semua pengukuran dilakukan dengan akurat
    - Lakukan pengukuran dalam kondisi sehat dan tidak sedang sakit
    - Untuk pengukuran tekanan darah, lakukan dalam keadaan istirahat
    - Pengukuran kekuatan genggaman sebaiknya dilakukan dengan grip strength dynamometer
    - Sit and bend forward diukur dengan duduk di lantai dan mencoba meraih ujung kaki
    - Broad jump diukur dari posisi berdiri kemudian melompat sejauh mungkin ke depan
    - Sit-ups dihitung dalam satu sesi latihan dengan teknik yang benar
    
    #### Catatan Penting
    - Hasil analisis bersifat indikatif dan sebaiknya dikonsultasikan dengan profesional kesehatan
    - Gunakan hasil analisis sebagai motivasi untuk meningkatkan kesehatan dan kebugaran
    - Lakukan pengukuran secara berkala untuk memantau perkembangan
    """)

# Add a sidebar with additional information
with st.sidebar:
    st.header("💡 Tentang Model")
    st.write("""
    Model clustering digunakan untuk mengklasifikasikan performa tubuh berdasarkan berbagai parameter fisik. 
    Model ini menggunakan algoritma K-Means dengan 2 cluster yang merepresentasikan performa tinggi dan rendah.
    """)
    
    st.divider()
    
    st.header("📊 Panduan Nilai")
    st.write("""
    **Persentase Lemak Tubuh:**
    - Pria: 10-20% (ideal), >25% (tinggi)
    - Wanita: 18-28% (ideal), >32% (tinggi)
    
    **Tekanan Darah Sistolik:**
    - Normal: <120 mmHg
    - Tinggi: >130 mmHg
    
    **Kekuatan Genggaman:**
    - Pria: >40kg (baik)
    - Wanita: >30kg (baik)
    
    **Sit and Bend Forward:**
    - >25cm (baik)
    
    **Broad Jump:**
    - Pria: >200cm (baik)
    - Wanita: >170cm (baik)
    
    **Sit-ups:**
    - Pria: >30 (baik)
    - Wanita: >25 (baik)
    """)
