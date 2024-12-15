import pickle
import pandas as pd
import streamlit as st

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

# Load model
@st.cache_resource
def load_model():
    with open('Model_KMeans_Body_Performance.pkl', 'rb') as f:
        return pickle.load(f)

pipeline = load_model()
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
                               step=0.1,
                               help="Masukkan tinggi badan Anda dalam sentimeter")
    
    with col4:
        weight = st.number_input('Berat Badan (kg)',
                               min_value=0.0,
                               max_value=200.0,
                               step=0.1,
                               help="Masukkan berat badan Anda dalam kilogram")

    # Pengukuran Fisik section
    st.markdown("### Pengukuran Fisik")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        body_fat = st.number_input('Persentase Lemak Tubuh (%)',
                                 min_value=0.0,
                                 max_value=100.0,
                                 step=0.1,
                                 help="Masukkan persentase lemak tubuh Anda")
        
        systolic = st.number_input('Tekanan Darah Sistolik (mmHg)',
                                 min_value=0,
                                 max_value=200,
                                 step=1,
                                 format="%d",
                                 help="Masukkan tekanan darah sistolik Anda")
    
    with col2:
        grip_force = st.number_input('Kekuatan Genggaman (kg)',
                                   min_value=0.0,
                                   max_value=100.0,
                                   step=0.1,
                                   help="Masukkan kekuatan genggaman Anda dalam kilogram")
        
        sit_and_bend_forward = st.number_input('Sit and Bend Forward (cm)',
                                             min_value=0.0,
                                             max_value=100.0,
                                             step=0.1,
                                             help="Masukkan hasil pengukuran sit and bend forward dalam sentimeter")
    
    with col3:
        broad_jump = st.number_input('Broad Jump (cm)',
                                   min_value=0.0,
                                   max_value=400.0,
                                   step=0.1,
                                   help="Masukkan hasil pengukuran broad jump dalam sentimeter")
        
        sit_ups = st.number_input('Jumlah Sit-Ups',
                                min_value=0,
                                max_value=100,
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
                data_baru[column] = encoder[column].transform(data_baru[column])

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
            else:
                st.warning("⚠️ Anda termasuk dalam kelompok dengan **PERFORMA RENDAH**")

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
