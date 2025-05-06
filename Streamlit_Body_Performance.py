import pandas as pd
import streamlit as st
import numpy as np
import pickle
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

# Simple model implementation (without requiring sklearn)
class SimpleBodyPerformanceModel:
    def __init__(self):
        # Simplified thresholds based on typical fitness guidelines
        self.thresholds = {
            'body fat_%': {'Male': 20, 'Female': 28},
            'systolic': 130,
            'gripForce': {'Male': 40, 'Female': 30},
            'sit and bend forward_cm': 25,
            'broad jump_cm': {'Male': 200, 'Female': 170},
            'sit-ups counts': {'Male': 30, 'Female': 25}
        }
    
    def predict(self, data):
        """Predict performance based on simplified rules"""
        gender = 'Male' if data['gender'] == 'M' else 'Female'
        
        # Count how many metrics meet high performance thresholds
        score = 0
        total_metrics = 6
        
        # Body fat (lower is better)
        if data['body fat_%'] <= self.thresholds['body fat_%'][gender]:
            score += 1
            
        # Systolic blood pressure (lower is better)
        if data['systolic'] <= self.thresholds['systolic']:
            score += 1
            
        # Grip force (higher is better)
        if data['gripForce'] >= self.thresholds['gripForce'][gender]:
            score += 1
            
        # Sit and bend forward (higher is better)
        if data['sit and bend forward_cm'] >= self.thresholds['sit and bend forward_cm']:
            score += 1
            
        # Broad jump (higher is better)
        if data['broad jump_cm'] >= self.thresholds['broad jump_cm'][gender]:
            score += 1
            
        # Sit-ups (higher is better)
        if data['sit-ups counts'] >= self.thresholds['sit-ups counts'][gender]:
            score += 1
            
        # Calculate performance ratio
        performance_ratio = score / total_metrics
        
        # Return cluster prediction (0 for high performance, 1 for low performance)
        # and distances (calculated as 1 - performance_ratio for high, performance_ratio for low)
        if performance_ratio >= 0.5:
            return 0, [(1 - performance_ratio), performance_ratio]
        else:
            return 1, [(1 - performance_ratio), performance_ratio]

# Create the model
model = SimpleBodyPerformanceModel()

# Title section
st.markdown("""
    <div class="title-container">
        <h1 style="color: black;">🏃‍♂️ Analisis Performa Tubuh</h1>
        <p style='font-size: 1.2rem; color: #666;'>Sistem analisis untuk menentukan tingkat performa tubuh Anda</p>
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
            data_input = {
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
            }

            # Predict cluster
            cluster_pred, distances = model.predict(data_input)
            jarak_cluster = [distances]  # Format for consistency

            # Display results
            st.markdown("""
                <div class="result-container">
                    <h2 style='color: black; text-align: center; margin-bottom: 2rem;'>Hasil Analisis</h2>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Kedekatan dengan Profil Performa Tinggi",
                    value=f"{(1-jarak_cluster[0][1])*100:.1f}%"
                )
            
            with col2:
                st.metric(
                    label="Kedekatan dengan Profil Performa Rendah",
                    value=f"{jarak_cluster[0][1]*100:.1f}%"
                )

            # Final result
            if cluster_pred == 0:
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
                
                # Show which metrics need improvement
                st.markdown("#### Area yang Perlu Ditingkatkan:")
                
                metrics_to_improve = []
                gender_display = 'Male' if gender_code == 'M' else 'Female'
                
                if body_fat > model.thresholds['body fat_%'][gender_display]:
                    metrics_to_improve.append(f"- **Persentase Lemak Tubuh**: {body_fat}% (target: di bawah {model.thresholds['body fat_%'][gender_display]}%)")
                
                if systolic > model.thresholds['systolic']:
                    metrics_to_improve.append(f"- **Tekanan Darah**: {systolic} mmHg (target: di bawah {model.thresholds['systolic']} mmHg)")
                
                if grip_force < model.thresholds['gripForce'][gender_display]:
                    metrics_to_improve.append(f"- **Kekuatan Genggaman**: {grip_force} kg (target: di atas {model.thresholds['gripForce'][gender_display]} kg)")
                
                if sit_and_bend_forward < model.thresholds['sit and bend forward_cm']:
                    metrics_to_improve.append(f"- **Fleksibilitas**: {sit_and_bend_forward} cm (target: di atas {model.thresholds['sit and bend forward_cm']} cm)")
                
                if broad_jump < model.thresholds['broad jump_cm'][gender_display]:
                    metrics_to_improve.append(f"- **Broad Jump**: {broad_jump} cm (target: di atas {model.thresholds['broad jump_cm'][gender_display]} cm)")
                
                if sit_ups < model.thresholds['sit-ups counts'][gender_display]:
                    metrics_to_improve.append(f"- **Sit-ups**: {sit_ups} (target: di atas {model.thresholds['sit-ups counts'][gender_display]})")
                
                for metric in metrics_to_improve:
                    st.markdown(metric)
                
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
    Aplikasi ini menggunakan metode analisis untuk mengevaluasi performa tubuh berdasarkan berbagai parameter fisik. 
    
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
    Model analisis ini menggunakan pendekatan berbasis aturan untuk mengklasifikasikan performa tubuh berdasarkan standar kesehatan dan kebugaran yang diakui.
    """)
    
    st.divider()
    
    st.header("📊 Panduan Nilai")
    st.markdown("""
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
