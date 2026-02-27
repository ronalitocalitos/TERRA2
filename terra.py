import streamlit as st
import pandas as pd
import joblib
import firebase_admin
from firebase_admin import credentials, firestore

# ==========================================
# ⚙️ 1. Page Configuration (ต้องอยู่บรรทัดที่ 1 เท่านั้น)
# ==========================================
# หมายเหตุ: Streamlit ไม่อนุญาตให้เปลี่ยน Page Title ผ่าน set_page_config หลังจากรันไปแล้ว
# เราจึงตั้งชื่อกลางไว้ และใช้การแสดงผลบนหน้าจอเพื่อบอกสถานะหน้าแทน
st.set_page_config(
    page_title="TERRA - AI System",
    page_icon="🌱",
    layout="wide"
)

# ==========================================
# 🔥 2. Backend & Model Loading
# ==========================================
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        try:
            key_dict = st.secrets["firebase_key"]
            cred = credentials.Certificate(dict(key_dict))
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Firebase Configuration Error: {e}")
    return firestore.client()

@st.cache_resource
def load_terra_model():
    return joblib.load("terra_model.pkl")

# เรียกใช้งาน
db = init_firebase()
model_data = load_terra_model()
clf = model_data['classifier']
reg = model_data['regressor']

# ==========================================
# 📊 3. Helper Functions
# ==========================================
def get_sensor_latest(device_id):
    try:
        query = db.collection('devices').document(device_id).collection('soilData')
        docs = query.order_by("__name__", direction=firestore.Query.DESCENDING).limit(1).get()
        for doc in docs:
            data = doc.to_dict()
            return {
                'N': data.get('N', 0), 'P': data.get('P', 0), 'K': data.get('K', 0),
                'pH': data.get('pH', 0), 'Moist': data.get('moisture', 0), 
                'temp': data.get('temperature', 0), 'cond': data.get('conductivity', 0)
            }
    except Exception as e:
        st.error(f"Firebase Error: {e}")
    return None

# ==========================================
# 🧠 4. Session State Management
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_device' not in st.session_state:
    st.session_state.current_device = None

# ==========================================
# 🚪 5. Login Page View
# ==========================================
if not st.session_state.logged_in:
    # แสดง Title จำลองที่มุมขวา
    st.markdown("<p style='text-align: right; color: #888;'>TERRA - home</p>", unsafe_html=True)
    
    st.markdown("<br><br>", unsafe_html=True)
    # จุดที่ 1: จัดกึ่งกลางหัวข้อ Login
    st.markdown("<h1 style='text-align: center; color: #2E7D32;'>เข้าสู่ระบบ TERRA (WEFARM)</h1>", unsafe_html=True)
    st.markdown("<p style='text-align: center;'>กรุณากรอกรหัสเครื่องเซนเซอร์ (Serial Number) เพื่อเข้าดูข้อมูล</p>", unsafe_html=True)
    
    # จัด Form ให้อยู่กึ่งกลาง
    _, col_mid, _ = st.columns([1, 1.5, 1])
    with col_mid:
        with st.form("login_form"):
            device_input = st.text_input("Serial Number:", placeholder="เช่น TERRA0001")
            submit_login = st.form_submit_button("เข้าสู่ระบบ", use_container_width=True)
            
            if submit_login:
                if device_input:
                    device_id_upper = device_input.strip().upper()
                    doc_ref = db.collection('devices').document(device_id_upper).get()
                    if doc_ref.exists:
                        st.session_state.logged_in = True
                        st.session_state.current_device = device_id_upper
                        st.rerun() 
                    else:
                        st.error("❌ ไม่พบรหัสเครื่องนี้ในระบบ")
                else:
                    st.warning("⚠️ กรุณากรอกรหัสเครื่อง")

# ==========================================
# 🌾 6. Dashboard View
# ==========================================
else:
    # แสดง Title จำลองที่มุมขวา
    st.markdown("<p style='text-align: right; color: #888;'>TERRA - dashboard</p>", unsafe_html=True)
    
    device_id = st.session_state.current_device
    
    with st.sidebar:
        st.success(f"🟢 เครื่อง: **{device_id}**")
        st.divider()
        if st.button("🚪 ออกจากระบบ", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_device = None
            st.rerun()

    st.title("🌱 TERRA Dashboard")
    st.markdown("ระบบวิเคราะห์ดินและแนะนำการใส่ปุ๋ยอัจฉริยะ")

    sensor_data = get_sensor_latest(device_id)
    if sensor_data:
        st.subheader("📊 ข้อมูลล่าสุดจากเซนเซอร์")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Nitrogen (N)", f"{sensor_data['N']}")
        m2.metric("Phosphorus (P)", f"{sensor_data['P']}")
        m3.metric("Potassium (K)", f"{sensor_data['K']}")
        m4.metric("ค่า pH (ดิน)", f"{sensor_data['pH']}")
        m5.metric("ความชื้น", f"{sensor_data['Moist']}%")
        
        st.divider()
        
        st.subheader("⚙️ ตั้งค่าการประมวลผล")
        c1, c2 = st.columns(2)
        with c1:
            stage_name = st.selectbox("ระยะการเจริญเติบโต:", ["ฟื้นต้น", "สะสมอาหาร", "ขยายผล", "ก่อนเก็บเกี่ยว"])
        with c2:
            yield_target = st.number_input("เป้าหมายผลผลิต (กก./ต้น):", min_value=1, value=100)

        if st.button("🚀 เริ่มวิเคราะห์แผนการใส่ปุ๋ย", use_container_width=True):
            current_ph = sensor_data['pH']
            stage_map = {"ฟื้นต้น": 1, "สะสมอาหาร": 2, "ขยายผล": 3, "ก่อนเก็บเกี่ยว": 4}

            if stage_name == "ฟื้นต้น" and (current_ph < 5.5 or current_ph > 7.0):
                st.error(f"⚠️ ค่า pH {current_ph} ไม่อยู่ในเกณฑ์ที่เหมาะสม")
                st.info("แนะนำให้ปรับสภาพดินด้วยโดโลไมต์หรือกำมะถันก่อนใส่ปุ๋ย")
            else:
                input_df = pd.DataFrame([[
                    sensor_data['N'], sensor_data['P'], sensor_data['K'],
                    sensor_data['pH'], sensor_data['Moist'], 
                    stage_map[stage_name], yield_target
                ]], columns=['N_soil', 'P_soil', 'K_soil', 'pH', 'Moisture', 'Stage', 'Target_Yield_kg'])

                action_result = clf.predict(input_df)[0]
                nums_result = reg.predict(input_df)[0]

                st.success(f"### 💡 ผลวิเคราะห์: \n {action_result}")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.info(f"**N (กรัม)**: {max(0, nums_result[1]):.1f}")
                res_col2.info(f"**P (กรัม)**: {max(0, nums_result[2]):.1f}")
                res_col3.info(f"**K (กรัม)**: {max(0, nums_result[3]):.1f}")

    st.divider()
    st.caption("Project Terra | Engineering CMU 2026")