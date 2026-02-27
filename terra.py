import streamlit as st
import pandas as pd
import joblib
import firebase_admin
from firebase_admin import credentials, firestore

# ==========================================
# 1. INITIAL CONFIGURATION (Must be first)
# ==========================================
st.set_page_config(
    page_title="TERRA - AI Fertilizer System",
    page_icon="🌱",
    layout="wide"
)

# ==========================================
# 2. SESSION STATE MANAGEMENT
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_device' not in st.session_state:
    st.session_state.current_device = None

# ==========================================
# 3. BACKEND & AI MODEL (Cached)
# ==========================================
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        try:
            key_dict = st.secrets["firebase_key"]
            cred = credentials.Certificate(dict(key_dict))
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error("Firebase connection failed. Please check secrets.")
            return None
    return firestore.client()

@st.cache_resource
def load_terra_model():
    # Make sure 'terra_model.pkl' exists in your repository
    return joblib.load("terra_model.pkl")

# Initialize Backend
db = init_firebase()
model_data = load_terra_model()
clf = model_data['classifier']
reg = model_data['regressor']

# ==========================================
# 4. DATA FUNCTIONS
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
    except:
        return None
    return None

# ==========================================
# 5. MAIN INTERFACE LOGIC
# ==========================================

if not st.session_state.logged_in:
    # --- VIEW: LOGIN PAGE (TERRA - home) ---
    st.markdown("<p style='text-align: right; color: gray; font-size: 0.8rem;'>TERRA - home</p>", unsafe_html=True)
    st.markdown("<br><br>", unsafe_html=True)
    
    # Title centered
    st.markdown("<h1 style='text-align: center; color: #2E7D32;'>เข้าสู่ระบบ TERRA (WEFARM)</h1>", unsafe_html=True)
    st.markdown("<p style='text-align: center;'>กรุณากรอกรหัสเครื่องเซนเซอร์ (Serial Number) เพื่อเข้าดูข้อมูลแปลงเกษตรของคุณ</p>", unsafe_html=True)
    
    _, col_mid, _ = st.columns([1, 1.5, 1])
    with col_mid:
        with st.form("login_form"):
            device_input = st.text_input("Serial Number:", placeholder="ระบุรหัสเครื่อง เช่น TERRA0001")
            submit_login = st.form_submit_button("เข้าสู่ระบบ", use_container_width=True)
            
            if submit_login:
                if device_input:
                    device_id_upper = device_input.strip().upper()
                    # Check Firebase for device existence
                    doc_ref = db.collection('devices').document(device_id_upper).get()
                    if doc_ref.exists:
                        st.session_state.logged_in = True
                        st.session_state.current_device = device_id_upper
                        st.rerun()
                    else:
                        st.error("❌ ไม่พบรหัสเครื่องนี้ในระบบ")
                else:
                    st.warning("⚠️ กรุณากรอกรหัสเครื่อง")

else:
    # --- VIEW: DASHBOARD (TERRA - dashboard) ---
    st.markdown("<p style='text-align: right; color: gray; font-size: 0.8rem;'>TERRA - dashboard</p>", unsafe_html=True)
    
    device_id = st.session_state.current_device
    
    # Sidebar
    with st.sidebar:
        st.success(f"🟢 กำลังเชื่อมต่อกับเครื่อง:\n**{device_id}**")
        st.divider()
        if st.button("🚪 ออกจากระบบ", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_device = None
            st.rerun()

    # Dashboard Content
    st.title("🌱 TERRA Dashboard")
    st.markdown("วิเคราะห์ธาตุอาหารในดินและแนะนำการใส่ปุ๋ยด้วย AI โดยกลุ่ม WEFARM")

    sensor_data = get_sensor_latest(device_id)

    if sensor_data:
        st.subheader("📊 ข้อมูลล่าสุดจากเซนเซอร์")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Nitrogen (N)", f"{sensor_data['N']}")
        m2.metric("Phosphorus (P)", f"{sensor_data['P']}")
        m3.metric("Potassium (K)", f"{sensor_data['K']}")
        m4.metric("ค่า pH (ดิน)", f"{sensor_data['pH']}")
        m5.metric("ความชื้น (Moisture)", f"{sensor_data['Moist']}%")
        
        with st.expander("ดูค่าเพิ่มเติม"):
            st.write(f"อุณหภูมิดิน: {sensor_data['temp']} °C")
            st.write(f"ค่าการนำไฟฟ้า (Conductivity): {sensor_data['cond']}")

        st.divider()

        st.subheader("⚙️ ตั้งค่าการประมวลผล (User Input)")
        col_input1, col_input2 = st.columns(2)
        with col_input1:
            stage_name = st.selectbox("ระยะการเจริญเติบโตของลำไย:", ["ฟื้นต้น", "สะสมอาหาร", "ขยายผล", "ก่อนเก็บเกี่ยว"])
        with col_input2:
            yield_target = st.number_input("เป้าหมายผลผลิตที่ต้องการ (กก./ต้น):", min_value=1, value=100)

        # AI Analysis
        if st.button("🚀 เริ่มวิเคราะห์แผนการใส่ปุ๋ย", use_container_width=True):
            current_ph = sensor_data['pH']
            stage_map = {"ฟื้นต้น": 1, "สะสมอาหาร": 2, "ขยายผล": 3, "ก่อนเก็บเกี่ยว": 4}

            if stage_name == "ฟื้นต้น" and (current_ph < 5.5 or current_ph > 7.0):
                st.error(f"⚠️ ตรวจพบค่า pH {current_ph} (ไม่อยู่ในเกณฑ์มาตรฐาน 5.5 - 7.0)")
                st.warning("💡 **คำแนะนำ:** กรุณาปรับสภาพดินก่อนเริ่มให้ปุ๋ยเคมี")
            else:
                # Prepare data for model
                input_df = pd.DataFrame([[
                    sensor_data['N'], sensor_data['P'], sensor_data['K'],
                    sensor_data['pH'], sensor_data['Moist'], 
                    stage_map[stage_name], yield_target
                ]], columns=['N_soil', 'P_soil', 'K_soil', 'pH', 'Moisture', 'Stage', 'Target_Yield_kg'])

                action_result = clf.predict(input_df)[0]
                nums_result = reg.predict(input_df)[0] # [Lime, N, P, K]

                st.success(f"### 💡 ผลวิเคราะห์จาก AI: \n {action_result}")
                
                st.markdown("#### 🧪 ปริมาณธาตุอาหารที่แนะนำ (กรัมต่อต้น):")
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.info(f"**ไนโตรเจน (N)**\n{max(0, nums_result[1]):.1f} กรัม")
                res_col2.info(f"**ฟอสฟอรัส (P)**\n{max(0, nums_result[2]):.1f} กรัม")
                res_col3.info(f"**โพแทสเซียม (K)**\n{max(0, nums_result[3]):.1f} กรัม")
    else:
        st.error(f"❌ ไม่พบข้อมูลเซนเซอร์ในระบบ Cloud ของเครื่อง {device_id}")

    st.divider()
    st.caption("Project Terra | Engineering, Chiang Mai University 2026")