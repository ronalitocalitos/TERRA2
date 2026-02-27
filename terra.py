import streamlit as st
import pandas as pd
import joblib
import firebase_admin
from firebase_admin import credentials, firestore

# --- 1. การเชื่อมต่อ Firebase (Backend - Cloud Connection) ---
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        # ดึงข้อมูลกุญแจจาก Streamlit Secrets
        key_dict = st.secrets["firebase_key"]
        cred = credentials.Certificate(dict(key_dict))
        firebase_admin.initialize_app(cred)
    return firestore.client()

# --- 2. การโหลดโมเดล AI (AI Brain Loading) ---
@st.cache_resource
def load_terra_model():
    # ตรวจสอบว่าไฟล์ชื่อตรงกับบน GitHub (terra_model.pkl)
    return joblib.load("terra_model.pkl")

# เรียกใช้งานฟังก์ชันพื้นฐาน
db = init_firebase()
model_data = load_terra_model()
clf = model_data['classifier']
reg = model_data['regressor']

# --- 3. ฟังก์ชันดึงข้อมูลล่าสุด ---
def get_sensor_latest(device_id):
    try:
        query = db.collection('devices').document(device_id).collection('soilData')
        docs = query.order_by("__name__", direction=firestore.Query.DESCENDING).limit(1).get()
        
        for doc in docs:
            data = doc.to_dict()
            return {
                'N': data.get('N', 0),
                'P': data.get('P', 0),
                'K': data.get('K', 0),
                'pH': data.get('pH', 0),
                'Moist': data.get('moisture', 0), 
                'temp': data.get('temperature', 0),
                'cond': data.get('conductivity', 0)
            }
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")
    return None

# --- 4. ระบบจัดการ Session ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_device' not in st.session_state:
    st.session_state.current_device = None

# ==========================================
# 🚪 หน้า Login (TERRA - home)
# ==========================================
if not st.session_state.logged_in:
    # เปลี่ยน Page Title เป็น home
    st.set_page_config(page_title="TERRA - home", page_icon="🌱", layout="wide")

    # จัดข้อความให้อยู่กึ่งกลางด้วย HTML
    st.markdown("<h1 style='text-align: center;'>เข้าสู่ระบบ TERRA (WEFARM)</h1>", unsafe_html=True)
    st.markdown("<p style='text-align: center;'>กรุณากรอกรหัสเครื่องเซนเซอร์ (Serial Number) เพื่อเข้าดูข้อมูลแปลงเกษตรของคุณ</p>", unsafe_html=True)
    
    # สร้างคอลัมน์เพื่อบีบ Form ให้อยู่ตรงกลางหน้าจอ
    _, col_mid, _ = st.columns([1, 2, 1])
    
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
                        st.error("❌ ไม่พบรหัสเครื่องนี้ในระบบ กรุณาตรวจสอบความถูกต้อง")
                else:
                    st.warning("⚠️ กรุณากรอกรหัสเครื่อง")

# ==========================================
# 🌾 หน้าหลัก (TERRA - dashboard)
# ==========================================
else:
    # เปลี่ยน Page Title เป็น dashboard
    st.set_page_config(page_title="TERRA - dashboard", page_icon="🌱", layout="wide")
    
    device_id = st.session_state.current_device
    
    # --- Sidebar ---
    with st.sidebar:
        st.success(f"🟢 กำลังเชื่อมต่อกับเครื่อง:\n**{device_id}**")
        st.divider()
        if st.button("🚪 ออกจากระบบ", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_device = None
            st.rerun()

    # --- Main Dashboard ---
    st.title("TERRA")
    st.markdown("วิเคราะห์ธาตุอาหารในดินและแนะนำการใส่ปุ๋ยด้วย AI โดยกลุ่ม WEFARM")

    sensor_data = get_sensor_latest(device_id)

    if sensor_data:
        st.subheader("ข้อมูลล่าสุดจากเซนเซอร์ (The most recent data)")
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
            stage_name = st.selectbox(
                "ระยะการเจริญเติบโตของลำไย:",
                ["ฟื้นต้น", "สะสมอาหาร", "ขยายผล", "ก่อนเก็บเกี่ยว"]
            )
        with col_input2:
            yield_target = st.number_input("เป้าหมายผลผลิตที่ต้องการ (กก./ต้น):", min_value=1, value=100)

        # --- AI Processing ---
        if st.button("เริ่มวิเคราะห์แผนการใส่ปุ๋ย", use_container_width=True):
            current_ph = sensor_data['pH']
            stage_map = {"ฟื้นต้น": 1, "สะสมอาหาร": 2, "ขยายผล": 3, "ก่อนเก็บเกี่ยว": 4}

            if stage_name == "ฟื้นต้น" and (current_ph < 5.5 or current_ph > 7.0):
                st.error(f"⚠️ ตรวจพบค่า pH {current_ph} (ไม่อยู่ในเกณฑ์มาตรฐาน 5.5 - 7.0)")
                st.warning("💡 **คำแนะนำ:** ในระยะฟื้นต้นหลังเก็บเกี่ยว สิ่งสำคัญที่สุดคือการ 'ปรับสภาพดิน' ก่อนเริ่มให้ปุ๋ยเคมี")
                
                if current_ph > 7.0:
                    st.info("🛠 **วิธีปรับสภาพดินด่าง:** เติมผงกำมะถัน และอินทรียวัตถุ")
                elif current_ph < 5.5:
                    st.info("🛠 **วิธีปรับสภาพดินกรด:** หว่านปูนโดโลไมต์ หรือ ปูนขาว")

                st.success("📌 **สิ่งที่ต้องทำถัดไป:** ปรับสภาพดินและวัดค่าใหม่อีกครั้ง")

            else:
                input_df = pd.DataFrame([[
                    sensor_data['N'], sensor_data['P'], sensor_data['K'],
                    sensor_data['pH'], sensor_data['Moist'], 
                    stage_map[stage_name], yield_target
                ]], columns=['N_soil', 'P_soil', 'K_soil', 'pH', 'Moisture', 'Stage', 'Target_Yield_kg'])

                action_result = clf.predict(input_df)[0]
                nums_result = reg.predict(input_df)[0]

                n_pred = max(0, nums_result[1])
                p_pred = max(0, nums_result[2])
                k_pred = max(0, nums_result[3])

                st.success(f"### 💡 ผลวิเคราะห์จาก AI: \n {action_result}")
                
                if current_ph < 5.5 or current_ph > 7.0:
                    st.warning(f"⚠️ ข้อควรระวัง: ค่า pH ปัจจุบันคือ {current_ph} ควรเติมปุ๋ยอินทรีย์เพื่อรักษาสมดุล")

                st.markdown("#### 🧪 ปริมาณที่ต้องเติมโดยประมาณ:")
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.info(f"**ไนโตรเจน (N)**\n{n_pred:.1f} กรัม")
                res_col2.info(f"**ฟอสฟอรัส (P)**\n{p_pred:.1f} กรัม")
                res_col3.info(f"**โพแทสเซียม (K)**\n{k_pred:.1f} กรัม")

    else:
        st.error(f"❌ ไม่พบข้อมูลเซนเซอร์ในเครื่อง {device_id}")

    st.divider()
    st.caption("Project Terra | Engineering, Chiang Mai University 2026")