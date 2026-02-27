import streamlit as st
import pandas as pd
import joblib
import firebase_admin
from firebase_admin import credentials, firestore

# ==========================================
# ⚙️ 1. Page Configuration (ต้องอยู่บรรทัดแรกสุด)
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# กำหนดชื่อ Title ตามสถานะการ Login
page_title = "TERRA - dashboard" if st.session_state.logged_in else "TERRA - home"

st.set_page_config(
    page_title=page_title,
    page_icon="🌱",
    layout="wide"
)

# ==========================================
# 🔥 2. Firebase & AI Model Connection
# ==========================================
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        # ดึงข้อมูลกุญแจจาก Streamlit Secrets
        try:
            key_dict = st.secrets["firebase_key"]
            cred = credentials.Certificate(dict(key_dict))
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Firebase Secret Error: {e}")
    return firestore.client()

@st.cache_resource
def load_terra_model():
    # โหลดไฟล์โมเดล AI (ต้องมีไฟล์ terra_model.pkl อยู่ในโฟลเดอร์เดียวกับโค้ด)
    return joblib.load("terra_model.pkl")

# เรียกใช้งาน Backend
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
        # ดึงข้อมูลล่าสุดโดยอิงจากชื่อ Document (ID) เรียงจากมากไปน้อย
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
        st.error(f"ไม่สามารถดึงข้อมูลจาก Firebase ได้: {e}")
    return None

# ==========================================
# 🚪 4. Logic: Login Page vs Dashboard
# ==========================================
if 'current_device' not in st.session_state:
    st.session_state.current_device = None

if not st.session_state.logged_in:
    # --- หน้า LOGIN (TERRA - home) ---
    st.write("") # เว้นระยะด้านบน
    st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🌱 เข้าสู่ระบบ TERRA (WEFARM)</h1>", unsafe_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem;'>กรุณากรอกรหัสเครื่องเซนเซอร์ (Serial Number)</p>", unsafe_html=True)
    
    # จัดตำแหน่ง Form ให้อยู่กึ่งกลางหน้าจอ
    _, col_mid, _ = st.columns([1, 1.5, 1])
    
    with col_mid:
        with st.form("login_form"):
            device_input = st.text_input("Serial Number:", placeholder="ระบุรหัสเครื่อง เช่น TERRA0001")
            submit_login = st.form_submit_button("เข้าสู่ระบบ", use_container_width=True)
            
            if submit_login:
                if device_input:
                    device_id_upper = device_input.strip().upper()
                    # ตรวจสอบว่ามี Document ID นี้ใน Collection 'devices' หรือไม่
                    doc_ref = db.collection('devices').document(device_id_upper).get()
                    
                    if doc_ref.exists:
                        st.session_state.logged_in = True
                        st.session_state.current_device = device_id_upper
                        st.rerun() 
                    else:
                        st.error("❌ ไม่พบรหัสเครื่องนี้ในระบบ กรุณาตรวจสอบอีกครั้ง")
                else:
                    st.warning("⚠️ กรุณากรอกรหัสเครื่องก่อนกดปุ่ม")

else:
    # --- หน้า DASHBOARD (TERRA - dashboard) ---
    device_id = st.session_state.current_device
    
    # Sidebar
    with st.sidebar:
        st.success(f"🟢 เชื่อมต่อ: **{device_id}**")
        st.divider()
        if st.button("🚪 ออกจากระบบ", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_device = None
            st.rerun()

    # Main Content
    st.title("🌾 TERRA Dashboard")
    st.markdown("ระบบวิเคราะห์ดินและแนะนำการใส่ปุ๋ยอัจฉริยะสำหรับสวนลำไย")

    sensor_data = get_sensor_latest(device_id)

    if sensor_data:
        # แสดงค่าจากเซนเซอร์ล่าสุด
        st.subheader("📊 ข้อมูลปัจจุบันจากเซนเซอร์")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Nitrogen (N)", f"{sensor_data['N']}")
        m2.metric("Phosphorus (P)", f"{sensor_data['P']}")
        m3.metric("Potassium (K)", f"{sensor_data['K']}")
        m4.metric("ค่า pH ดิน", f"{sensor_data['pH']}")
        m5.metric("ความชื้น", f"{sensor_data['Moist']}%")
        
        with st.expander("🔍 ดูค่าทางกายภาพเพิ่มเติม"):
            col_extra1, col_extra2 = st.columns(2)
            col_extra1.write(f"🌡️ อุณหภูมิดิน: {sensor_data['temp']} °C")
            col_extra2.write(f"⚡ ค่าการนำไฟฟ้า (EC): {sensor_data['cond']}")

        st.divider()

        # ส่วนรับค่าจากผู้ใช้
        st.subheader("⚙️ การตั้งค่าประมวลผล AI")
        col_input1, col_input2 = st.columns(2)
        with col_input1:
            stage_name = st.selectbox(
                "ระยะการเจริญเติบโตของลำไย:",
                ["ฟื้นต้น", "สะสมอาหาร", "ขยายผล", "ก่อนเก็บเกี่ยว"]
            )
        with col_input2:
            yield_target = st.number_input("เป้าหมายผลผลิต (กก./ต้น):", min_value=1, value=100)

        # AI Logic
        if st.button("🚀 เริ่มวิเคราะห์แผนการใส่ปุ๋ย", use_container_width=True):
            current_ph = sensor_data['pH']
            stage_map = {"ฟื้นต้น": 1, "สะสมอาหาร": 2, "ขยายผล": 3, "ก่อนเก็บเกี่ยว": 4}

            # ตรวจสอบค่า pH ในระยะฟื้นต้น
            if stage_name == "ฟื้นต้น" and (current_ph < 5.5 or current_ph > 7.0):
                st.error(f"⚠️ ตรวจพบค่า pH {current_ph} ซึ่งไม่เหมาะสม (ควรอยู่ระหว่าง 5.5 - 7.0)")
                st.warning("💡 **คำแนะนำ:** กรุณาปรับสภาพดินก่อนการใส่ปุ๋ยเคมี")
                if current_ph > 7.0:
                    st.info("🛠 **วิธีแก้ดินด่าง:** ใช้ผงกำมะถัน และเติมปุ๋ยอินทรีย์")
                else:
                    st.info("🛠 **วิธีแก้ดินกรด:** หว่านปูนโดโลไมต์ หรือปูนขาว แล้วพักดิน 14-20 วัน")
            else:
                # เตรียมข้อมูลสำหรับ Model
                input_df = pd.DataFrame([[
                    sensor_data['N'], sensor_data['P'], sensor_data['K'],
                    sensor_data['pH'], sensor_data['Moist'], 
                    stage_map[stage_name], yield_target
                ]], columns=['N_soil', 'P_soil', 'K_soil', 'pH', 'Moisture', 'Stage', 'Target_Yield_kg'])

                # ทำนายผล
                action_result = clf.predict(input_df)[0]
                nums_result = reg.predict(input_df)[0] # [Lime, N, P, K]

                # กรองค่าติดลบ
                n_val = max(0, nums_result[1])
                p_val = max(0, nums_result[2])
                k_val = max(0, nums_result[3])

                st.success(f"### 💡 ข้อแนะนำจาก AI: \n {action_result}")
                
                st.markdown("#### 🧪 ปริมาณธาตุอาหารที่ต้องเติม (กรัมต่อต้น):")
                res_c1, res_c2, res_c3 = st.columns(3)
                res_c1.info(f"**ไนโตรเจน (N)**\n{n_val:.1f} g")
                res_c2.info(f"**ฟอสฟอรัส (P)**\n{p_val:.1f} g")
                res_c3.info(f"**โพแทสเซียม (K)**\n{k_val:.1f} g")

    else:
        st.error(f"❌ ไม่พบข้อมูลจากเซนเซอร์ของเครื่อง {device_id} ในระบบ Cloud")

    st.divider()
    st.caption("Project Terra | Faculty of Engineering, Chiang Mai University 2026")