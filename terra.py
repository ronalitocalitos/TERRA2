import streamlit as st
import pandas as pd
import joblib
import firebase_admin
from firebase_admin import credentials, firestore

# --- 1. การตั้งค่าหน้าเว็บ (Frontend Configuration) ---
st.set_page_config(
    page_title="TERRA - AI Fertilizer System",
    page_icon="🌱",
    layout="wide"
)

# --- 2. การเชื่อมต่อ Firebase (Backend - Cloud Connection) ---
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        # ดึงข้อมูลกุญแจจาก Streamlit Secrets
        key_dict = st.secrets["firebase_key"]
        cred = credentials.Certificate(dict(key_dict))
        firebase_admin.initialize_app(cred)
    return firestore.client()

# --- 3. การโหลดโมเดล AI (AI Brain Loading) ---
@st.cache_resource
def load_terra_model():
    # ตรวจสอบว่าไฟล์ชื่อตรงกับบน GitHub (terra_model.pkl)
    return joblib.load("terra_model.pkl")

# เรียกใช้งานฟังก์ชัน
db = init_firebase()
model_data = load_terra_model()
clf = model_data['classifier']
reg = model_data['regressor']

# --- 4. ฟังก์ชันดึงข้อมูลล่าสุดจาก Sub-collection (ตามรูปใหม่) ---
def get_sensor_latest():
    try:
        # Path ใหม่: devices -> TERRA0001 -> soilData
        query = db.collection('devices').document('TERRA0001').collection('soilData')
        
        # เรียงลำดับตามชื่อ Document (Timestamp) จากใหม่ไปเก่า แล้วเอาอันแรก
        docs = query.order_by("__name__", direction=firestore.Query.DESCENDING).limit(1).get()
        
        for doc in docs:
            data = doc.to_dict()
            # แมปชื่อ Field ให้ตรงตามรูปเป๊ะๆ (N, P, K, pH, moisture)
            return {
                'N': data.get('N', 0),
                'P': data.get('P', 0),
                'K': data.get('K', 0),
                'pH': data.get('pH', 0),
                'Moist': data.get('moisture', 0), # ในรูปเพื่อนใช้ชื่อ 'moisture'
                'temp': data.get('temperature', 0),
                'cond': data.get('conductivity', 0)
            }
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")
    return None

# --- 5. ส่วนแสดงผลหน้าเว็บ (UI Design) ---
st.title("🌱 TERRA: ระบบแนะนำปุ๋ยลำไยอัจฉริยะ")
st.markdown("วิเคราะห์ธาตุอาหารในดินและแนะนำการใส่ปุ๋ยด้วย AI โดยกลุ่ม Computer Engineering CMU")

sensor_data = get_sensor_latest()

if sensor_data:
    st.subheader("📡 ข้อมูลล่าสุดจากเซนเซอร์ (Real-time Cloud Data)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Nitrogen (N)", f"{sensor_data['N']}")
    m2.metric("Phosphorus (P)", f"{sensor_data['P']}")
    m3.metric("Potassium (K)", f"{sensor_data['K']}")
    m4.metric("ค่า pH (ดิน)", f"{sensor_data['pH']}")
    m5.metric("ความชื้น (Moisture)", f"{sensor_data['Moist']}%")
    
    # เพิ่มค่าเสริมที่เพื่อนส่งมา (Temp/Cond)
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

    if st.button("🚀 เริ่มวิเคราะห์แผนการใส่ปุ๋ย", use_container_width=True):
        stage_map = {"ฟื้นต้น": 1, "สะสมอาหาร": 2, "ขยายผล": 3, "ก่อนเก็บเกี่ยว": 4}
        
        # จัดเตรียมข้อมูล (ต้องเรียง Column ให้ตรงกับตอน Train)
        input_df = pd.DataFrame([[
            sensor_data['N'], sensor_data['P'], sensor_data['K'],
            sensor_data['pH'], sensor_data['Moist'], 
            stage_map[stage_name], yield_target
        ]], columns=['N_soil', 'P_soil', 'K_soil', 'pH', 'Moisture', 'Stage', 'Target_Yield_kg'])

        action_result = clf.predict(input_df)[0]
        nums_result = reg.predict(input_df)[0] # [Lime, N, P, K]

        st.success(f"### 💡 ผลวิเคราะห์จาก AI: \n {action_result}")
        
        st.markdown("#### 🧪 ปริมาณที่ต้องเติมโดยประมาณ:")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.info(f"**ไนโตรเจน (N)**\n{nums_result[1]:.1f} กรัม")
        res_col2.info(f"**ฟอสฟอรัส (P)**\n{nums_result[2]:.1f} กรัม")
        res_col3.info(f"**โพแทสเซียม (K)**\n{nums_result[3]:.1f} กรัม")
        res_col4.warning(f"**ปูนขาว (Lime)**\n{nums_result[0]:.2f} กิโลกรัม")
else:
    st.error("❌ ไม่พบข้อมูลเซนเซอร์ในระบบ Cloud (ตรวจสอบ Path: devices/TERRA0001/soilData)")

st.divider()
st.caption("Project Terra | Computer Engineering, Chiang Mai University 2026")