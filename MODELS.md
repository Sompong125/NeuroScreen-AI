# 🧠 Model Files & Download Instructions

เนื่องจากไฟล์โมเดล Machine Learning (`.keras`) มีขนาดใหญ่ จึงถูกแยกเก็บออกจาก Git Repository หลักเพื่อรักษาขนาดของ Repository ให้เล็กและรวดเร็ว 

คุณสามารถดาวน์โหลดไฟล์โมเดลที่ผ่านการตรวจสอบและคัดเลือกแล้วได้จาก **GitHub Releases** อย่างเป็นทางการของโปรเจกต์นี้

## 📥 วิธีดาวน์โหลดและติดตั้งโมเดล (สำหรับผู้ใช้ทั่วไป)

1. ไปที่หน้า [Releases](https://github.com/Sompong125/NeuroScreen-AI/releases) ของ Repository นี้
2. คลิกที่ Release ล่าสุด (เช่น `v1.0.0`)
3. ในส่วน "Assets" ให้ดาวน์โหลดไฟล์ `day8_02_improved_model.keras`
4. นำไฟล์ที่ดาวน์โหลดมา วางไว้ในโฟลเดอร์ต่อไปนี้ภายในโปรเจกต์ของคุณ:
   ```text
   src/day8_Model/day8_02/day8_02_improved_model.keras
5. รันแอปพลิเคชัน Streamlit ตามปกติ ระบบจะโหลดโมเดลนี้โดยอัตโนมัติและสถานะ Pipeline จะเปลี่ยนเป็น "ready" (& $PYTHON -m streamlit run app.py)

## ℹ️ หมายเหตุเกี่ยวกับโมเดลอื่นๆ
ในโค้ดต้นฉบับอาจมีการอ้างอิงถึงไฟล์โมเดลอื่นๆ เช่น:
day7_04_best_model.keras (โมเดล Baseline)
day8_03_resnet18_best_model.keras (โมเดล ResNet-18 สำหรับการเปรียบเทียบ)
day9_01_best_model_seed*.keras (โมเดลจากการทดสอบ Robustness)
ไฟล์เหล่านี้เป็น artifacts สำหรับการทดลองและเปรียบเทียบภายในทีมพัฒนาเท่านั้น ไม่จำเป็น สำหรับการรันแอปพลิเคชันคัดกรองทั่วไป หากนักวิจัยท่านใดต้องการไฟล์เหล่านี้เพื่อวัตถุประสงค์ในการศึกษาเพิ่มเติม โปรดติดต่อผู้พัฒนาหรือตรวจสอบในหน้า Releases เวอร์ชัน Full Archive (ถ้ามี)

---

### 📄 2. ไฟล์ `README.md` (ฉบับอัปเดต)
*สร้างหรือทับไฟล์ `README.md` ในโฟลเดอร์หลัก (เพิ่มขั้นตอนการดาวน์โหลดโมเดลเข้าไป)*

```markdown
# 🧠 NeuroScreen AI

EEG-based ADHD Screening Support — Research Prototype

## ⚠️ Disclaimer (ข้อจำกัดความรับผิดชอบ)
NeuroScreen AI เป็น **ต้นแบบสำหรับช่วยคัดกรองในงานวิจัยเท่านั้น (Research Prototype)**
- ระบบนี้ **ไม่ใช่** เครื่องมือวินิจฉัยทางการแพทย์
- ผลลัพธ์ **ห้าม** ใช้วินิจฉัย ADHD หรือภาวะทางคลินิกใดๆ
- ผลลัพธ์ **ห้าม** ใช้แทนการประเมินโดยผู้เชี่ยวชาญ
- ผลที่แสดงเป็นผลลัพธ์จากโมเดล ไม่ใช่การยืนยันว่าบุคคลนั้นเป็นหรือไม่เป็น ADHD

## 📖 Overview
ระบบนี้ทดลองว่าโมเดล Deep Learning บนสัญญาณ EEG สามารถช่วยคัดกรองความเสี่ยงในงานวิจัยได้หรือไม่ โดยรับไฟล์ EEG 19 ช่อง (128 Hz) ผ่าน Preprocessing เดียวกับตอนฝึกโมเดล และแสดงความน่าจะเป็นระดับ Participant โดยใช้ Threshold คงที่ที่ 0.5

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/Sompong125/NeuroScreen-AI.git
cd NeuroScreen-AI

2. ติดตั้ง Dependencies
แนะนำให้สร้าง Virtual Environment ก่อน:
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt