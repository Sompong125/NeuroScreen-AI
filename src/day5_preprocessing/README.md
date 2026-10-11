# Day 5 — EEG Preprocessing (Experiment)

Day 5 คือการทดลองสร้างและตรวจสอบ **preprocessing pipeline** ของสัญญาณ EEG
ก่อนนำไปใช้กับโมเดล CNN ในขั้นต่อไป — **ยังไม่ใช่การ train model**
และแยกขาดจากเว็บหลัก (app.py / Streamlit dashboard) โดยสิ้นเชิง

## Pipeline

Raw EEG → Select 19 channels → Check NaN/Inf → 50 Hz Notch Filter
→ Standardization → Segment (4 s = 512 samples @ 128 Hz) → Spectrogram

## ไฟล์ในแต่ละขั้น

- `day5_01_basic_check.py` — โหลด 50,000 rows ตรวจสอบ columns/channels/
  participant คนแรก/NaN/Inf/จำนวน samples ต้อง ≥ 512
- `day5_02_preprocessing_test.py` — ตัด segment 4 วินาที (512×19),
  กรอง 50 Hz ด้วย iirnotch(w0=50, Q=30, fs=128) + filtfilt,
  standardize ราย channel (mean≈0, std≈1), บันทึก CSV/PNG/TXT
- `day5_03_spectrogram_test.py` — โหลด CSV ที่ preprocessed แล้ว,
  สร้าง spectrogram ทั้ง 19 channels (nperseg=128, noverlap=64),
  แปลงเป็น dB, plot ถึง 60 Hz, บันทึก PNG/TXT

## วิธีรัน

เปิด Terminal ในโฟลเดอร์ `day5_preprocessing/` แล้วรันตามลำดับ:

python day5_01_basic_check.py
python day5_02_preprocessing_test.py
python day5_03_spectrogram_test.py

ต้องรันตามลำดับ — ถ้าขั้นใดพิมพ์ `FAILED` ให้หยุดแก้ก่อนจึงไปขั้นถัดไป

## ผลลัพธ์ (อยู่ในโฟลเดอร์ output/ เท่านั้น)

- day5_01_basic_check.txt
- day5_preprocessed_19channels_test.csv (512 rows)
- day5_preprocessed_19channels.png
- day5_02_preprocessing_summary.txt
- day5_spectrogram_19channels.png
- day5_03_spectrogram_summary.txt

## หมายเหตุความปลอดภัย

- โหลด dataset เพียง 50,000 rows (`nrows=50000`) ไม่โหลดทั้งหมด
- Sampling rate 128 Hz มาจาก dataset metadata (ไม่ใช่ column ใน CSV)
- ไม่แก้ไข `data/adhdata.csv` และไม่เขียนทับไฟล์นอกโฟลเดอร์ output/
- Day 5 เป็นการทดลอง preprocessing เท่านั้น ยังไม่มีการ train AI ใดๆ