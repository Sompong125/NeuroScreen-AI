# ============================================================
# สร้างไฟล์ทดสอบสำหรับ Error Handling ของ Day 10.3
# ข้อมูลสังเคราะห์ใช้ทดสอบโครงสร้างเท่านั้น
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
EEG_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]
rng = np.random.default_rng(42)

print("=" * 60)
print("สร้างไฟล์ทดสอบ Error Handling")
print("=" * 60)

# 1. ไฟล์ที่ขาดคอลัมน์ (มีแค่ 3 คอลัมน์)
df1 = pd.DataFrame(rng.standard_normal((100, 3)),
                   columns=["Fp1", "Fp2", "timestamp"])
df1.to_csv(OUT_DIR / "test_missing_columns.csv", index=False)
print("✅ test_missing_columns.csv  (ขาด 16 คอลัมน์)")

# 2. ไฟล์สั้นเกินไป (มีแค่ 100 samples < 512)
df2 = pd.DataFrame(rng.standard_normal((100, 19)), columns=EEG_CHANNELS)
df2.to_csv(OUT_DIR / "test_too_short.csv", index=False)
print("✅ test_too_short.csv  (100 samples < 512)")

# 3. ไฟล์ที่มี NaN
df3 = pd.DataFrame(rng.standard_normal((1024, 19)), columns=EEG_CHANNELS)
df3.iloc[10, 3] = np.nan
df3.iloc[50, 10] = np.nan
df3.to_csv(OUT_DIR / "test_has_nan.csv", index=False)
print("✅ test_has_nan.csv  (มี NaN 2 จุด)")

# 4. ไฟล์ที่มี Inf
df4 = pd.DataFrame(rng.standard_normal((1024, 19)), columns=EEG_CHANNELS)
df4.iloc[5, 1] = np.inf
df4.iloc[20, 8] = -np.inf
df4.to_csv(OUT_DIR / "test_has_inf.csv", index=False)
print("✅ test_has_inf.csv  (มี Inf 2 จุด)")

# 5. ไฟล์ว่างเปล่า
(OUT_DIR / "test_empty.csv").write_text("")
print("✅ test_empty.csv  (ไฟล์ว่าง)")

# 6. ไฟล์ที่มีข้อมูลไม่ใช่ตัวเลข (มีข้อความ) — เขียนด้วยมือ
lines = [",".join(EEG_CHANNELS)]  # header
for i in range(1024):
    row = rng.standard_normal(19).tolist()
    if i == 15:
        row[5] = "not_a_number"  # ใส่ข้อความในเซลล์
    lines.append(",".join(str(v) for v in row))
(OUT_DIR / "test_non_numeric.csv").write_text("\n".join(lines))
print("✅ test_non_numeric.csv  (มีข้อความในเซลล์)")

# 7. ไฟล์ที่ถูกต้อง (ใช้ทดสอบ pipeline จริง)
df7 = pd.DataFrame(rng.standard_normal((512 * 3, 19)), columns=EEG_CHANNELS)
df7.to_csv(OUT_DIR / "test_valid.csv", index=False)
print("✅ test_valid.csv  (ถูกต้อง 1536 samples = 3 เซกเมนต์)")

print("\n" + "=" * 60)
print(f"สร้างไฟล์ทดสอบ 7 ไฟล์ใน: {OUT_DIR}")
print("=" * 60)