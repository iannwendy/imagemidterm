# YÊU CẦU BÀI THI GIỮA KỲ - XỬ LÝ ẢNH SỐ

**MSSV:** 523H0054  
**Môn:** Introduction to Digital Image Processing (505060)  
**Học kỳ:** Học kỳ 1, Năm học 2026-2027

---

## 📋 TỔNG QUAN BÀI THI

### Phần lập trình (7.0 điểm)
Xây dựng chương trình Python phát hiện và vẽ bounding box hình chữ nhật xung quanh **biển báo giao thông hình tròn** trong 2 video đầu vào.

### Phần báo cáo (3.0 điểm)
Viết báo cáo theo đúng format của Khoa Công nghệ Thông tin.

---

## 🎯 PHẦN LẬP TRÌNH

### Input/Output
| Loại | Chi tiết |
|------|----------|
| **Input** | `video1.mp4`, `video2.mp4` |
| **Output** | Video mới có bounding box quanh biển báo hình tròn |
| **Định dạng output** | MP4, cùng color model, frame size, frame rate với input |
| **Sample output** | `video1_output_sample.mp4` |

### Yêu cầu kỹ thuật
1. **Thư viện cho phép:** Chỉ dùng OpenCV, math, Numpy
2. **Cấm:** Không dùng pre-trained models
3. **Nghiêm cấm:** Hardcode giải pháp cho từng frame cụ thể
4. **Bắt buộc:** Dùng **cùng một source code** cho cả 2 video
5. **Mỗi frame output** phải có Student ID (523H0054) được chèn vào

### Chiến lược phát hiện biển báo (theo Guideline)

#### 1. Color Filtering (HSV-based)
- Các màu chính trong biển báo: **ĐỎ, XANH DƯƠNG, VÀNG, TRẮNG**
- Chuyển đổi sang HSV color space
- Sử dụng `cv2.inRange()` để tạo mask

Ví dụ HSV thresholds:
```
lower_red = np.array([159, 50, 70])
upper_red = np.array([180, 255, 255])
lower_blue = np.array([105, 150, 70])
upper_blue = np.array([135, 255, 255])
```

#### 2. Tìm Contours
- Sử dụng `cv2.findContours()` trên mask đã tạo

#### 3. Kiểm tra hình dạng contour
- **Circularity check:** Kiểm tra contour có phải hình tròn không
  - Cách 1: Tính circularity ratio
  - Cách 2: Sử dụng `cv2.HoughCircles()`
- **Triangle/Rectangle:** Dùng `cv2.approxPolyDP()`

#### 4. Lọc contours không liên quan
- Kiểm tra: Area/Perimeter ratio
- Kiểm tra: Width & Height của bounding box
- Loại bỏ redundant detections

#### 5. Vẽ Bounding Box
- Vẽ rectangle xung quanh mỗi biển báo hình tròn được phát hiện

### Video Link
- Input videos: https://drive.google.com/drive/folders/15IMKKVqdRqsybV7-hVhwEgIMch1mKz1H?usp=sharing

---

## 📝 PHẦN BÁO CÁO

### Format bắt buộc
- **Định dạng:** PDF
- **Phải tuân theo format báo cáo của Khoa CNTT**
- **Sai format → 0 điểm phần báo cáo**

### Chương 1: Phương pháp giải quyết (2.5 điểm)
Mô tả chi tiết các bước lập trình:
- Giải thích từng bước chính trong thuật toán
- Cung cấp source code tương ứng với giải thích
- Cung cấp ví dụ output images/information

### Chương 2: Kết quả (0.5 điểm)
- Mỗi video: **05 output frames** có biển báo hình tròn ở 05 thời điểm khác nhau
- **Khoảng cách tối thiểu** giữa 2 frame liên tiếp: **3 giây**
- **Chia sẻ link Google Drive** (public access) của output videos
- Hình ảnh phải rõ ràng, có caption và mô tả

---

## 📤 NỘP BÀI

### Đặt tên file
| Thành phần | Tên file |
|------------|----------|
| Python source | `523H0054.py` |
| Report | `523H0054.pdf` |
| Output video 1 | `523H0054_video1.mp4` |
| Output video 2 | `523H0054_video2.mp4` |

### Nơi nộp
- Python source + Report → "MidTerm_Essay" assignment trên Elearning
- Output videos → Upload lên Google Drive (public link), paste link vào Chương 2

### Lưu ý quan trọng
- Tất cả thành viên đều phải nộp bài
- File Python phải chạy được, không lỗi
- Sai định dạng .py → 0 điểm

---

## 📊 RUBRIC CHẤM ĐIỂM

### Phần Lập trình (7.0 điểm)

| Tiêu chí | Điểm | Chi tiết |
|----------|------|----------|
| **Phát hiện đúng biển báo hình tròn** | 3.0 | Tổng cộng 24 biển báo trong 2 video → Mỗi biển báo = 3.0/24 điểm |
| **Detection ổn định và duy nhất** | 3.0 | Từ xa đến gần phải detection stable |
| **Redundant detections** | -0.25/_detection | Mỗi detection thừa cắt ngang biển báo |

**Cách tính điểm redundant detections:**
| Số redundant detections | Điểm nhận được |
|------------------------|----------------|
| > 10 | 0 điểm |
| 5 - 10 | 0.5 điểm |
| < 5 | 1.0 điểm |

### Phần Báo cáo (3.0 điểm)

| Tiêu chí | Điểm | Chi tiết |
|----------|------|----------|
| **Chương 1 - Main steps** | 2.5 | Xem bảng dưới |
| **Chương 2 - Kết quả** | 0.5 | -0.05/mỗi frame sai |

**Chi tiết Chương 1:**
| Mức độ đúng | Điểm |
|-------------|------|
| > 80% sai/unclear | 0 - 0.5 |
| 20% - 80% sai/unclear | 0.5 - 1.5 |
| < 20% sai/unclear | 1.5 - 2.5 |

---

## ⚠️ QUY ĐỊNH NGHIÊM KHẮC

1. **Nộp trễ** → 0.0 điểm
2. **Thiếu materials** → 0.0 điểm
3. **Không share link video** → 0.0 điểm
4. **Link không truy cập được** → 0.0 điểm
5. **Source code lỗi/sai format** → 0.0 điểm
6. **Copy code từ internet/bạn** → 0.0 điểm (tất cả các nhóm liên quan)
7. **Nghi ngờ copy** → Phải phỏng vấn với giảng viên

---

## ✅ CHECKLIST TRƯỚC KHI NỘP

- [ ] Python code dùng được OpenCV, Numpy, math
- [ ] Không dùng pre-trained models
- [ ] Không hardcode cho từng frame
- [ ] Cùng source code cho cả 2 video
- [ ] Output video cùng format với input (MP4, size, fps)
- [ ] Mỗi frame có Student ID 523H0054
- [ ] Bounding box chỉ bao quanh biển báo hình tròn (không phải tam giác/chữ nhật)
- [ ] Report đúng format Khoa CNTT
- [ ] Chương 1 có code và giải thích
- [ ] Chương 2 có đủ 05 frames/video, cách nhau ≥3 giây
- [ ] Đã upload video lên Google Drive và có public link
- [ ] Tên file đúng: 523H0054.py, 523H0054.pdf
