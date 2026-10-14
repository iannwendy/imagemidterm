# Guideline for Video Processing in Midterm and Final Exams

> Tài liệu được biên soạn lại từ **Guideline DIP.pdf** (3 trang). Nội dung gốc bằng tiếng Anh được giữ lại, với tiêu đề, danh sách và code được định dạng lại để dễ đọc. Đoạn code ví dụ trong tài liệu gốc kết thúc ở `cv2.findContours(...)` và không phải một chương trình hoàn chỉnh.

## 1. Sample Code: Video Processing with OpenCV

```python
# Importing libraries
import cv2
import numpy as np

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('video1.mp4')

# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video file")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
result = cv2.VideoWriter(
    'video1_output_sample.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),  # mp4v / avc1
    30,
    size
)
print("Resolution =", size)

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break

    # cv2.imshow('Frame', frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold of red/blue in HSV space
    lower_red = np.array([159, 50, 70])
    upper_red = np.array([180, 255, 255])
    lower_blue = np.array([105, 150, 70])
    upper_blue = np.array([135, 255, 255])

    # Preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours
    # Original PDF leaves this call unfinished:
    # contours, hierarchy = cv2.findContours(image=mask, ...)

    # ...
```

**Notes from the sample:**

- Reads the input video `video1.mp4` using `cv2.VideoCapture`.
- Gets the frame width and height, then initializes `cv2.VideoWriter` for `video1_output_sample.mp4` using the `mp4v` codec at 30 FPS; the source also mentions `avc1`.
- Converts each frame from BGR to HSV.
- Defines HSV ranges for red and blue, then creates a **red** mask using `cv2.inRange`.
- Introduces contour detection, but does not provide a complete `findContours` call or frame-writing/release logic.

## 2. Some Strategies for Detecting Traffic Signs

### 2.1. Color Filtering

1. Apply **HSV-based color filtering**. Main colors in a traffic sign include **red, blue, yellow, and white**.
2. Find contours using `cv2.findContours()`.
3. Check whether each contour belongs to a traffic sign based on the following characteristics.

#### A. Shape of the Contour

Possible shapes include **circle**, **rectangle**, and **triangle**.

##### Check Whether a Contour Is a Circle

The source suggests comparing the contour's area with the area of its minimum enclosing circle. It also mentions `cv2.isContourConvex()` and optional area/circularity filtering.

1. **Find the contour:** use `cv2.findContours()` to detect contours in the image.
2. **Calculate the minimum enclosing circle:** use `cv2.minEnclosingCircle()` to find the smallest circle that can enclose the contour.
3. **Compare areas:** calculate the contour area using `cv2.contourArea()`, calculate the minimum enclosing circle area (`π × radius²`), and compare the two areas. If the areas are similar, the contour is likely a circle.
4. **Check convexity (optional):** use `cv2.isContourConvex()` to determine whether the contour is convex. A circle is a convex shape.
5. **Filter by area and circularity (optional):** filter contours by size and circularity.

Circularity formula:

$$
\mathrm{Circularity} = \frac{4\pi\,\mathrm{Area}}{\mathrm{Perimeter}^{2}}
$$

A circle has circularity close to **1.0**.

**Alternative:** use `cv2.HoughCircles()`.

Reference: [Circle Detection Using OpenCV (GeeksforGeeks)](https://www.geeksforgeeks.org/python/circle-detection-using-opencv-python/)

##### Check Whether a Contour Is a Triangle or Rectangle

- Use `cv2.approxPolyDP()`.
- Refer to **Lab04: "Contour features"** section.

#### B. Area and Perimeter of the Contour

Inspect the contour's **area** and **perimeter** when deciding whether it represents a traffic sign.

#### C. Width and Height of the Bounding Box

Inspect the **width** and **height** of the bounding box.

**Additional guideline:** try to remove redundant contours that are not related to traffic signs.

### 2.2. Feature Matching

Feature matching is listed as another possible strategy for detecting traffic signs.

### 2.3. Template Matching

Template matching is also listed as a detection strategy.

## 3. Final Report: Infer the Traffic Sign Content

To infer the content of a detected traffic sign, the original guideline suggests combining:

1. **Color information:** inside the contour, calculate the ratio or count of **red, blue, yellow, and white pixels**.
2. **Contour shape:** determine whether the shape is a **circle**, **rectangle**, or **triangle**.
3. **Feature matching:** use a feature-matching approach.
4. **Template matching:** compare against traffic-sign templates.

Reference for feature matching: [Feature Matching Using Brute Force in OpenCV (GeeksforGeeks)](https://www.geeksforgeeks.org/feature-matching-using-brute-force-in-opencv/)

---

**Source:** *Guideline DIP.pdf*, pages 1–3. This Markdown reorganizes the source for readability and explicitly labels the example's unfinished portions; it does not supply missing implementation or assume additional exam requirements.
