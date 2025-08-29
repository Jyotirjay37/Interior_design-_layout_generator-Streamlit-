# 🏠 Interior Design Generator

## Overview
This project is an interactive room interior design generator using computer vision and deep learning. It leverages YOLOv8 for object detection, Streamlit for the web interface, and Plotly for 3D visualization. Users can upload room images, detect furniture, visualize layouts, generate shopping lists, and preview designs in AR.

---

## ✨ Features & Insights

- **Furniture Detection:** Uses YOLOv8 (`yolov8n.pt`) to detect furniture in room images.
- **2D Layout Visualization:** Detected furniture is highlighted with bounding boxes and labels using OpenCV.
- **3D Room Visualization:** Interactive 3D layouts generated with Plotly, showing furniture placement and room dimensions.
- **Shopping List Generation:** (Planned) Generates a shopping list of detected furniture with product details and export options (PDF/CSV).
- **AR Preview:** (Planned) Allows users to preview furniture placement in AR, including QR code generation for mobile viewing.
- **PDF Export:** (Planned) Export shopping lists and layouts as PDF using fpdf2.
- **User Interface:** Built with Streamlit for easy interaction and visualization.

---

## 🖼️ Output Examples

- **Detected Furniture:**  
  ![Sample Detection](output/sample_detection.png)  
  *Bounding boxes and labels for detected furniture.*

- **3D Room Layout:**  
  ![3D Layout](output/sample_3d_layout.png)  
  *Interactive 3D visualization of room and furniture.*

- **Shopping List (Planned):**  
  | Item      | Brand   | Price   | Link      |
  |-----------|---------|---------|-----------|
  | Sofa      | IKEA    | $499    | ...       |
  | Table     | Wayfair | $199    | ...       |

---

## 📦 Datasets Used

- **YOLOv8 Pretrained Model:**  
  - File: [`yolov8n.pt`](yolov8n.pt)  
  - Source: [Ultralytics YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
  - Trained on COCO dataset for general object detection.

- **Kaggle Datasets:**  
  - [Furniture Detection Dataset](https://www.kaggle.com/datasets/ahmedkhanak1995/furniture-object-detection)  
    *Used for fine-tuning and testing furniture detection models.*
  - [Room Images Dataset](https://www.kaggle.com/datasets/mahmoudnafifi/indoor-scenes)  
    *Provides diverse indoor room images for evaluation and visualization.*

- **User Images:**  
  - Users upload their own room images for analysis.
  - No proprietary datasets included; all processing is done on user-provided images.

---

## ⚡ Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download YOLOv8 model:**
   - Automatically via `download_model.py` or manually from [Ultralytics YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt).

3. **Run the application:**
   ```bash
   streamlit run room.py
   ```

---

## 📁 File Structure

- `room.py` — Main application logic
- `requirements.txt` — Python dependencies
- `yolov8n.pt` — YOLOv8n model weights
- `download_model.py` — Script to download model
- `README.md` — Project documentation
- `TODO.md` — Development roadmap
- `output/` — Example output images (add your own)


---

## 🛠️ Troubleshooting

- Ensure `yolov8n.pt` is ~6.2MB and not corrupted.
- If model loading fails, re-download using provided scripts.
- See [`README_FIX.md`](README_FIX.md) for model loading error solutions.

---

## 📜 License

This project uses open-source libraries and pretrained models. See individual library licenses for
