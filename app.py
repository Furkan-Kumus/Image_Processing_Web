from flask import Flask, render_template, request, url_for
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'  # Yüklenen görüntülerin klasörü
PROCESSED_FOLDER = 'static/processed'  # İşlenmiş görüntülerin klasörü
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Eğer klasörler yoksa oluştur
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_image_path = None
    processed_image_path = None

    if request.method == 'POST':
        # Dosya kontrolü
        if 'file' not in request.files or request.files['file'].filename == '':
            return "Lütfen bir dosya seçin ve işlem türünü belirtin!"

        file = request.files['file']
        operation = request.form.get('operation')
        if file and operation:
            # Yüklenen görüntüyü kaydet
            filename = secure_filename(file.filename)
            uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(uploaded_path)
            uploaded_image_path = uploaded_path.replace('\\', '/')
            uploaded_image_path = uploaded_image_path.replace('static/', '')

            # Görüntüyü işleyip işlenmiş görüntüyü kaydet
            processed_image_path = process_image(uploaded_path, filename, operation)
            processed_image_path = ("processed" + '/' + processed_image_path.split('static/processed\\')[1] )

    return render_template('index.html',
                           uploaded_image=uploaded_image_path,
                           processed_image=processed_image_path)



def process_image(image_path, filename, operation):
    img = cv2.imread(image_path)
    processed_img = None

    # İşlemler
    if operation == 'gray':
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif operation == 'otsu':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif operation == 'border_constant':
        processed_img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 255, 0))
    elif operation == 'border_replicate':
        processed_img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    elif operation == 'gamma_correction':
        gamma = 2.0
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        processed_img = cv2.LUT(img, table)
    elif operation == 'histogram':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        processed_img = cv2.equalizeHist(gray)
    elif operation == 'hist_equalize':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.equalizeHist(gray)
    elif operation == 'l2_gradient':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.Laplacian(gray, cv2.CV_64F)
    elif operation == 'deriche':
        processed_img = cv2.GaussianBlur(img, (5, 5), 1)
    elif operation == 'harris':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        img[dst > 0.01 * dst.max()] = [0, 0, 255]
        processed_img = img
    elif operation == 'face_cascade':
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        processed_img = img
    elif operation == 'contour':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        processed_img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    elif operation == 'morphology':
        kernel = np.ones((5, 5), np.uint8)
        processed_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif operation == 'blur':
        processed_img = cv2.GaussianBlur(img, (15, 15), 0)
    elif operation == 'box_filter':
        processed_img = cv2.boxFilter(img, -1, (5, 5))
    elif operation == 'median_filter':
        processed_img = cv2.medianBlur(img, 5)
    elif operation == 'bilateral_filter':
        processed_img = cv2.bilateralFilter(img, 9, 75, 75)
    elif operation == 'gaussian_filter':
        processed_img = cv2.GaussianBlur(img, (5, 5), 0)
    elif operation == '2d_filter':
        kernel = np.ones((5, 5), np.float32) / 25
        processed_img = cv2.filter2D(img, -1, kernel)

    processed_filename = f"processed_{operation}_{filename}"
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, processed_img)
    return processed_path


if __name__ == '__main__':
    app.run(debug=True)
