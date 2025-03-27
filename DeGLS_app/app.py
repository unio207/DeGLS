import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance
import numpy as np
from yolo_inference import YOLOSegmenter
from lesion_segmentation import GAUNetSegmenter
import time  
from utils.disease_severity import calculate_disease_severity

app = Flask(__name__)

# Configuring the upload and processed folders
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to overlay lesion mask on the segmented leaf
def overlay_mask_on_image(original_image_path, lesion_mask_path, save_path):
    original = Image.open(original_image_path).convert("RGB")
    mask_img = Image.open(lesion_mask_path).convert("L")
    mask_img = mask_img.resize(original.size, Image.NEAREST)

    enhancer = ImageEnhance.Brightness(original)
    darkened = enhancer.enhance(0.9)

    mask_array = np.array(mask_img)
    colored_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
    colored_mask[:, :, 0] = mask_array

    alpha_mask = (mask_array > 0).astype(np.uint8) * 128

    darkened_array = np.array(darkened)
    combined = Image.fromarray(darkened_array)
    overlay = Image.fromarray(colored_mask)
    overlay.putalpha(Image.fromarray(alpha_mask))

    combined.paste(overlay, (0, 0), overlay)
    combined.save(save_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"})

        if file and allowed_file(file.filename):

            start_time = time.time()  # Start time for performance measurement

            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Step 1: YOLO segmentation and classification
            yolo_segmenter = YOLOSegmenter(model_path='models/yolo.pt')
            segmented_leaf, disease_type = yolo_segmenter.segment_and_classify(filepath)

            # Save segmented leaf
            segmented_leaf_filename = f"segmented_{filename}"
            segmented_leaf_path = os.path.join(app.config['PROCESSED_FOLDER'], segmented_leaf_filename)
            yolo_segmenter.save_segmented_leaf(segmented_leaf, segmented_leaf_path)

            # Step 2: GAUNet lesion segmentation
            gaunet_segmenter = GAUNetSegmenter(model_path='models/gaunet.pth', device='cpu')
            lesion_mask = gaunet_segmenter.segment(segmented_leaf_path)

            # Save lesion mask
            lesion_mask_filename = f"lesion_mask_{filename}"
            lesion_mask_path = os.path.join(app.config['PROCESSED_FOLDER'], lesion_mask_filename)
            gaunet_segmenter.save_mask(lesion_mask, lesion_mask_path)

            # Overlay lesion mask on segmented leaf
            overlayed_image_filename = f"overlayed_image_{filename}"
            overlayed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], overlayed_image_filename)
            overlay_mask_on_image(segmented_leaf_path, lesion_mask_path, overlayed_image_path)

            end_time = time.time()  # End time for performance measurement
            processing_time = round(end_time - start_time, 2)  # Calculate processing time

            # Step 3: Calculate disease severity
            percent_severity = calculate_disease_severity(segmented_leaf_path, lesion_mask_path)

            # Grab form inputs from frontend
            corn_hybrid = request.form['corn_hybrid']
            location = request.form['location']
            date = request.form['date']


            # Return the results as JSON for AJAX
            return jsonify({
                'original_image': filename,
                'overlayed_image': overlayed_image_filename,
                'corn_hybrid': corn_hybrid,
                'location':location,
                'date':date,
                'disease_type': disease_type,
                'percent_severity': percent_severity,
                'processing_time': processing_time
            })

    # Render the index page for GET requests
    return render_template('index.html')

if __name__ == '__main__':
    # Ensure the upload and processed folders exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    # Run the Flask app
    app.run(debug=True)