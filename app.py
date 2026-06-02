import os
import base64
import numpy as np
import cv2
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Set max upload size to 64MB
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

# Path to ONNX model
ONNX_PATH = os.path.join("ckpts", "DAv2S-4.onnx")

print("Initializing ONNX Runtime session...")
try:
    # Use CUDA execution provider (for GPU acceleration) if available, otherwise fall back to CPU
    session = ort.InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider'])
    print("ONNX Session loaded successfully.")
except Exception as e:
    print(f"Error loading ONNX session: {e}")
    session = None

def preprocess_image(img_bgr, target_size=(480, 640)):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize to exact model shape to ensure shape consistency
    img_resized = cv2.resize(img_rgb, (target_size[1], target_size[0]))
    img_float = img_resized.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - mean) / std
    
    # Transpose to BCHW
    img_tensor = img_norm.transpose(2, 0, 1)
    img_batch = np.expand_dims(img_tensor, axis=0)
    return img_batch

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if session is None:
        return jsonify({"success": False, "error": "ONNX model session not initialized. Please check server logs."}), 500

    left_file = request.files.get('left')
    right_file = request.files.get('right')

    if not left_file or not right_file:
        return jsonify({"success": False, "error": "Both Left (L) and Right (R) images are required."}), 400

    try:
        # Read uploaded files into OpenCV images
        left_bytes = left_file.read()
        right_bytes = right_file.read()

        nparr_l = np.frombuffer(left_bytes, np.uint8)
        nparr_r = np.frombuffer(right_bytes, np.uint8)

        img_l = cv2.imdecode(nparr_l, cv2.IMREAD_COLOR)
        img_r = cv2.imdecode(nparr_r, cv2.IMREAD_COLOR)

        if img_l is None or img_r is None:
            return jsonify({"success": False, "error": "Failed to decode one or both of the uploaded images. Please upload valid image files."}), 400

        orig_h, orig_w = img_l.shape[:2]

        # Preprocess images
        tensor_l = preprocess_image(img_l, target_size=(480, 640))
        tensor_r = preprocess_image(img_r, target_size=(480, 640))

        # Run inference
        outputs = session.run(['disp_pred'], {
            'img1': tensor_l,
            'img2': tensor_r
        })
        
        disp = outputs[0][0]  # Shape: (480, 640)

        # Resize disparity map back to original Left image size
        disp_orig = cv2.resize(disp, (orig_w, orig_h))

        # Normalize disparity map for visualization
        min_val = disp_orig.min()
        max_val = disp_orig.max()
        disp_norm = ((disp_orig - min_val) / (max_val - min_val + 1e-10) * 255.0).astype(np.uint8)

        # Apply TURBO colormap
        disp_colored = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)

        # Concatenate Left image and Colormapped depth map side-by-side
        concat_img = np.concatenate([img_l, disp_colored], axis=1)

        # Encode outputs to base64
        _, buffer_depth = cv2.imencode('.jpg', disp_colored)
        depth_base64 = base64.b64encode(buffer_depth).decode('utf-8')

        _, buffer_concat = cv2.imencode('.jpg', concat_img)
        concat_base64 = base64.b64encode(buffer_concat).decode('utf-8')
        
        # Encode original normalized disparity (grayscale) as base64 for 3D pointcloud generation
        _, buffer_disp = cv2.imencode('.png', disp_norm)
        disp_base64 = base64.b64encode(buffer_disp).decode('utf-8')

        # Also encode original left image as base64 for interactive comparison slider
        _, buffer_left = cv2.imencode('.jpg', img_l)
        left_base64 = base64.b64encode(buffer_left).decode('utf-8')

        return jsonify({
            "success": True,
            "depth_map": f"data:image/jpeg;base64,{depth_base64}",
            "concat": f"data:image/jpeg;base64,{concat_base64}",
            "left_image": f"data:image/jpeg;base64,{left_base64}",
            "raw_depth": f"data:image/png;base64,{disp_base64}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"An error occurred during inference: {str(e)}"}), 500

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
