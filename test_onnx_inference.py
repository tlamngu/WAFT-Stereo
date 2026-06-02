import cv2
import numpy as np
import onnxruntime as ort
import time

def preprocess_image(img_bgr, target_size=None):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    if target_size is not None:
        new_h, new_w = target_size
    else:
        # Scale to max width of 640 and ensure multiples of 16
        max_w = 640
        scale = max_w / float(w)
        new_w = max_w
        new_h = int(h * scale)
        new_h = ((new_h + 15) // 16) * 16
        new_w = ((new_w + 15) // 16) * 16
    
    img_resized = cv2.resize(img_rgb, (new_w, new_h))
    img_float = img_resized.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - mean) / std
    
    # Transpose to BCHW
    img_tensor = img_norm.transpose(2, 0, 1)
    img_batch = np.expand_dims(img_tensor, axis=0)
    return img_batch, (new_h, new_w)

def main():
    onnx_path = "ckpts/DAv2S-4.onnx"
    left_path = "assets/Keyboard/left.png"
    right_path = "assets/Keyboard/right.png"
    output_path = "assets/Keyboard/onnx_vis_disp.png"
    
    print(f"Loading ONNX model from {onnx_path}...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    print(f"Reading images...")
    img_l = cv2.imread(left_path)
    img_r = cv2.imread(right_path)
    orig_h, orig_w = img_l.shape[:2]
    
    print(f"Preprocessing images...")
    tensor_l, (new_h, new_w) = preprocess_image(img_l, target_size=(480, 640))
    tensor_r, _ = preprocess_image(img_r, target_size=(480, 640))
    
    print(f"Running inference with ONNX Runtime...")
    start_time = time.time()
    outputs = session.run(['disp_pred'], {
        'img1': tensor_l,
        'img2': tensor_r
    })
    elapsed = time.time() - start_time
    print(f"Inference took {elapsed:.3f} seconds.")
    
    disp = outputs[0][0]  # shape (H, W)
    print(f"Disparity prediction shape: {disp.shape}")
    
    # Resize back to original size
    disp_orig = cv2.resize(disp, (orig_w, orig_h))
    
    # Normalize and color map
    min_val = disp_orig.min()
    max_val = disp_orig.max()
    disp_norm = ((disp_orig - min_val) / (max_val - min_val + 1e-10) * 255.0).astype(np.uint8)
    
    # Apply colormap (turbo or jet)
    disp_colored = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)
    
    # Concatenate Left and Disparity
    concat = np.concatenate([img_l, disp_colored], axis=1)
    
    print(f"Saving output to {output_path}...")
    cv2.imwrite(output_path, concat)
    print("Test passed successfully!")

if __name__ == '__main__':
    main()
