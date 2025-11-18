import os
import cv2
import numpy as np
from difflib import get_close_matches
from PIL import Image
import torch
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Globals (lazy init)
_ocr_reader = None
_trocr_processor = None
_trocr_model = None

# candidate model names (lowercased)
model_dict = [m.lower() for m in [
    "320d", "320i", "330i", "335d", "335i", "530e", "530i", "540i",
    "M2", "M3", "M4", "M5", "M8", "X3", "X4", "X5", "Z3", "Z4",
    "Figo", "Focus", "Fusion", "Mondeo", "Ranger",
    "Civic", "Accord", "City", "CR-V", "HR-V",
    "Accent", "Elantra", "Tucson", "SantaFe", "Sonata",
    "K3", "K5", "Morning", "Seltos", "Sorento",
    "C200", "C300", "E200", "E300", "GLC200", "GLC300",
    "GLE53", "GLE450", "S450", "S500",
    "Attrage", "Mirage", "PAJERO", "OUTLANDER", "MIRAGE", "PAJERO SPORT",
    "Camry", "Corolla", "Highlander", "Prius",
    "VF3", "VF5", "VF6", "VF8", "VF9"
]]

def get_easyocr_reader(lang_list=('en',)):
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(list(lang_list), gpu=torch.cuda.is_available())
    return _ocr_reader

def get_trocr_model():
    global _trocr_processor, _trocr_model
    if _trocr_model is None:
        _trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
        _trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
        _trocr_model.eval()
    return _trocr_processor, _trocr_model

def _crop_with_padding(img, bbox_pts, padding=5, zoom=1.0):
    box_np = np.array(bbox_pts, dtype=np.int32)
    x_min, y_min = np.min(box_np, axis=0)
    x_max, y_max = np.max(box_np, axis=0)
    h, w = img.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    if x_min >= x_max or y_min >= y_max:
        return None
    crop = img[y_min:y_max, x_min:x_max]
    if zoom != 1.0 and crop.size > 0:
        new_w = max(1, int((x_max - x_min) * zoom))
        new_h = max(1, int((y_max - y_min) * zoom))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return crop


def detect_and_crop(image_path, padding=5, crop_zoom=2.0, langs=('en',)):
    """
    Returns list of dicts: { 'bbox': pts, 'easy_text': text, 'easy_conf': conf, 'crop_pil': PIL.Image }
    """
    print("detect_and_crop image_path:", image_path)
    try:
        # Read the image first
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # Convert to grayscale for EasyOCR
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        reader = get_easyocr_reader(langs)
        # Pass the grayscale image as numpy array instead of file path
        results = reader.readtext(img_gray, detail=1)

        outputs = []
        for result in results:
            bbox, text, conf = result
            # Use the original color image for cropping
            crop = _crop_with_padding(img, bbox, padding=padding, zoom=crop_zoom)
            if crop is None or crop.size == 0:
                continue
            # Convert BGR cv2 crop to RGB PIL
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            outputs.append({
                'bbox': bbox,
                'easy_text': text,
                'easy_conf': float(conf),
                'crop_pil': pil_img
            })
        return outputs
    except Exception as e:
        import traceback
        print(f"Error in detect_and_crop: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return []


def recognize_trocr(crops, device=None, max_length=64, num_beams=1, match_cutoff=0.7):
    """
    Accepts list of PIL images. Returns list of dicts: { 'trocr_text': str, 'match': str or None }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = get_trocr_model()
    model.to(device)
    model.eval()
    results = []
    for img in crops:
        try:
            pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_length=max_length, num_beams=num_beams)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            clean_text = ''.join(filter(str.isalnum, text)).lower()
            match = get_close_matches(clean_text, model_dict, n=1, cutoff=match_cutoff)
            match_str = match[0] if match else None
            results.append({'trocr_text': text, 'match': match_str})
        except Exception as e:
            results.append({'trocr_text': f'ERROR: {e}', 'match': None})
    return results

def run_ocr_pipeline(image_path, padding=5, crop_zoom=2.0, langs=('en',), trocr_max_length=64, trocr_num_beams=1, match_cutoff=0.7):
    """
    High-level entry point.
    Returns a dict with:
      - easyocr_texts: list of strings (EasyOCR raw)
      - trocr_texts: list of strings (TrOCR outputs)
      - matches: list of matched model names or None
      - details: list of dicts per region (contains bbox, easy_conf, etc.)
    """
    details = detect_and_crop(image_path, padding=padding, crop_zoom=crop_zoom, langs=langs)
    if not details:
        return {'easyocr_texts': [], 'trocr_texts': [], 'matches': [], 'details': []}
    crops = [d['crop_pil'] for d in details]
    trocr_results = recognize_trocr(crops, max_length=trocr_max_length, num_beams=trocr_num_beams, match_cutoff=match_cutoff)
    easy_texts = [d['easy_text'] for d in details]
    trocr_texts = [r['trocr_text'] for r in trocr_results]
    #matches = [r['match'] for r in trocr_results]
    for d, r in zip(details, trocr_results):
        d.update(r)
    return {
        'easyocr_texts': easy_texts,
        'trocr_texts': trocr_texts,
        #'matches': matches,
        'details': details
    }

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python ocr/no_ui_pipeline.py <image_path>")
        sys.exit(1)
    path = sys.argv[1]
    out = run_ocr_pipeline(path)
    print(json.dumps({
        'easyocr_texts': out['easyocr_texts'],
        'trocr_texts': out['trocr_texts'],
        'matches': out['matches']
    }, indent=2))
