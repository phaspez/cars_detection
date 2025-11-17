from text_detection.text_detection import run_ocr_pipeline


if __name__ == "__main__":
    image_path = r"data_png/bmw/iX3_1.png"
    result = run_ocr_pipeline(image_path, padding=5, crop_zoom=2.0, langs=('en',), trocr_max_length=64, trocr_num_beams=1, match_cutoff=0.7)
    print(result)