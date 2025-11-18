import pandas as pd
from thefuzz import process, fuzz
import os
import sys


def resource_path(relative_path: str) -> str:
    # running as a PyInstaller onefile exe, files are unpacked to _MEIPASS
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, relative_path)

data_path = resource_path(os.path.join("cars_info", "automobiles_cleaned_2025_11_17.csv"))

df = pd.read_csv(data_path, sep=",")


def find_cars(query: str, limit=5, scorer=fuzz.partial_ratio, return_metrics=False):
    clean_series = df['car_model'].dropna().astype(str)
    top_matches = process.extract(query, clean_series.tolist(), limit=limit, scorer=scorer)

    def _safe(val):
        if pd.isna(val):
            return None
        try:
            return val.item()
        except Exception:
            return val

    results = []
    seen_combinations = set()
    scores = []

    for text_match, score in top_matches:
        scores.append(score)
        matching_indices = clean_series[clean_series == text_match].index.tolist()

        for original_index in matching_indices[:limit]:
            if original_index in df.index:
                combo = (text_match, int(score), int(original_index))
                if combo in seen_combinations:
                    continue
                seen_combinations.add(combo)

                model = df.loc[original_index]
                results.append({
                    'suggestion_text': _safe(text_match),
                    'score': int(score),
                    'original_index': int(original_index),
                    'brand_name': _safe(model.get('brand_name')),
                    'length_mm': _safe(model.get('length_mm')),
                    'width_mm': _safe(model.get('width_mm')),
                    'height_mm': _safe(model.get('height_mm')),
                    'kg': _safe(model.get('kg')),
                    'rank': _safe(model.get('rank')),
                })

    response = {'query': query, 'results': results}

    if return_metrics:
        avg_score = sum(scores) / len(scores) if scores else 0
        response['metrics'] = {
            'avg_score': avg_score,
            'scorer': scorer.__name__
        }

    return response
