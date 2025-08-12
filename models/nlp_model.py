# models/nlp_model.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import os

class TweetClassifier:
    def __init__(self, model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english", device=-1):
        # device=-1 => CPU by pipeline will detect
        self.device = 0 if torch.cuda.is_available() else -1
        try:
            self.nlp = pipeline("text-classification", model=model_name_or_path, tokenizer=model_name_or_path, device=self.device)
            self.mode = "hf"
        except Exception as e:
            print("HF pipeline failed, falling back to rule-based classifier:", e)
            self.mode = "rule"

    def predict(self, text):
        if self.mode == "hf":
            out = self.nlp(text, truncation=True)
            # pipeline returns label & score; map to disaster severity heuristically
            label = out[0]["label"].lower()
            score = float(out[0]["score"])
            if "positive" in label or score > 0.9:
                severity = "Severe"
            elif score > 0.7:
                severity = "Moderate"
            else:
                severity = "Not disaster"
            return {"label": label, "score": score, "severity": severity}
        else:
            # simple keyword-based fallback
            txt = text.lower()
            words = ["flood", "earthquake", "help", "rescue", "collapsed", "injured", "trapped", "typhoon", "hurricane", "flooding"]
            hits = sum(1 for w in words if w in txt)
            if hits >= 2:
                severity = "Severe"
            elif hits == 1:
                severity = "Moderate"
            else:
                severity = "Not disaster"
            return {"label": "keyword-rule", "score": float(hits)/len(words), "severity": severity}

