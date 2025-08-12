# streamlit_app/utils.py
import os, json, time
from pathlib import Path
from models.unet import load_unet
from models.nlp_model import TweetClassifier
from PIL import Image
import numpy as np
import io, base64

# load models (lazy)
MODEL = {}
def get_unet():
    if "unet" not in MODEL:
        ckpt = os.path.join("models", "weights", "unet_ckpt.pth")
        MODEL["unet"] = load_unet(ckpt)
    return MODEL["unet"]

def get_tweet_clf():
    if "nlp" not in MODEL:
        MODEL["nlp"] = TweetClassifier()
    return MODEL["nlp"]

def run_unet_on_pil(img_pil):
    model = get_unet()
    img = img_pil.convert("RGB").resize((256,256))
    arr = np.array(img).astype("float32")/255.0
    import torch
    tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
    mask = (torch.sigmoid(out).squeeze().cpu().numpy() > 0.5).astype("uint8")*255
    # overlay
    overlay = Image.fromarray(mask).convert("RGBA").resize(img_pil.size)
    base = img_pil.convert("RGBA")
    # green overlay with alpha
    overlay_arr = np.array(overlay)
    overlay_col = np.zeros_like(overlay_arr)
    overlay_col[...,1] = overlay_arr[...,0]  # put mask into green channel
    overlay_col[...,3] = (overlay_arr[...,0]//2)  # semi alpha
    overlay_img = Image.fromarray(overlay_col)
    blended = Image.alpha_composite(base.convert("RGBA"), overlay_img)
    return blended, mask

def load_sample_tweets():
    sample_file = Path("data/sample/tweets_sample.json")
    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            return json.load(f)
    # fallback sample
    return [
        {"id":"1","text":"Huge flooding near the river, houses submerged, need help!", "time":"2025-01-01T00:00:00Z"},
        {"id":"2","text":"Road closed due to storm debris", "time":"2025-01-01T00:01:00Z"},
        {"id":"3","text":"Beautiful view after the storm", "time":"2025-01-01T00:02:00Z"},
    ]

