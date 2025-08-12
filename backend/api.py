# backend/api.py
import io, base64, os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import torch
from models.unet import load_unet
from models.nlp_model import TweetClassifier

app = FastAPI(title="DisasterHybridAPI")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNET_PATH = os.path.join("models", "weights", "unet_ckpt.pth")
nlp = TweetClassifier()
unet = load_unet(UNET_PATH, device=DEVICE)

def preprocess_pil(img: Image.Image, size=(256,256)):
    img = img.convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE)
    return tensor

def postprocess_mask(mask_tensor):
    mask = torch.sigmoid(mask_tensor).squeeze().detach().cpu().numpy()
    mask = (mask > 0.5).astype("uint8")*255
    return mask

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    tensor = preprocess_pil(img)
    with torch.no_grad():
        out = unet(tensor)
    mask = postprocess_mask(out)
    # return base64 encoded mask png
    buf = io.BytesIO()
    Image.fromarray(mask).convert("L").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return JSONResponse({"mask_b64": b64})

@app.post("/predict/tweet")
async def predict_tweet(payload: dict):
    text = payload.get("text", "")
    if not text:
        return JSONResponse({"error":"no text provided"}, status_code=400)
    res = nlp.predict(text)
    return JSONResponse(res)

