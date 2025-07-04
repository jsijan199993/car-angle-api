from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import timm
from PIL import Image
from torchvision import transforms
import io
import os
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['back', 'front', 'left-side', 'right-side', 'tachometer', 'unknown']
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = "angle_classifier_convnext.pt"
MODEL_URL = "https://huggingface.co/jsijan/car-angle-model/resolve/main/angle_classifier_convnext.pt"

# 📦 Load model
model = None
try:
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Hugging Face...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

    print("🧠 Loading model...")
    model = timm.create_model('convnext_base', pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("✅ Model ready.")
except Exception as e:
    print(f"❌ Model loading failed: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.get("/")
def root():
    return {"message": "API running", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    top_prob, top_class = torch.max(probs, 0)
    return {
        "prediction": CLASS_NAMES[top_class],
        "confidence": f"{top_prob.item():.2f}",
        "all_confidences": {
            CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(NUM_CLASSES)
        }
    }

# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# import torch
# import timm
# from PIL import Image
# from torchvision import transforms
# import io
# import os

# app = FastAPI()

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # 📦 Config
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CLASS_NAMES = ['back', 'front', 'left-side', 'right-side', 'tachometer', 'unknown']
# NUM_CLASSES = len(CLASS_NAMES)
# MODEL_PATH = "angle_classifier_convnext.pt"
# GDRIVE_ID = "16vAuLUiL9Jy-S0eIie7oFoawDHavLZQ9"
# GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"

# # Load model
# model = None
# try:
#     if not os.path.exists(MODEL_PATH):
#         import gdown
#         print("📦 Downloading model from Google Drive...")
#         gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

#     print("🧠 Loading convnext_base...")
#     model = timm.create_model('convnext_base', pretrained=False, num_classes=NUM_CLASSES)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model.to(DEVICE)
#     model.eval()
#     print("✅ Model loaded.")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")

# # 🔁 Transform
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# @app.get("/")
# def root():
#     return {"message": "API is running", "model_loaded": model is not None}

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     if model is None:
#         return {"error": "Model not loaded."}

#     image_bytes = await file.read()
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     input_tensor = transform(image).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         outputs = model(input_tensor)
#         probs = torch.nn.functional.softmax(outputs[0], dim=0)

#     top_prob, top_class = torch.max(probs, 0)
#     return {
#         "prediction": CLASS_NAMES[top_class],
#         "confidence": f"{top_prob.item():.2f}",
#         "all_confidences": {
#             CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(NUM_CLASSES)
#         }
#     }

# # from fastapi import FastAPI, UploadFile, File
# # from fastapi.middleware.cors import CORSMiddleware
# # import torch
# # import timm
# # from PIL import Image
# # from torchvision import transforms
# # import io
# # import os
# # import gdown

# # app = FastAPI()

# # # CORS for React frontend
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],  # or restrict to your frontend URL
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )
# # @app.get("/")
# # def root():
# #     return {"message": "API is running (no model loaded)"}

# # # # Load model
# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #  = ['back', 'front', 'left-side', 'right-side', 'tachometer', 'unknown']
# # # NUM_CLASSES = len()

# # # model = timm.create_model('convnext_base', pretrained=False, num_classes=NUM_CLASSES)
# # # model.load_state_dict(torch.load("angle_classifier_convnext.pt", map_location=DEVICE))
# # # model.to(DEVICE)
# # # model.eval()

# # # https://drive.google.com/file/d/16vAuLUiL9Jy-S0eIie7oFoawDHavLZQ9/view?usp=sharing

# # MODEL_PATH = "angle_classifier_convnext.pt"
# # GDRIVE_ID = "16vAuLUiL9Jy-S0eIie7oFoawDHavLZQ9"
# # GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"

# # if not os.path.exists(MODEL_PATH):
# #     print("📦 Model not found, downloading from Google Drive...")
# #     gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False, use_cookies=True)

# # model = timm.create_model('convnext_base', pretrained=False, num_classes=NUM_CLASSES)
# # model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# # model.to(DEVICE)
# # model.eval()
# # print("✅ Model loaded")

# # # 
# # transform = transforms.Compose([
# #     transforms.Resize((224, 224)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                          std=[0.229, 0.224, 0.225])
# # ])

# # @app.post("/predict")
# # async def predict(file: UploadFile = File(...)):
# #     image_bytes = await file.read()
# #     image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
# #     input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# #     with torch.no_grad():
# #         outputs = model(input_tensor)
# #         probs = torch.nn.functional.softmax(outputs[0], dim=0)

# #     top_prob, top_class = torch.max(probs, 0)

# #     return {
# #         "prediction": [top_class],
# #         "confidence": f"{top_prob.item():.2f}",
# #         "all_confidences": {
# #             [i]: round(probs[i].item(), 4) for i in range(NUM_CLASSES)
# #         }
# #     }
