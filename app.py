from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch, timm, os
from PIL import Image
import torchvision.transforms as transforms
import io

app = FastAPI()

# ---- SAFETY: ensure dirs exist ----
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---- model loading ----
device = torch.device("cpu")

checkpoint = torch.load("model.pth", map_location=device)

model = timm.create_model(
    checkpoint["model_name"],
    pretrained=False,
    num_classes=checkpoint["num_classes"]
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

classes = [
    'Bank statements',
    'Gov Forms',
    'Insurance_Policy',
    'Invoice',
    'Resume'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Read image file
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            out = model(img)
            probabilities = torch.nn.functional.softmax(out, dim=1)
            pred = torch.argmax(out, 1).item()
            confidence = float(probabilities[0][pred]) * 100

        return JSONResponse({
            "prediction": classes[pred],
            "confidence": round(confidence, 2),
            "class_index": int(pred),
            "all_classes": classes,
            "status": "success"
        })
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "status": "error"
        }, status_code=500)

# Fallback route for backward compatibility
@app.post("/", response_class=HTMLResponse)
async def predict_html(request: Request, image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            out = model(img)
            pred = torch.argmax(out, 1).item()

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction": classes[pred]}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction": f"Error: {str(e)}"}
        )
@app.post("/predict_all")
async def predict_all(image: UploadFile = File(...)):
    try:
        # Read image file
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            out = model(img)
            probabilities = torch.nn.functional.softmax(out, dim=1)
            pred = torch.argmax(out, 1).item()
            
            # Get all probabilities
            all_probs = probabilities[0].tolist()
            
            # Create dictionary of class: probability
            class_probabilities = {
                classes[i]: float(all_probs[i] * 100)  # Convert to percentage
                for i in range(len(classes))
            }
            
            # Sort by probability (highest first)
            sorted_probs = dict(sorted(
                class_probabilities.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))

        return JSONResponse({
            "prediction": classes[pred],
            "top_confidence": round(float(probabilities[0][pred]) * 100, 2),
            "all_probabilities": sorted_probs,
            "status": "success"
        })
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "status": "error"
        }, status_code=500)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)