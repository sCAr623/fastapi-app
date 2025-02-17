from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

# Load your trained YOLO model
model = YOLO("best.pt")  # Replace with your trained model file

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Plant Disease Prediction API is Running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image = Image.open(io.BytesIO(await file.read()))

        # Run YOLO model
        results = model(image)

        # Extract predictions (Modify this based on your model's output format)
        predictions = []
        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls)]
                confidence = float(box.conf)
                predictions.append({"label": label, "confidence": confidence})

        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

