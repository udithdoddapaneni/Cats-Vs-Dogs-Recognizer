from fastapi import FastAPI, File, UploadFile
from Main import MODEL
from PIL import Image
from pydantic import BaseModel
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this to specific origins for production)
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.post("/evaluate/")
async def get(image: UploadFile = File()):
    image_data = await image.read()
    image = Image.open(BytesIO(image_data))
    result = MODEL.predict(image)
    print(result)
    return {"result": result}
