from fastapi import FastAPI, File, UploadFile
from Main import MODEL
from PIL import Image
from pydantic import BaseModel
from io import BytesIO

app = FastAPI()



@app.post("/evaluate/")
async def get(image: UploadFile = File()):
    image_data = await image.read()
    image = Image.open(BytesIO(image_data))

    result = MODEL.predict(image)

    return {"result": result}