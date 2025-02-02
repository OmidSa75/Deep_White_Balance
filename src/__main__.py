import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import torch
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager

# Import your existing modules
from src.arch import deep_wb_model, deep_wb_single_task
import src.utilities.utils as utls
from src.utilities.deepWB import deep_wb
import src.arch.splitNetworks as splitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# Global variables for models
MODEL_DIR = "./models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wb_models = {
    "net_awb": None,
    "net_t": None,
    "net_s": None
}

# Pydantic model for request body
class WhiteBalanceRequest(BaseModel):
    task: str  # 'AWB', 'editing', or 'all'
    target_color_temp: Optional[int] = None  # Required if task is 'editing'

# Load models at startup
@asynccontextmanager
async def load_models(app: FastAPI):
    logging.info("Loading models...")

    # Load AWB model
    awb_path = os.path.join(MODEL_DIR, "net_awb.pth")
    if os.path.exists(awb_path):
        net_awb = deep_wb_single_task.deepWBnet()
        net_awb.load_state_dict(torch.load(awb_path, map_location=DEVICE))
        net_awb.to(DEVICE)
        net_awb.eval()
        wb_models["net_awb"] = net_awb
    else:
        logging.warning("AWB model not found!")

    # Load Tungsten and Shade models
    t_path = os.path.join(MODEL_DIR, "net_t.pth")
    s_path = os.path.join(MODEL_DIR, "net_s.pth")
    if os.path.exists(t_path) and os.path.exists(s_path):
        net_t = deep_wb_single_task.deepWBnet()
        net_t.load_state_dict(torch.load(t_path, map_location=DEVICE))
        net_t.to(DEVICE)
        net_t.eval()

        net_s = deep_wb_single_task.deepWBnet()
        net_s.load_state_dict(torch.load(s_path, map_location=DEVICE))
        net_s.to(DEVICE)
        net_s.eval()
        wb_models["net_t"] = net_t
        wb_models["net_s"] = net_s
    else:
        logging.warning("Tungsten or Shade model not found!")

    logging.info("Models loaded successfully.")
    yield
    wb_models.clear()

# Helper function to process the image
def process_image(image: Image.Image, task: str, target_color_temp: Optional[int] = None):
    if task == "AWB":
        if wb_models["net_awb"] is None:
            raise HTTPException(status_code=500, detail="AWB model not loaded.")
        out_awb = deep_wb(image, task=task, net_awb=wb_models["net_awb"], device=DEVICE)
        return {"AWB": utls.to_image(out_awb)}

    elif task == "editing":
        if wb_models["net_t"] is None or wb_models["net_s"] is None:
            raise HTTPException(status_code=500, detail="Editing models not loaded.")
        if target_color_temp is None:
            raise HTTPException(status_code=400, detail="Target color temperature is required for editing task.")
        out_t, out_s = deep_wb(image, task=task, net_t=wb_models["net_t"], net_s=wb_models["net_s"], device=DEVICE)
        out = utls.colorTempInterpolate_w_target(out_t, out_s, target_color_temp)
        return {"edited": utls.to_image(out)}

    elif task == "all":
        if wb_models["net_awb"] is None or wb_models["net_t"] is None or wb_models["net_s"] is None:
            raise HTTPException(status_code=500, detail="One or more models not loaded.")
        out_awb, out_t, out_s = deep_wb(image, task=task, net_awb=wb_models["net_awb"], net_t=wb_models["net_t"], net_s=wb_models["net_s"], device=DEVICE)
        out_f, out_d, out_c = utls.colorTempInterpolate(out_t, out_s)
        return {
            "AWB": utls.to_image(out_awb),
            "T": utls.to_image(out_t),
            "S": utls.to_image(out_s),
            "F": utls.to_image(out_f),
            "D": utls.to_image(out_d),
            "C": utls.to_image(out_c),
        }

    else:
        raise HTTPException(status_code=400, detail="Invalid task. Use 'AWB', 'editing', or 'all'.")

# Initialize FastAPI app
app = FastAPI(title="Deep White-Balance Editing API", version="1.0", lifespan=load_models)

# Endpoint for white-balance editing
@app.post("/whitebalance/")
async def whitebalance_endpoint(
    file: UploadFile = File(...),
    task: str = "all",
    target_color_temp: Optional[int] = None,
):
    try:
        # Validate task
        if task not in ["AWB", "editing", "all"]:
            raise HTTPException(status_code=400, detail="Invalid task. Use 'AWB', 'editing', or 'all'.")

        # Validate target color temperature for editing task
        if task == "editing" and target_color_temp is None:
            raise HTTPException(status_code=400, detail="Target color temperature is required for editing task.")

        # Read and validate the image
        image = Image.open(file.file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process the image
        results = process_image(image, task, target_color_temp)

        # Save results to temporary files
        output_files = {}
        for key, img in results.items():
            output_path = f"/tmp/{key}_{file.filename}"
            img.save(output_path)
            output_files[key] = output_path
            logging.info(f"key: {key}")

        # Return the processed images
        return FileResponse(output_files["AWB"] if task == "all" else output_files["edited"])

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Deep White-Balance Editing API"}
