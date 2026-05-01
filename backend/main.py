from fastapi import FastAPI, File, UploadFile, HTTPException
from utils import strip_dicom_pii
import io

app = FastAPI(title="DentaVision Backend")

@app.get("/")
async def root():
    return {"message": "DentaVision API is running", "status": "active"}

@app.post("/upload/dicom")
async def upload_dicom(file: UploadFile = File(...)):
    """
    Receives a .dcm file, strips PII, and returns basic metadata.
    """
    if not file.filename.endswith(".dcm"):
         raise HTTPException(status_code=400, detail="Invalid file format. Only .dcm files allowed.")
    
    try:
        content = await file.read()
        
        # Strip PII using utility function
        cleaned_dcm = strip_dicom_pii(content)
        
        return {
            "filename": file.filename,
            "status": "processed",
            "modality": cleaned_dcm.get("Modality", "Unknown"),
            "image_size": f"{cleaned_dcm.Rows}x{cleaned_dcm.Columns}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")