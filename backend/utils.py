import pydicom
from fastapi import UploadFile
import io

def strip_dicom_pii(file_bytes: bytes) -> pydicom.dataset.FileDataset:
    """
    Reads a DICOM file from bytes, removes private headers, 
    and returns the cleaned dataset.
    """

    dcm = pydicom.dcmread(io.BytesIO(file_bytes))
    
    # based on paper's privacy requirements, PII to remove
    tags_to_remove = [
        (0x0010, 0x0010), # Name
        (0x0010, 0x0030), # Birth Date
        (0x0010, 0x0040), # Gender
        (0x0010, 0x1010), # Age
        (0x0010, 0x2154), # Phone Number
    ]

    for tag in tags_to_remove:
        if tag in dcm:
            del dcm[tag]
            
    return dcm