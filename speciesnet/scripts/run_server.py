import uvicorn
import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException

from megadetector.detection.run_md_and_speciesnet import (
    run_md_and_speciesnet,
    RunMDSpeciesNetOptions
)

app = FastAPI()


@app.post("/predict")
async def predict(
    files: list[UploadFile] = File(...)
):

    # ==========================================
    # CREATE TEMP FOLDER
    # ==========================================

    temp_dir = tempfile.mkdtemp()

    # ==========================================
    # SAVE IMAGES
    # ==========================================

    for file in files:

        file_path = Path(temp_dir) / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # ==========================================
    # OUTPUT FILE
    # ==========================================

    output_file = Path(temp_dir) / "results.json"

    # ==========================================
    # CONFIGURE PIPELINE
    # ==========================================

    options = RunMDSpeciesNetOptions()

    options.source = temp_dir
    options.output_file = str(output_file)

    options.keep_intermediate_files = False
    options.verbose = True

    # ==========================================
    # RUN PIPELINE
    # ==========================================

    run_md_and_speciesnet(options)

    # ==========================================
    # LOAD RESULTS
    # ==========================================

    if not output_file.exists():
        raise HTTPException(500, "No output generated")

    with open(output_file, "r") as f:
        results = json.load(f)

    return results

if __name__ == "__main__":
    uvicorn.run(
        "run_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )