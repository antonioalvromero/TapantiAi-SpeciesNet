import uvicorn
import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from megadetector.detection.run_md_and_speciesnet import (
    run_md_and_speciesnet,
    RunMDSpeciesNetOptions
)

app = FastAPI()


# =========================================================
# REQUEST MODEL
# =========================================================

class Instance(BaseModel):
    filepath: str

class PredictRequest(BaseModel):
    instances: list[Instance]

# =========================================================
# API ENDPOINT
# =========================================================

@app.post("/predict")
async def predict(request: PredictRequest):

    # ==========================================
    # VALIDATE PATHS
    # ==========================================

    for instance in request.instances:
        if not Path(instance.filepath).exists():
            raise HTTPException(400, f"File not found: {instance.filepath}")

    # ==========================================
    # COPY FILES TO TEMP DIR
    # ==========================================

    temp_dir = tempfile.mkdtemp()

    for instance in request.instances:
        src = Path(instance.filepath)
        dst = Path(temp_dir) / src.name
        shutil.copy2(src, dst)

    # ==========================================
    # OUTPUT FILE
    # ==========================================

    output_file = Path(temp_dir) / "results.json"

    # ==========================================
    # CONFIGURE PIPELINE
    # ==========================================

    options = RunMDSpeciesNetOptions()
    options.source = str(temp_dir)
    options.output_file = str(output_file)
    options.keep_intermediate_files = False
    options.verbose = True

    # ==========================================
    # RUN PIPELINE
    # ==========================================

    try:
        run_md_and_speciesnet(options)

        if not output_file.exists():
            raise HTTPException(500, "No output generated")

        with open(output_file, "r") as f:
            results = json.load(f)

        return results

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    uvicorn.run(
        "run_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )