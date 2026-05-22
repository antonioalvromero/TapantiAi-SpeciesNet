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


# =========================================================
# SIMPLIFY RESPONSE
# =========================================================

def simplify_results(results):

    simplified = {
        "images": []
    }

    # =====================================================
    # CATEGORY MAP
    # =====================================================

    category_map = results.get("classification_categories", {})

    for image in results.get("images", []):

        image_result = {
            "file": image.get("file"),
            "detections": []
        }

        for det in image.get("detections", []):

            detection = {
                "bbox": det.get("bbox"),
                "detection_confidence": det.get("conf")
            }

            # =================================================
            # CLASSIFICATIONS
            # =================================================

            classifications = det.get("classifications", [])

            if classifications:

                top_class = max(
                    classifications,
                    key=lambda x: x[1]
                )

                species_id = str(top_class[0])

                detection["species_id"] = species_id

                detection["species"] = category_map.get(
                    species_id,
                    "unknown"
                )

                detection["species_confidence"] = top_class[1]

            image_result["detections"].append(detection)

        simplified["images"].append(image_result)

    return simplified


# =========================================================
# API ENDPOINT
# =========================================================

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

    try:

        run_md_and_speciesnet(options)

        if not output_file.exists():
            raise HTTPException(500, "No output generated")

        with open(output_file, "r") as f:
            results = json.load(f)

        # ======================================
        # SIMPLIFY JSON
        # ======================================

        simplified_results = simplify_results(results)

        return simplified_results

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