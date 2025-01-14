from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from . import schemas, database, dependencies
from typing import List

app = FastAPI(
    title="API Prédiction Immobilière Dijon",
    description="API pour la prédiction des prix immobiliers à Dijon",
    version="1.0.0"
)

@app.post("/predict", response_model=schemas.PredictionOutput)
def predict_price(
    property_input: schemas.PropertyInput,
    db: Session = Depends(dependencies.get_db),
    model_service: dependencies.ModelService = Depends(dependencies.get_model_service)
):
    try:
        result = model_service.predict(property_input.dict())
        prediction = database.Prediction(
            input_data=property_input.dict(),
            predicted_price=result["predicted_price"],
            model_version="1.0.0",
            confidence_interval=result["confidence_interval"]
        )
        db.add(prediction)
        db.commit()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)