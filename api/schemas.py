# api/schemas.py
from pydantic import BaseModel, Field, validator
from typing import Optional, Tuple, List, Dict

class PropertyInput(BaseModel):
    type_bien: str = Field(..., description="Type de bien (appartement/maison)")
    typologie: str = Field(..., 
                         description="Typologie (T1-T5+)", 
                         pattern="^T[1-5]\+?$")
    sbati: int = Field(..., 
                      description="Surface bâtie en m²",
                      gt=0,
                      lt=1000)
    sterr: int = Field(..., 
                      description="Surface du terrain en m²",
                      ge=0,
                      lt=10000)
    commune: str = Field(..., description="Commune")
    x: float = Field(..., description="Coordonnée X (Lambert 93)")
    y: float = Field(..., description="Coordonnée Y (Lambert 93)")
    
    @validator('type_bien')
    def validate_type_bien(cls, v):
        allowed = ['Appartement', 'Maison']
        if v not in allowed:
            raise ValueError(f'type_bien doit être un de {allowed}')
        return v
    
    @validator('commune')
    def validate_commune(cls, v):
        communes = [
            'Dijon', 'Chenôve', 'Talant', 'Fontaine-lès-Dijon', 'Longvic', 
            'Saint-Apollinaire', 'Quetigny', 'Chevigny-Saint-Sauveur', 
            'Neuilly-Crimolois', 'Marsannay-la-Côte', 'Perrigny-lès-Dijon',
            'Plombières-lès-Dijon', 'Ahuy', 'Daix', 'Hauteville-lès-Dijon',
            'Ouges', 'Sennecey-lès-Dijon', 'Magny-sur-Tille', 'Bressey-sur-Tille',
            'Bretenière', 'Corcelles-les-Monts', 'Flavignerot', 'Fénay'
        ]
        if v not in communes:
            raise ValueError(f'commune doit être une de {communes}')
        return v

    class Config:
        schema_extra = {
            "example": {
                "type_bien": "Appartement",
                "typologie": "T3",
                "sbati": 65,
                "sterr": 0,
                "commune": "Dijon",
                "x": 847564.2,
                "y": 6689618.1
            }
        }

class PredictionOutput(BaseModel):
    predicted_price: float
    confidence_interval: Tuple[float, float]
    comparable_properties: List[Dict]
    features_importance: Dict

class FeedbackInput(BaseModel):
    prediction_id: str
    actual_price: float
    feedback_type: str = Field(..., description="Type de feedback (too_high, too_low, accurate)")
    comments: Optional[str] = None
    
    @validator('feedback_type')
    def validate_feedback_type(cls, v):
        allowed = ['too_high', 'too_low', 'accurate']
        if v not in allowed:
            raise ValueError(f'feedback_type doit être un de {allowed}')
        return v

    class Config:
        schema_extra = {
            "example": {
                "prediction_id": "12345",
                "actual_price": 250000,
                "feedback_type": "accurate",
                "comments": "La prédiction était très précise"
            }
        }