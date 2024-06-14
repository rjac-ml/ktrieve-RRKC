from pydantic import BaseModel, Field

class ClassificationInput(BaseModel):
    text: str = Field(description="text to be classify in the ENN Process")

class ClassificationOutput(BaseModel):
    text: str = Field(description="text to be classify in the ENN Process")
    label: str = Field(description="ENN Classification")

class DetachProcess(BaseModel):
    process_id: str = Field(description="Process ID reference")
    process_key: str 