from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List
from modal import Image, Function, App,Stub, asgi_app, Secret, Volume
from modal.functions import FunctionCall
from schemas import ClassificationInput, ClassificationOutput, DetachProcess

app = App("Modal")

api_image = (
    Image.debian_slim(python_version="3.12")
    .pip_install(
        "pydantic==2.6.4",
        "fastapi==0.110.1",
        "python-multipart==0.0.9",
        "kafka-python-ng==2.2.2",
    )
)

@app.function(
        image=api_image
        ,container_idle_timeout=10
)
@asgi_app()
def fastapi_app():

    api = FastAPI()

    @api.get("/")
    async def index():
        return {"message":"version-0.0.1"}

    @api.post("/classification")
    async def detach_classification(input: ClassificationInput):
        process = DetachProcess("12893765","asLKJFHDSALKH89#(@)")
        return process


    return api
