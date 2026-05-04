import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.database import create_tables
from app.services import local_generation
from app.routers import simulate


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Iniciando Aesthetic API...")

    await create_tables()
    print("✅ Banco de dados OK.")

    print(f"⏳ Carregando modelo SD (device={settings.SD_DEVICE})...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        local_generation.load_pipeline,
        settings.SD_DEVICE,
        settings.SD_PRECISION,
    )
    print("✅ Modelo carregado. API pronta!")

    yield

    print("🛑 Encerrando.")


app = FastAPI(
    title="Aesthetic Simulation API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve as imagens geradas
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(simulate.router, prefix="/api/v1", tags=["Simulate"])


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}