import asyncio
import contextlib
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import CLEANUP_INTERVAL_SECONDS
from .routes import bind_service, router
from .service import Sam3Service


service = Sam3Service()
bind_service(service)
async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        await service.close_expired_sessions()


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(_cleanup_loop())
    app.state.cleanup_task = cleanup_task
    try:
        yield
    finally:
        cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup_task
        for session_id in list(service.sessions.keys()):
            service.close_session(session_id)


app = FastAPI(title="SAM3 Video Masking API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


