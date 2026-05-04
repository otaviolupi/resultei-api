from app.db.database import AsyncSessionLocal
from app.db.models import SimulationJob


async def save_job(job_id, procedure, intensity, mode,
                   original_url, result_url, comparison_url,
                   processing_time_ms=0):
    async with AsyncSessionLocal() as session:
        job = SimulationJob(
            id=job_id,
            procedure=procedure,
            intensity=intensity,
            mode=mode,
            original_url=original_url,
            result_url=result_url,
            comparison_url=comparison_url,
            processing_time_ms=processing_time_ms,
        )
        session.add(job)
        await session.commit()


async def get_job(job_id: str):
    async with AsyncSessionLocal() as session:
        return await session.get(SimulationJob, job_id)