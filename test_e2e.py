import os
import sys
import asyncio
import argparse
from app.models.schemas import PipelineRequest
from app.services.orchestrator import create_pipeline_job, JOBS

async def main():
    parser = argparse.ArgumentParser(description="Test DemoDay Agent Backend E2E Pipeline")
    parser.add_argument("--repo", default="https://github.com/fastapi/fastapi", help="GitHub repository URL")
    parser.add_argument("--url", default="https://fastapi.tiangolo.com", help="Live deployed application URL")
    parser.add_argument("--seconds", type=int, default=30, help="Target demo length in seconds")
    args = parser.parse_args()

    fal_key = os.getenv("FAL_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    print("=" * 60)
    print("DemoDay Agent - End-to-End Pipeline Test")
    print("=" * 60)
    print(f"Target Repo:  {args.repo}")
    print(f"Target URL:   {args.url}")
    print(f"Target Time:  {args.seconds}s")
    print(f"FAL_KEY:      {'[CONFIGURED]' if fal_key else '[UNSET - will use fallback audio]'}")
    print(f"GEMINI_KEY:   {'[CONFIGURED]' if gemini_key else '[UNSET - will use template storyboard]'}")
    print("=" * 60)

    req = PipelineRequest(
        repo_url=args.repo,
        app_url=args.url,
        target_seconds=args.seconds,
        voice_style="energetic_tech"
    )

    job = create_pipeline_job(req)
    job_id = job.job_id
    print(f"Started Pipeline Job ID: {job_id}")

    last_log_count = 0
    while True:
        await asyncio.sleep(1.0)
        curr_job = JOBS.get(job_id)
        if not curr_job:
            break

        # Print new logs
        if len(curr_job.logs) > last_log_count:
            for log in curr_job.logs[last_log_count:]:
                print(f"[{curr_job.progress:3d}%] {log}")
            last_log_count = len(curr_job.logs)

        if curr_job.status in ["completed", "failed"]:
            print("=" * 60)
            if curr_job.status == "completed":
                print(f"SUCCESS! Output demo video: storage/{job_id}/demo.mp4")
                if curr_job.launch_kit:
                    print("\n--- Pitch Preview ---")
                    print(curr_job.launch_kit.devpost_pitch[:300] + "...\n")
            else:
                print(f"FAILED: {curr_job.error}")
            print("=" * 60)
            break

if __name__ == "__main__":
    asyncio.run(main())
