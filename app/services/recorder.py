import os
import asyncio
from playwright.async_api import async_playwright
from app.models.schemas import Scene

async def record_scene(scene: Scene, app_url: str, output_video_path: str, target_duration_sec: float) -> str:
    """Launches Playwright headless browser, executes scene actions, and records video."""
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    temp_dir = os.path.dirname(output_video_path)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=temp_dir,
            record_video_size={"width": 1920, "height": 1080}
        )
        page = await context.new_page()

        start_time = asyncio.get_event_loop().time()
        target_url = app_url.rstrip("/") + ("" if scene.route.startswith("/") else "/") + scene.route

        try:
            await page.goto(target_url, timeout=15000, wait_until="networkidle")
        except Exception:
            try:
                await page.goto(target_url, timeout=10000, wait_until="domcontentloaded")
            except Exception as e:
                print(f"Navigation warning: {e}")

        # Execute actions
        for act in scene.actions:
            try:
                if act.type == "navigate" and act.value:
                    dest = app_url.rstrip("/") + ("" if act.value.startswith("/") else "/") + act.value
                    await page.goto(dest, timeout=10000)
                elif act.type == "click" and act.selector:
                    await page.click(act.selector, timeout=3000)
                elif act.type == "type" and act.selector and act.value:
                    await page.fill(act.selector, act.value, timeout=3000)
                elif act.type == "scroll":
                    if act.value == "down":
                        await page.evaluate("window.scrollBy({ top: 600, behavior: 'smooth' })")
                    elif act.value == "top":
                        await page.evaluate("window.scrollTo({ top: 0, behavior: 'smooth' })")
                await asyncio.sleep(act.dwell_ms / 1000.0)
            except Exception as act_err:
                print(f"Action failed non-critically ({act.type}): {act_err}")

        # Ensure video duration matches audio narration length
        elapsed = asyncio.get_event_loop().time() - start_time
        remaining = target_duration_sec - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)

        # Close context to flush video file
        await page.close()
        video = page.video
        raw_video_path = await video.path() if video else None
        await context.close()
        await browser.close()

        if raw_video_path and os.path.exists(raw_video_path):
            os.replace(raw_video_path, output_video_path)
            return output_video_path

    return output_video_path
