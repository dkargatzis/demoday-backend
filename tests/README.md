# Checks

Offline suites need ffmpeg and Playwright's Chromium, nothing else: no API keys and no network beyond local fake
servers. Each is a plain script that prints one line per check and exits 1 on a failure.

```bash
.venv/bin/python tests/run_checks.py            # all suites side by side, about 30 s
.venv/bin/python tests/run_checks.py audience   # only the suites whose name contains "audience"
.venv/bin/python tests/check_bookends.py        # one suite, every check printed
```

| Suite | Covers |
| --- | --- |
| `check_api.py` | CORS for the dashboard (3000) and Launch Kit (3780); `GET /api/pipeline/cover/{id}` (session, ownership, 404 until painted); `coverUrl` in `GET /api/jobs`; `_publish` keeping `cover.jpg` next to `demo.mp4`; the planner asking for the poster's art direction without adding title cards |
| `check_audience.py` | Personalized demos: the models, prospect research against a fake site, the brief, the planner and launch-kit sections, the "Made for" cards, the outreach email and DM. Includes the four review cases: without an audience the prompts are byte-identical, an audience without a website, a website that times out, a private address refused before any fetch |
| `check_bookends.py` | FLUX title cards and the cover poster: cards rendered before the image call, the product name never sent to the image model, the fallbacks (FLUX failure, timeout, no key, `FEATURE_COVER=0`), `prepare_poster` |
| `check_core.py` | Vault, password hashing, netguard, request validation, private and multi-repo ingest against a fake GitHub |
| `check_music.py` | The soundtrack: the ducked mix (loudness, A/V sync, the video stream copied), fail-soft compose on fake fal responses, the final-cut wrapper |
| `check_plancheck.py` | The plan check before filming: selectors and routes verified against the inventory, the Gemini repair (faked), pacing fit to the measured voice |
| `check_review.py` | The footage review: contact sheets, sanitizing Gemini's verdicts, the retake gate |
| `check_vision.py` | The vision brief: screenshots, page insights, thumbnails, the planner's insight lines |

`LIVE=1` adds one real call to some suites (`check_bookends`, `check_music`, `check_plancheck`, `check_vision`) and
uses the keys in `.env`.

## Against a running server

These start real jobs, so the server needs `GEMINI_API_KEY` and `FAL_KEY`. Without `SIGNUP_CODE` on the server, they
create their own accounts; pass the code as the second argument when one is set.

```bash
.venv/bin/python tests/live_contract.py http://127.0.0.1:8000        # Launch Kit v2 contract, one 15 s job, about 2 min
.venv/bin/python tests/live_personalized.py http://127.0.0.1:8000    # one personalized 30 s demo, about 3 min
```
