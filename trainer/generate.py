"""Stage 1: generate wake word voice samples with the OpenAI TTS API.

Writes 16 kHz mono 16-bit WAV files to paths.generated_samples. Generation is
resumable: existing files are counted and only the remainder is generated.
"""

from __future__ import annotations

import itertools
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from trainer import audio


@dataclass
class Job:
    index: int
    text: str
    voice: str
    instructions: str
    speed: float
    path: Path


def build_jobs(cfg: dict) -> list[Job]:
    oai = cfg["openai"]
    out_dir = Path(cfg["paths"]["generated_samples"])
    phrases = [cfg["wake_word"]] + list(cfg.get("phonetic_spellings") or [])
    lo, hi = (list(oai["speed_range"]) + [1.0, 1.0])[:2]

    rng = random.Random(1234)  # fixed seed so re-runs plan the same jobs
    combos = itertools.cycle(
        itertools.product(oai["voices"], oai["instructions"], phrases)
    )
    jobs = []
    for i in range(int(oai["num_samples"])):
        voice, instructions, text = next(combos)
        jobs.append(
            Job(
                index=i,
                text=text,
                voice=voice,
                instructions=instructions,
                speed=rng.uniform(float(lo), float(hi)),
                path=out_dir / f"openai_{voice}_{i:05d}.wav",
            )
        )
    return jobs


def synthesize(client, model: str, job: Job) -> None:
    kwargs = {
        "model": model,
        "voice": job.voice,
        "input": job.text,
        "response_format": "wav",
    }
    # gpt-4o-* models are steered with instructions; tts-1 models use `speed`.
    if model.startswith("tts-1"):
        kwargs["speed"] = round(min(max(job.speed, 0.25), 4.0), 2)
    else:
        kwargs["instructions"] = job.instructions

    response = client.audio.speech.create(**kwargs)
    clip = audio.decode_bytes(response.content)
    if not model.startswith("tts-1"):
        clip = audio.speed_perturb(clip, job.speed)
    clip = audio.peak_normalize(clip)
    audio.write_wav(job.path, clip)


def run(cfg: dict, dry_run: bool = False) -> None:
    oai = cfg["openai"]
    jobs = [job for job in build_jobs(cfg) if not job.path.exists()]

    if dry_run:
        print(f"Would generate {len(jobs)} samples with model {oai['model']!r}:")
        for job in jobs[:10]:
            print(
                f"  {job.path.name}: voice={job.voice} speed={job.speed:.2f} "
                f"text={job.text!r} instructions={job.instructions[:50]!r}"
            )
        if len(jobs) > 10:
            print(f"  ... and {len(jobs) - 10} more")
        return

    if not jobs:
        print("All samples already generated — nothing to do.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            "OPENAI_API_KEY is not set. Export it first:\n"
            "  export OPENAI_API_KEY=sk-..."
        )

    from openai import OpenAI  # deferred so --dry-run works without the key

    client = OpenAI()
    failures = 0
    with ThreadPoolExecutor(max_workers=int(oai["concurrency"])) as pool:
        futures = {
            pool.submit(synthesize, client, oai["model"], job): job for job in jobs
        }
        for future in tqdm(as_completed(futures), total=len(jobs), desc="Generating"):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # keep going; failed clips can be retried
                failures += 1
                tqdm.write(f"FAILED {job.path.name}: {exc}")

    done = len(jobs) - failures
    print(f"Generated {done} samples in {cfg['paths']['generated_samples']}")
    if failures:
        print(f"{failures} samples failed — re-run `python -m trainer generate` to retry them.")
