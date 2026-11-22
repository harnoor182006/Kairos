#!/usr/bin/env python3
"""Download, infer, validate, and remove one Track 4 source video at a time."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

try:
    from .download_fpv import youtube_video_id
except ImportError:
    from download_fpv import youtube_video_id
from schema import validate_output

EPIC_DATASETS = {"epic", "epic-kitchens-100", "epic_kitchens_100"}
OWNED_MARKER = ".kairos-stream-owned"
GIB = 1024**3
SAFE_VIDEO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _is_owned_download(video_work: Path, raw_root: Path) -> bool:
    marker = video_work / OWNED_MARKER
    try:
        ownership = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return ownership == {
        "video_id": video_work.name,
        "raw_root": str(raw_root.resolve()),
    }


def _read_manifest(path: Path, dataset_filter: str, limit: int | None):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"video_id", "duration_sec", "dataset"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must have columns: {', '.join(sorted(required))}")

        entries = []
        seen = set()
        for line, row in enumerate(reader, start=2):
            video_id = (row.get("video_id") or "").strip()
            if not SAFE_VIDEO_ID.fullmatch(video_id):
                raise ValueError(f"{path}:{line} has an unsafe video_id: {video_id!r}.")
            if video_id in seen:
                raise ValueError(f"{path}:{line} has a duplicate video_id: {video_id!r}.")
            seen.add(video_id)
            dataset = (row.get("dataset") or "").strip().lower()
            if dataset not in EPIC_DATASETS | {"fpvsum"}:
                raise ValueError(f"{path}:{line} has an unsupported dataset: {dataset!r}.")
            try:
                duration = float(row["duration_sec"])
            except (TypeError, ValueError):
                raise ValueError(f"{path}:{line} has an invalid duration_sec.")
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"{path}:{line} duration_sec must be positive and finite.")
            if dataset_filter == "all" or (
                dataset_filter == "fpvsum" and dataset == "fpvsum"
            ) or (
                dataset_filter == "epic" and dataset in EPIC_DATASETS
            ):
                entries.append((video_id, duration, dataset))

    if limit is not None:
        entries = entries[:limit]
    if not entries:
        raise ValueError("The selected manifest contains no videos.")
    return entries


def _load_valid_output(
    path: Path,
    video_id: str,
    duration: float,
    dataset: str,
    team_id: str,
):
    if not path.is_file():
        return None
    try:
        prediction = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    valid, _, _ = validate_output(
        prediction,
        expected_video_id=video_id,
        manifest_duration=duration,
        dataset=dataset,
    )
    if valid and prediction.get("team_id") == team_id:
        return prediction
    return None


def _require_models(dataset: str, weights_dir: Path, avion_root: Path) -> None:
    if dataset == "all" or dataset == "fpvsum":
        required_models = {
            "SigLIP": weights_dir / "siglip",
            "VideoMAE": weights_dir / "videomae_kinetics",
        }
        for name, directory in required_models.items():
            if not directory.is_dir() or not (directory / "config.json").is_file():
                raise FileNotFoundError(f"{name} checkpoint is missing: {directory}")
            if not any(directory.glob("*.safetensors")) and not any(
                directory.glob("pytorch_model*.bin")
            ):
                raise FileNotFoundError(f"{name} weight file is missing: {directory}")

    if dataset == "all" or dataset == "epic":
        checkpoint = weights_dir / "avion" / "avion_finetune_cls_lavila_vitb_best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"AVION checkpoint is missing: {checkpoint}")
        if not (avion_root / "avion" / "models" / "transformer.py").is_file():
            raise FileNotFoundError(f"AVION source tree is missing: {avion_root}")
        if not (weights_dir / "epic_priors.npz").is_file():
            raise FileNotFoundError(f"EPIC priors are missing: {weights_dir / 'epic_priors.npz'}")


def _download_video(
    video_id: str,
    dataset: str,
    video_work: Path,
    epic_downloader: Path,
) -> Path:
    if video_work.is_symlink() or (
        video_work.exists()
        and not _is_owned_download(video_work, video_work.parent)
    ):
        raise RuntimeError(
            f"Refusing to use or remove unowned directory: {video_work}"
        )
    video_work.mkdir(parents=True, exist_ok=True)
    (video_work / OWNED_MARKER).write_text(
        json.dumps(
            {
                "video_id": video_work.name,
                "raw_root": str(video_work.parent.resolve()),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    existing = [
        path
        for path in video_work.rglob("*")
        if path.is_file()
        and path.stem.lower() == video_id.lower()
        and path.suffix.lower() == ".mp4"
    ]
    if len(existing) > 1:
        raise RuntimeError(f"Found multiple MP4 files for {video_id} in {video_work}")
    verified_epic_downloader = epic_downloader.with_name(
        "epic_video_downloader_verify.py"
    )
    use_verified_epic_downloader = (
        dataset in EPIC_DATASETS and verified_epic_downloader.is_file()
    )
    if existing and not use_verified_epic_downloader:
        return existing[0]

    if dataset in EPIC_DATASETS:
        if not epic_downloader.is_file():
            raise FileNotFoundError(
                f"Official EPIC downloader not found: {epic_downloader}"
            )
        if use_verified_epic_downloader:
            command = [
                sys.executable,
                str(verified_epic_downloader),
                "--specific-videos",
                video_id,
                "--output-path",
                str(video_work),
                "--verify-ssl",
                "--no-md5-progress",
                "--chunk-size",
                "262144",
                "--max-retries",
                "10",
            ]
        else:
            command = [
                sys.executable,
                str(epic_downloader),
                "--videos",
                "--specific-videos",
                video_id,
                "--output-path",
                str(video_work),
            ]
    else:
        youtube_id = youtube_video_id(video_id)
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--continue",
            "-f",
            "bestvideo[height<=480]+bestaudio/best[height<=480]",
            "--merge-output-format",
            "mp4",
            "-o",
            str(video_work / (video_id + ".%(ext)s")),
            f"https://www.youtube.com/watch?v={youtube_id}",
        ]

    subprocess.run(
        command,
        check=True,
        cwd=epic_downloader.parent if dataset in EPIC_DATASETS else video_work,
    )
    downloaded = [
        path
        for path in video_work.rglob("*")
        if path.is_file()
        and path.stem.lower() == video_id.lower()
        and path.suffix.lower() == ".mp4"
    ]
    if len(downloaded) != 1:
        raise FileNotFoundError(
            f"Expected one downloaded MP4 for {video_id}; found {len(downloaded)}."
        )
    return downloaded[0]


def _remove_download(video_work: Path, raw_root: Path) -> None:
    resolved_work = video_work.resolve()
    resolved_root = raw_root.resolve()
    if (
        resolved_work.parent != resolved_root
        or video_work.is_symlink()
        or not _is_owned_download(video_work, raw_root)
    ):
        raise RuntimeError(f"Refusing to remove unowned download directory: {video_work}")
    shutil.rmtree(resolved_work)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream manifest videos through inference and validation."
    )
    parser.add_argument("manifest", nargs="?", type=Path, default=ROOT / "manifest.csv")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--epic-downloader",
        type=Path,
        help="Path to epic_downloader.py from the official EPIC downloader repo.",
    )
    parser.add_argument("--dataset", choices=("all", "epic", "fpvsum"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-free-gb", type=float, default=12.0)
    parser.add_argument("--team-id", default=os.environ.get("TEAM_ID_STR", "TEAM_ID"))
    parser.add_argument("--keep-videos", action="store_true")
    prefetch_group = parser.add_mutually_exclusive_group()
    prefetch_group.add_argument(
        "--prefetch",
        dest="prefetch",
        action="store_true",
        help="Download one next source while inference runs (default).",
    )
    prefetch_group.add_argument(
        "--no-prefetch",
        dest="prefetch",
        action="store_false",
        help="Disable overlap between downloading and inference.",
    )
    parser.set_defaults(prefetch=True)
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        parser.error(f"Manifest not found: {args.manifest}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 1:
        parser.error("--min-free-gb must be at least 1")
    if not args.team_id.strip():
        parser.error("--team-id must not be empty")

    work_dir = args.work_dir.expanduser().resolve()
    weights_dir = Path(
        os.environ.get("KAIROS_WEIGHTS_DIR", work_dir / "weights")
    ).expanduser().resolve()
    avion_root = Path(
        os.environ.get("KAIROS_AVION_ROOT", ROOT / "third_party" / "AVION")
    ).expanduser().resolve()
    epic_downloader = (
        args.epic_downloader.expanduser().resolve()
        if args.epic_downloader
        else work_dir / "epic-kitchens-download-scripts" / "epic_downloader.py"
    )
    raw_root = work_dir / "downloads" / "raw"
    state_dir = work_dir / "state"
    output_dir = args.output_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        entries = _read_manifest(args.manifest, args.dataset, args.limit)
        _require_models(args.dataset, weights_dir, avion_root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    env = os.environ.copy()
    env["TEAM_ID_STR"] = args.team_id
    env["KAIROS_WEIGHTS_DIR"] = str(weights_dir)
    env["KAIROS_AVION_ROOT"] = str(avion_root)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    runner = ROOT / "code" / "run_video.py"
    processed = 0
    download_pool = ThreadPoolExecutor(max_workers=1)
    prefetch_task: tuple[str, Future[Path]] | None = None
    print(
        "Prefetch: "
        + (
            f"enabled (one next source; reserve {args.min_free_gb:.1f} GB)"
            if args.prefetch
            else "disabled"
        )
    )

    for index, (video_id, duration, dataset) in enumerate(entries, start=1):
        output_path = output_dir / f"{video_id}.json"
        done_marker = state_dir / f"{video_id}.done"
        video_work = raw_root / video_id
        existing = _load_valid_output(
            output_path, video_id, duration, dataset, args.team_id
        )
        if existing is not None and done_marker.is_file():
            print(f"[{index}/{len(entries)}] Already valid: {video_id}")
            if not args.keep_videos and video_work.exists():
                _remove_download(video_work, raw_root)
            continue

        free_gb = shutil.disk_usage(work_dir).free / GIB
        if free_gb < args.min_free_gb:
            print(
                f"STOP: only {free_gb:.1f} GB free under {work_dir}; "
                f"keeping {args.min_free_gb:.1f} GB reserved."
            )
            download_pool.shutdown(wait=True, cancel_futures=True)
            return 1

        cache_path = weights_dir / "epic_cache" / f"{video_id}.npz"
        cache_existed = cache_path.is_file()
        candidate_path = video_work / "candidate.json"
        print(
            f"[{index}/{len(entries)}] Downloading {video_id} "
            f"({dataset}, {duration:.2f}s); {free_gb:.1f} GB free."
        )
        try:
            if prefetch_task is not None:
                prefetched_video_id, download_future = prefetch_task
                if prefetched_video_id != video_id:
                    raise RuntimeError(
                        f"Expected prefetched {prefetched_video_id}, "
                        f"but manifest reached {video_id}."
                    )
                source = download_future.result()
                prefetch_task = None
            else:
                source = _download_video(
                    video_id, dataset, video_work, epic_downloader
                )

            if args.prefetch and not args.keep_videos and prefetch_task is None:
                prefetch_free_gb = shutil.disk_usage(work_dir).free / GIB
                if prefetch_free_gb >= args.min_free_gb + 8:
                    for next_id, next_duration, next_dataset in entries[index:]:
                        next_output = output_dir / f"{next_id}.json"
                        next_done = state_dir / f"{next_id}.done"
                        if (
                            _load_valid_output(
                                next_output,
                                next_id,
                                next_duration,
                                next_dataset,
                                args.team_id,
                            )
                            is not None
                            and next_done.is_file()
                        ):
                            continue
                        next_work = raw_root / next_id
                        prefetch_task = (
                            next_id,
                            download_pool.submit(
                                _download_video,
                                next_id,
                                next_dataset,
                                next_work,
                                epic_downloader,
                            ),
                        )
                        print(
                            f"[{index + 1}/{len(entries)}] Prefetching {next_id} "
                            f"while processing {video_id}."
                        )
                        break

            completed = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    str(source),
                    str(candidate_path),
                    "--dataset",
                    dataset,
                    "--manifest-duration",
                    str(duration),
                ],
                check=False,
                env=env,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Inference failed for {video_id} with exit code "
                    f"{completed.returncode}; source retained at {source}"
                )

            prediction = _load_valid_output(
                candidate_path, video_id, duration, dataset, args.team_id
            )
            if prediction is None:
                raise RuntimeError(
                    f"Output failed manifest validation; source retained at {source}"
                )

            staged_output = output_dir / f".{video_id}.json.tmp"
            staged_output.write_text(
                candidate_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            if _load_valid_output(
                staged_output, video_id, duration, dataset, args.team_id
            ) is None:
                raise RuntimeError(
                    f"Could not stage a valid output for {video_id}; "
                    f"source retained at {source}"
                )
            os.replace(staged_output, output_path)
            done_marker.write_text("validated\n", encoding="ascii")
            candidate_path.unlink()
            if cache_path.is_file() and not cache_existed:
                cache_path.unlink()
            if not args.keep_videos:
                _remove_download(video_work, raw_root)
            processed += 1
            print(f"[{index}/{len(entries)}] Validated and retained {output_path}")
        except Exception as exc:
            print(f"STOP: {exc}", file=sys.stderr)
            download_pool.shutdown(wait=True, cancel_futures=True)
            return 1

    download_pool.shutdown(wait=True, cancel_futures=True)

    if args.dataset == "all" and args.limit is None:
        validator = ROOT / "validate.py"
        result = subprocess.run(
            [
                sys.executable,
                str(validator),
                "--dir",
                str(output_dir),
                "--manifest",
                str(args.manifest.resolve()),
            ],
            check=False,
            env=env,
        )
        if result.returncode:
            return result.returncode

    print(f"Stream complete: {processed} new predictions; videos kept: {args.keep_videos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
