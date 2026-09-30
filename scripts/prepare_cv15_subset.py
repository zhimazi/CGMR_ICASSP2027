from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import av
import pandas as pd


def audio_duration(path: Path) -> float:
    with av.open(str(path)) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        total = 0.0
        for frame in container.decode(stream):
            total += float(frame.samples) / float(frame.sample_rate)
        return total


def stable_rng(seed: int, *parts: str) -> random.Random:
    digest = hashlib.sha256((str(seed) + "|" + "|".join(parts)).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def index_audio(root: Path) -> dict[str, Path]:
    return {p.name: p.resolve() for p in root.rglob("*.mp3")}


def choose_rows(
    frame: pd.DataFrame,
    audio_index: dict[str, Path],
    count: int,
    seed: int,
    lang: str,
    split: str,
    used_paths: set[str],
    max_seconds: float,
) -> list[dict]:
    indices = list(range(len(frame)))
    stable_rng(seed, lang, split).shuffle(indices)
    selected: list[dict] = []
    for idx in indices:
        row = frame.iloc[idx]
        basename = Path(str(row["path"])).name
        if basename in used_paths or basename not in audio_index:
            continue
        text = str(row.get("sentence", "")).strip()
        if not text or len(text) > 200:
            continue
        path = audio_index[basename]
        try:
            duration = audio_duration(path)
        except Exception:
            continue
        if duration <= 0 or duration > max_seconds:
            continue
        used_paths.add(basename)
        selected.append(
            {
                "id": f"{lang}|{split}|{basename}",
                "language": lang,
                "split": split,
                "audio_path": str(path),
                "audio_basename": basename,
                "text": text,
                "client_id": str(row.get("client_id", "")),
                "duration_seconds": duration,
            }
        )
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Only found {len(selected)}/{count} usable rows for {lang}/{split}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment_config.json")
    parser.add_argument("--data-root", default="data/cv15")
    parser.add_argument("--output", default="experiment/manifests/micro_panel_seed2027.jsonl")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data_root = Path(args.data_root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    counts: dict[str, int] = {}
    for lang in config["old_languages"] + [config["current_language"]]:
        lang_root = data_root / lang
        audio_index = index_audio(lang_root / "extracted")
        if not audio_index:
            raise RuntimeError(f"No extracted MP3 files under {lang_root / 'extracted'}")
        train = pd.read_csv(lang_root / "train.tsv", sep="\t", keep_default_na=False)
        test = pd.read_csv(lang_root / "test.tsv", sep="\t", keep_default_na=False)
        used: set[str] = set()

        if lang in config["old_languages"]:
            role_counts = [
                ("comparator_train", config["comparator_train_per_old_language"]),
                ("er_replay", config["replay_train_per_old_language"]),
                ("cgmr_repair", config["repair_pool_per_old_language"]),
            ]
            total = sum(n for _, n in role_counts)
            pool = choose_rows(
                train,
                audio_index,
                total,
                config["seed"],
                lang,
                "train",
                used,
                config["max_audio_seconds"],
            )
            offset = 0
            for role, n_rows in role_counts:
                for rec in pool[offset : offset + n_rows]:
                    rec["role"] = role
                    records.append(rec)
                counts[f"{lang}:{role}"] = n_rows
                offset += n_rows
        else:
            pool = choose_rows(
                train,
                audio_index,
                config["current_train_rows"],
                config["seed"],
                lang,
                "train",
                used,
                config["max_audio_seconds"],
            )
            for rec in pool:
                rec["role"] = "current_train"
                records.append(rec)
            counts[f"{lang}:current_train"] = len(pool)

        test_rows = choose_rows(
            test,
            audio_index,
            config["test_rows_per_language"],
            config["seed"],
            lang,
            "test",
            used,
            config["max_audio_seconds"],
        )
        for rec in test_rows:
            rec["role"] = "test"
            records.append(rec)
        counts[f"{lang}:test"] = len(test_rows)

    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "schema": "cgmr-cv15-micro-manifest-v1",
        "seed": config["seed"],
        "source": config["dataset_release"],
        "warning": config["dataset_role"],
        "rows": len(records),
        "counts": counts,
    }
    out_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
