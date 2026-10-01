from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import av
import jiwer
import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.generation.utils import GenerationMixin

from cgmr.objective import restricted_score_gradient, restricted_target_cross_entropy
from cgmr.projection import project_language_cohort
from cgmr.states import classify_state


LOCALE_TOKENS = {
    "fy-NL": "<|fy-nl|>",
    "tr": "<|tr|>",
    "ja": "<|ja|>",
    "ia": "<|ia|>",
}
CUSTOM_LOCALE_TOKENS = ["<|fy-nl|>", "<|ia|>"]


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def decode_audio(path: str, target_rate: int = 16000) -> np.ndarray:
    chunks: list[np.ndarray] = []
    source_rate = None
    with av.open(path) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        for frame in container.decode(stream):
            source_rate = frame.sample_rate
            arr = frame.to_ndarray()
            if arr.ndim == 2:
                arr = arr.astype(np.float32).mean(axis=0)
            else:
                arr = arr.astype(np.float32).reshape(-1)
            if np.issubdtype(frame.to_ndarray().dtype, np.integer):
                info = np.iinfo(frame.to_ndarray().dtype)
                arr = arr / max(abs(info.min), info.max)
            chunks.append(arr)
    if not chunks or source_rate is None:
        raise RuntimeError(f"No audio decoded from {path}")
    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    if source_rate != target_rate:
        divisor = math.gcd(int(source_rate), target_rate)
        audio = resample_poly(audio, target_rate // divisor, int(source_rate) // divisor).astype(np.float32)
    return audio


def normalize_text(tokenizer, text: str) -> str:
    try:
        return tokenizer.normalize(text).strip()
    except Exception:
        return " ".join(text.lower().strip().split())


def edit_stats(reference: str, hypothesis: str, language: str, tokenizer) -> dict:
    ref = normalize_text(tokenizer, reference)
    hyp = normalize_text(tokenizer, hypothesis)
    if language == "ja":
        ref_units = list(ref.replace(" ", ""))
        hyp_units = list(hyp.replace(" ", ""))
        metric = "CER"
    else:
        ref_units = ref.split()
        hyp_units = hyp.split()
        metric = "WER"
    safe_ref = " ".join(ref_units) if ref_units else "<empty>"
    safe_hyp = " ".join(hyp_units) if hyp_units else "<empty>"
    result = jiwer.process_words(safe_ref, safe_hyp)
    return {
        "reference_normalized": ref,
        "hypothesis_normalized": hyp,
        "error_unit": metric,
        "substitutions": int(result.substitutions),
        "deletions": int(result.deletions),
        "insertions": int(result.insertions),
        "edits": int(result.substitutions + result.deletions + result.insertions),
        "reference_units": int(len(ref_units)),
    }


def autocast_context():
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def configure_locale_tokens(processor: WhisperProcessor) -> int:
    """Add the two CL-MASR locale tokens missing from stock Whisper.

    This follows the official CL-MASR ProgressiveWhisper recipe.  The new
    embeddings are resized before PEFT freezes the base model, so the paper's
    LoRA-only trainable-parameter count is preserved.
    """
    tokenizer = processor.tokenizer
    missing = [token for token in CUSTOM_LOCALE_TOKENS if token not in tokenizer.get_vocab()]
    tokenizer.add_tokens(missing, special_tokens=True)
    return len(tokenizer)


def locale_prompt_ids(processor: WhisperProcessor, language: str) -> list[int]:
    tokenizer = processor.tokenizer
    tokens = [
        "<|startoftranscript|>",
        LOCALE_TOKENS[language],
        "<|transcribe|>",
        "<|notimestamps|>",
    ]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if any(i == tokenizer.unk_token_id for i in ids):
        raise RuntimeError(f"Missing locale prompt token for {language}: {tokens}")
    return ids


def locale_forced_decoder_ids(processor: WhisperProcessor, language: str) -> list[list[int]]:
    prompt = locale_prompt_ids(processor, language)
    return [[idx, token_id] for idx, token_id in enumerate(prompt[1:], start=1)]


def build_model(config: dict, model_path: str, device: torch.device, vocab_size: int):
    base = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    base.config.forced_decoder_ids = None
    base.generation_config.language = None
    base.generation_config.task = None
    base.generation_config.forced_decoder_ids = None
    if base.get_input_embeddings().num_embeddings != vocab_size:
        base.resize_token_embeddings(vocab_size, mean_resizing=False)
    target_regex = (
        r"model\.decoder\.layers\.\d+\.(?:self_attn|encoder_attn)\."
        r"(?:q_proj|k_proj|v_proj|out_proj)|model\.decoder\.layers\.\d+\.(?:fc1|fc2)"
    )
    lora = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        target_modules=target_regex,
    )
    model = get_peft_model(base, lora)
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != config["expected_trainable_parameters"]:
        raise RuntimeError(
            f"LoRA trainable count {trainable} != expected {config['expected_trainable_parameters']}"
        )
    return model, trainable


def load_saved_model(
    config: dict, model_path: str, device: torch.device, vocab_size: int, adapter_path: Path
):
    base = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    base.config.forced_decoder_ids = None
    base.generation_config.language = None
    base.generation_config.task = None
    base.generation_config.forced_decoder_ids = None
    if base.get_input_embeddings().num_embeddings != vocab_size:
        base.resize_token_embeddings(vocab_size, mean_resizing=False)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != config["expected_trainable_parameters"]:
        raise RuntimeError(
            f"Loaded LoRA trainable count {trainable} != expected "
            f"{config['expected_trainable_parameters']}"
        )
    return model, trainable


def cache_key(record: dict) -> str:
    return hashlib.sha256(record["id"].encode("utf-8")).hexdigest()


def labels_for(processor: WhisperProcessor, text: str, language: str) -> torch.Tensor:
    tokenizer = processor.tokenizer
    prompt = locale_prompt_ids(processor, language)
    text_ids = tokenizer(text, add_special_tokens=False).input_ids
    # Whisper's loss shifts labels internally and inserts decoder_start_token_id,
    # so labels begin with the locale token rather than duplicating SOT.
    return torch.tensor(prompt[1:] + text_ids + [tokenizer.eos_token_id], dtype=torch.long)


def ensure_feature_cache(
    records: list[dict], processor: WhisperProcessor, cache_dir: Path, log
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for idx, record in enumerate(records, start=1):
        path = cache_dir / f"{cache_key(record)}.pt"
        if path.exists():
            continue
        audio = decode_audio(record["audio_path"])
        features = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features[0].half()
        labels = labels_for(processor, record["text"], record["language"])
        torch.save({"input_features": features, "labels": labels}, path)
        if idx % 50 == 0:
            log(f"feature-cache {idx}/{len(records)} elapsed={time.time()-started:.1f}s")


def load_cached(record: dict, cache_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(cache_dir / f"{cache_key(record)}.pt", map_location="cpu")
    return payload["input_features"], payload["labels"]


def train_ce_steps(
    model,
    records_by_role: dict[str, list[dict]],
    cache_dir: Path,
    steps: int,
    lr: float,
    grad_accum: int,
    seed: int,
    mode: str,
    log,
) -> list[dict]:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    model.train()
    history: list[dict] = []
    device = next(model.parameters()).device
    old = records_by_role.get("comparator_train", [])
    current = records_by_role.get("current_train", [])
    replay = records_by_role.get("er_replay", [])
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, steps + 1):
        losses = []
        for micro in range(grad_accum):
            if mode == "comparator":
                record = rng.choice(old)
            elif mode == "er":
                record = rng.choice(current if micro < grad_accum // 2 else replay)
            else:
                raise ValueError(mode)
            features, labels = load_cached(record, cache_dir)
            features = features.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            with autocast_context():
                loss = model(input_features=features, labels=labels).loss
            (loss / grad_accum).backward()
            losses.append(float(loss.detach().cpu()))
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        row = {
            "step": step,
            "loss": float(np.mean(losses)),
            "elapsed_seconds": time.time() - start,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        history.append(row)
        if step == 1 or step % 8 == 0 or step == steps:
            log(
                f"{mode} step={step}/{steps} loss={row['loss']:.4f} "
                f"elapsed={row['elapsed_seconds']:.1f}s alloc={row['peak_cuda_allocated_gib']:.2f}GiB"
            )
    return history


@torch.inference_mode()
def decode_one(model, processor, record: dict, cache_dir: Path, config: dict) -> dict:
    features, _ = load_cached(record, cache_dir)
    device = next(model.parameters()).device
    model.eval()
    model.config.use_cache = True
    generated = model.generate(
        input_features=features.unsqueeze(0).to(device),
        forced_decoder_ids=locale_forced_decoder_ids(processor, record["language"]),
        num_beams=config["beam_size"],
        do_sample=False,
        max_new_tokens=config["max_new_tokens"],
    )
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    stats = edit_stats(record["text"], text, record["language"], processor.tokenizer)
    model.config.use_cache = False
    return {
        "id": record["id"],
        "client_id": record.get("client_id", ""),
        "language": record["language"],
        "reference": record["text"],
        "hypothesis": text,
        **stats,
    }


def evaluate(model, processor, records: list[dict], cache_dir: Path, config: dict, log):
    started = time.time()
    rows = []
    for idx, record in enumerate(records, start=1):
        rows.append(decode_one(model, processor, record, cache_dir, config))
        if idx % 12 == 0:
            log(f"decode {idx}/{len(records)} elapsed={time.time()-started:.1f}s")
    per_language = {}
    for lang in sorted({r["language"] for r in rows}):
        subset = [r for r in rows if r["language"] == lang]
        edits = sum(r["edits"] for r in subset)
        units = sum(r["reference_units"] for r in subset)
        per_language[lang] = {
            "rows": len(subset),
            "error_unit": subset[0]["error_unit"],
            "edits": edits,
            "reference_units": units,
            "error_percent": 100.0 * edits / max(units, 1),
        }
    old_rates = [per_language[x]["error_percent"] for x in config["old_languages"]]
    summary = {
        "old_macro_error_percent": float(np.mean(old_rates)),
        "current_error_percent": per_language[config["current_language"]]["error_percent"],
        "per_language": per_language,
        "runtime_seconds": time.time() - started,
    }
    return summary, rows


@torch.inference_mode()
def beam_candidates(model, processor, record: dict, cache_dir: Path, config: dict) -> list[dict]:
    features, _ = load_cached(record, cache_dir)
    device = next(model.parameters()).device
    model.eval()
    model.config.use_cache = True
    # Whisper's custom long-form generate path duplicates the top hypothesis
    # when num_return_sequences > 1.  Invoke the standard generation mixin on
    # the LoRA-injected base model to obtain the actual K distinct beams.
    generation_model = model.get_base_model() if isinstance(model, PeftModel) else model
    prompt = torch.tensor(
        [locale_prompt_ids(processor, record["language"])],
        dtype=torch.long,
        device=device,
    )
    sequences = GenerationMixin.generate(
        generation_model,
        inputs=features.unsqueeze(0).to(device),
        decoder_input_ids=prompt,
        num_beams=config["beam_size"],
        num_return_sequences=config["beam_size"],
        do_sample=False,
        max_new_tokens=config["max_new_tokens"],
    )
    prompt_length = len(locale_prompt_ids(processor, record["language"]))
    eos = processor.tokenizer.eos_token_id
    trimmed_sequences = []
    for sequence in sequences.cpu().tolist():
        continuation = sequence[prompt_length:]
        if eos in continuation:
            end = prompt_length + continuation.index(eos) + 1
            sequence = sequence[:end]
        trimmed_sequences.append(sequence)
    raw_scores = [
        float(
            differentiable_sequence_score(model, features, sequence, prompt_length)
            .float()
            .cpu()
        )
        for sequence in trimmed_sequences
    ]
    texts = processor.batch_decode(trimmed_sequences, skip_special_tokens=True)
    dedup = {}
    for seq, score, text in zip(trimmed_sequences, raw_scores, texts):
        normalized = normalize_text(processor.tokenizer, text)
        candidate = {
            "text": text,
            "normalized": normalized,
            "token_ids": seq,
            "prompt_length": prompt_length,
            "raw_log_score": float(score),
            **edit_stats(record["text"], text, record["language"], processor.tokenizer),
        }
        if normalized not in dedup or candidate["raw_log_score"] > dedup[normalized]["raw_log_score"]:
            dedup[normalized] = candidate
    model.config.use_cache = False
    return sorted(dedup.values(), key=lambda x: x["raw_log_score"], reverse=True)


def build_cgmr_targets(
    model,
    processor,
    repair_records: list[dict],
    comparator_rows: dict[str, dict],
    cache_dir: Path,
    config: dict,
    log,
):
    diagnosed: list[dict] = []
    all_rows: list[dict] = []
    started = time.time()
    for idx, record in enumerate(repair_records, start=1):
        candidates = beam_candidates(model, processor, record, cache_dir, config)
        budget = comparator_rows[record["id"]]["edits"]
        top_edits = candidates[0]["edits"]
        oracle_edits = min(c["edits"] for c in candidates)
        state = classify_state(budget, top_edits, oracle_edits)
        row = {
            "id": record["id"],
            "language": record["language"],
            "reference": record["text"],
            "comparator_budget_edits": budget,
            "er_top_edits": top_edits,
            "er_oracle_edits": oracle_edits,
            "state": state,
            "diagnosed": state == "A",
            "candidates": candidates,
        }
        all_rows.append(row)
        if row["diagnosed"]:
            diagnosed.append(row)
        if idx % 8 == 0:
            log(f"cgmr-beams {idx}/{len(repair_records)} diagnosed={len(diagnosed)} elapsed={time.time()-started:.1f}s")

    by_language: dict[str, list[dict]] = defaultdict(list)
    for row in diagnosed:
        by_language[row["language"]].append(row)
    lambdas = {}
    for lang, rows in by_language.items():
        budget = float(sum(r["comparator_budget_edits"] for r in rows))
        projection_rows = [
            (
                [c["raw_log_score"] for c in r["candidates"]],
                [c["edits"] for c in r["candidates"]],
            )
            for r in rows
        ]
        projection = project_language_cohort(projection_rows, budget)
        lambdas[lang] = {
            "lambda": projection.lambda_value,
            "budget_edits": projection.comparator_budget,
            "risk_at_zero": projection.risk_before,
            "projected_risk": projection.risk_after,
            "rows": len(rows),
        }
        for r, q in zip(rows, projection.posteriors):
            r["target_q"] = q.tolist()
            r["target_expected_length"] = float(
                q @ np.array([max(1, len(c["token_ids"])) for c in r["candidates"]])
            )
    return diagnosed, all_rows, lambdas


def differentiable_sequence_score(
    model, features: torch.Tensor, token_ids: list[int], prompt_length: int
) -> torch.Tensor:
    device = next(model.parameters()).device
    sequence = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    decoder_input = sequence[:, :-1]
    target = sequence[:, 1:]
    with autocast_context():
        logits = model(
            input_features=features.unsqueeze(0).to(device),
            decoder_input_ids=decoder_input,
            use_cache=False,
        ).logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return token_log_probs[:, prompt_length - 1 :].sum()


def train_cgmr(
    model,
    targets: list[dict],
    repair_record_map: dict[str, dict],
    current_records: list[dict],
    cache_dir: Path,
    steps: int,
    lr: float,
    seed: int,
    log,
) -> list[dict]:
    if not targets:
        log("cgmr diagnosed cohort is empty; method-defined no-op")
        return []
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    model.train()
    device = next(model.parameters()).device
    history = []
    started = time.time()
    for step in range(1, steps + 1):
        target = rng.choice(targets)
        old_record = repair_record_map[target["id"]]
        old_features, _ = load_cached(old_record, cache_dir)
        # Obtain the current restricted posterior without retaining eight decoder
        # graphs at once.  For cross-entropy H(q, softmax(s)), the exact score
        # gradient is softmax(s) - q, so each candidate can then be backpropagated
        # independently.  This is mathematically equivalent and materially lowers
        # peak memory on an 8 GiB RTX 4060.
        with torch.no_grad():
            detached_scores = torch.stack(
                [
                    differentiable_sequence_score(
                        model, old_features, c["token_ids"], c["prompt_length"]
                    )
                    for c in target["candidates"]
                ]
            )
        q = torch.tensor(target["target_q"], dtype=torch.float32, device=device)
        expected_length = max(target["target_expected_length"], 1.0)
        old_loss = restricted_target_cross_entropy(
            detached_scores, q, normalizer=expected_length
        )

        optimizer.zero_grad(set_to_none=True)
        coefficients = restricted_score_gradient(
            detached_scores, q, normalizer=expected_length
        )
        for coefficient, candidate in zip(coefficients, target["candidates"]):
            score = differentiable_sequence_score(
                model, old_features, candidate["token_ids"], candidate["prompt_length"]
            )
            (0.5 * coefficient * score).backward()

        current = rng.choice(current_records)
        cur_features, cur_labels = load_cached(current, cache_dir)
        with autocast_context():
            cur_loss = model(
                input_features=cur_features.unsqueeze(0).to(device),
                labels=cur_labels.unsqueeze(0).to(device),
            ).loss
        loss = 0.5 * old_loss + 0.5 * cur_loss
        (0.5 * cur_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0
        )
        optimizer.step()
        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "old_cross_entropy": float(old_loss.detach().cpu()),
            "current_ce": float(cur_loss.detach().cpu()),
            "elapsed_seconds": time.time() - started,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        history.append(row)
        if step == 1 or step % 4 == 0 or step == steps:
            log(
                f"cgmr step={step}/{steps} loss={row['loss']:.4f} "
                f"elapsed={row['elapsed_seconds']:.1f}s alloc={row['peak_cuda_allocated_gib']:.2f}GiB"
            )
    return history


def paired_bootstrap(
    left_rows: list[dict], right_rows: list[dict], old_languages: list[str], seed: int, n: int
) -> dict:
    left = {r["id"]: r for r in left_rows}
    right = {r["id"]: r for r in right_rows}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        rates = []
        for lang in old_languages:
            ids = [r["id"] for r in left_rows if r["language"] == lang]
            draw = rng.choice(ids, size=len(ids), replace=True)
            l_edits = sum(left[x]["edits"] for x in draw)
            r_edits = sum(right[x]["edits"] for x in draw)
            units = sum(left[x]["reference_units"] for x in draw)
            rates.append(100.0 * (l_edits - r_edits) / max(units, 1))
        samples.append(float(np.mean(rates)))
    arr = np.array(samples)
    return {
        "left_minus_right_gain_pp": float(arr.mean()),
        "paired_bootstrap_95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "probability_positive": float(np.mean(arr > 0)),
        "resamples": n,
    }


def write_prediction_csv(path: Path, method_rows: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "id", "client_id", "language", "error_unit", "reference", "hypothesis",
        "substitutions", "deletions", "insertions", "edits", "reference_units",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, rows in method_rows.items():
            for row in rows:
                writer.writerow({k: method if k == "method" else row.get(k, "") for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment_config.json")
    parser.add_argument("--manifest", default="experiment/manifests/micro_panel_seed2027.jsonl")
    parser.add_argument("--model-path", default="models/whisper-small")
    parser.add_argument("--output-dir", default="experiment/output")
    parser.add_argument(
        "--resume-er-cgmr",
        action="store_true",
        help="Reuse saved C/ER adapters and endpoints after a post-ER interruption.",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    records = read_jsonl(Path(args.manifest))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    set_determinism(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda")
    run_started = time.time()
    log(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    processor = WhisperProcessor.from_pretrained(args.model_path)
    tokenizer_size = configure_locale_tokens(processor)
    cache_dir = output_dir / "feature_cache"
    ensure_feature_cache(records, processor, cache_dir, log)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["role"]].append(record)
    test_records = grouped["test"]
    repair_records = grouped["cgmr_repair"]
    if args.resume_er_cgmr:
        log("resuming from saved C and ER artifacts; no C/ER retraining")
        consistency = json.loads(
            (output_dir / "small_consistency_test.json").read_text(encoding="utf-8")
        )
        c_payload = json.loads((output_dir / "C_endpoint.json").read_text(encoding="utf-8"))
        c_summary, c_rows = c_payload["summary"], c_payload["rows"]
        model, trainable = load_saved_model(
            config, args.model_path, device, tokenizer_size, output_dir / "adapters" / "C"
        )
        log(f"loaded C adapter; trainable LoRA parameters={trainable}")
        comparator_repair_rows = {
            record["id"]: decode_one(model, processor, record, cache_dir, config)
            for record in repair_records
        }
        save_json(output_dir / "C_repair_rows.json", comparator_repair_rows)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        er_payload = json.loads((output_dir / "ER_endpoint.json").read_text(encoding="utf-8"))
        er_summary, er_rows = er_payload["summary"], er_payload["rows"]
        model, trainable = load_saved_model(
            config, args.model_path, device, tokenizer_size, output_dir / "adapters" / "ER"
        )
        log(f"loaded ER adapter; trainable LoRA parameters={trainable}")
        c_history = [{"status": "completed_before_resume", "source": "run.log"}]
        er_history = [{"status": "completed_before_resume", "source": "run.log"}]
    else:
        model, trainable = build_model(config, args.model_path, device, tokenizer_size)
        log(f"model loaded; trainable LoRA parameters={trainable}")

        # Deterministic small-scale consistency gate.
        repeat_subset = sorted(test_records, key=lambda x: x["id"])[:8]
        repeat_a = [decode_one(model, processor, r, cache_dir, config) for r in repeat_subset]
        repeat_b = [decode_one(model, processor, r, cache_dir, config) for r in repeat_subset]
        repeat_rate = float(
            np.mean([a["hypothesis"] == b["hypothesis"] for a, b in zip(repeat_a, repeat_b)])
        )
        consistency = {
            "rows": len(repeat_subset),
            "exact_repeat_rate": repeat_rate,
            "lora_trainable_parameters": trainable,
            "expected_lora_trainable_parameters": config["expected_trainable_parameters"],
            "passed": repeat_rate == 1.0
            and trainable == config["expected_trainable_parameters"],
        }
        save_json(output_dir / "small_consistency_test.json", consistency)
        if not consistency["passed"]:
            raise RuntimeError(f"Consistency gate failed: {consistency}")
        log(f"small consistency gate passed: exact_repeat_rate={repeat_rate:.3f}")

        torch.cuda.reset_peak_memory_stats()
        c_history = train_ce_steps(
            model,
            grouped,
            cache_dir,
            config["comparator_steps"],
            config["adaptation_learning_rate"],
            config["gradient_accumulation"],
            config["seed"],
            "comparator",
            log,
        )
        model.save_pretrained(output_dir / "adapters" / "C")
        c_summary, c_rows = evaluate(model, processor, test_records, cache_dir, config, log)
        comparator_repair_rows = {
            record["id"]: decode_one(model, processor, record, cache_dir, config)
            for record in repair_records
        }
        save_json(output_dir / "C_repair_rows.json", comparator_repair_rows)
        save_json(output_dir / "C_endpoint.json", {"summary": c_summary, "rows": c_rows})

        torch.cuda.reset_peak_memory_stats()
        er_history = train_ce_steps(
            model,
            grouped,
            cache_dir,
            config["er_steps"],
            config["adaptation_learning_rate"],
            config["gradient_accumulation"],
            config["seed"] + 1,
            "er",
            log,
        )
        model.save_pretrained(output_dir / "adapters" / "ER")
        er_summary, er_rows = evaluate(model, processor, test_records, cache_dir, config, log)
        save_json(output_dir / "ER_endpoint.json", {"summary": er_summary, "rows": er_rows})

    targets, diagnostic_rows, lambdas = build_cgmr_targets(
        model,
        processor,
        repair_records,
        comparator_repair_rows,
        cache_dir,
        config,
        log,
    )
    save_json(
        output_dir / "ER_to_CGMR_targets.json",
        {"lambdas": lambdas, "diagnosed_count": len(targets), "rows": diagnostic_rows},
    )
    repair_record_map = {r["id"]: r for r in repair_records}
    torch.cuda.reset_peak_memory_stats()
    cgmr_history = train_cgmr(
        model,
        targets,
        repair_record_map,
        grouped["current_train"],
        cache_dir,
        config["cgmr_steps"],
        config["repair_learning_rate"],
        config["seed"] + 2,
        log,
    )
    model.save_pretrained(
        output_dir / "adapters" / "ER_to_CGMR", save_embedding_layers=False
    )
    cgmr_summary, cgmr_rows = evaluate(model, processor, test_records, cache_dir, config, log)
    save_json(output_dir / "ER_to_CGMR_endpoint.json", {"summary": cgmr_summary, "rows": cgmr_rows})

    old_test_ids = {r["id"] for r in test_records if r["language"] in config["old_languages"]}
    c_old = [r for r in c_rows if r["id"] in old_test_ids]
    er_old = [r for r in er_rows if r["id"] in old_test_ids]
    cgmr_old = [r for r in cgmr_rows if r["id"] in old_test_ids]
    uncertainty = {
        "C_vs_ER": paired_bootstrap(
            c_old, er_old, config["old_languages"], config["seed"] + 11, config["bootstrap_resamples"]
        ),
        "ER_vs_ER_to_CGMR": paired_bootstrap(
            er_old, cgmr_old, config["old_languages"], config["seed"] + 12, config["bootstrap_resamples"]
        ),
    }
    save_json(output_dir / "paired_uncertainty.json", uncertainty)
    write_prediction_csv(
        output_dir / "predictions_seed2027.csv",
        {"C": c_rows, "ER": er_rows, "ER_to_CGMR": cgmr_rows},
    )
    save_json(
        output_dir / "training_history.json",
        {"C": c_history, "ER": er_history, "ER_to_CGMR": cgmr_history},
    )
    final = {
        "schema": "cgmr-independent-micro-validation-v1",
        "config": config,
        "consistency": consistency,
        "endpoints": {"C": c_summary, "ER": er_summary, "ER_to_CGMR": cgmr_summary},
        "cgmr": {"diagnosed_count": len(targets), "lambdas": lambdas},
        "paired_uncertainty": uncertainty,
        "runtime_seconds": time.time() - run_started,
        "resumed_from_saved_er": bool(args.resume_er_cgmr),
        "interpretation_guardrail": config["dataset_role"],
    }
    save_json(output_dir / "final_summary.json", final)
    log(f"complete runtime={final['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
