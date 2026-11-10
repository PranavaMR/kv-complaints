"""
Every context x every method, Q1 and Q2, one JSONL line per run. Resumable.

    python run.py --calibrate   # Full cache, 12 planted contexts of restaurants 0-3, Q1 only
    python run.py               # 240 runs -> data/runs_raw.jsonl; then `python redact.py`
"""
import argparse
import datetime as dt
import json
import subprocess
import time
from importlib.metadata import version

import kvpress  # noqa: F401  (registers the "kv-press-text-generation" pipeline)
import torch
from kvpress import KnormPress, RandomPress, SnapKVPress, StreamingLLMPress
from transformers import DynamicCache, pipeline

import config
from audit import RetentionRecorder, retention_by_review
from data import load_manifest, render
from detect import citation_mention, keyword_mention
from lexicon import Q2_PHRASES


def q1_prompt() -> str:
    return ("\n\nYou are advising the owner of this restaurant. Based only on the reviews above, "
            f"list the {config.N_RECOMMENDATIONS} most important, actionable recommendations, in priority order. "
            "For each, cite the supporting review numbers in the form [Review k].")


def q2_prompt(aspect: str) -> str:
    return (f"\n\nDid any customer mention {Q2_PHRASES[aspect]}? "
            "Answer 'yes' or 'no' first, then quote the relevant sentence if yes.")


def make_press(method: str, ratio: float, seed: int):
    """The kvpress press for a method, or None for Full."""
    presses = {
        "full": lambda: None,
        # RandomPress reseeds on every call, so Q1 and Q2 keep the same tokens.
        "random": lambda: RandomPress(compression_ratio=ratio, seed=seed),
        "streaming_llm": lambda: StreamingLLMPress(compression_ratio=ratio),
        "snapkv": lambda: SnapKVPress(compression_ratio=ratio),
        "knorm": lambda: KnormPress(compression_ratio=ratio),
    }
    return presses[method]()


class PrefillTimer:
    """The pipeline's first model.model call is the prefill; its end splits prefill from decode."""

    def __init__(self, base_model):
        base_model.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.prefill_end is None:
            torch.cuda.synchronize()
            self.prefill_end = time.perf_counter()

    def start(self):
        torch.cuda.synchronize()
        self.prefill_end, self.t0 = None, time.perf_counter()

    def stop(self) -> tuple[float, float]:
        torch.cuda.synchronize()
        return self.prefill_end - self.t0, time.perf_counter() - self.prefill_end


def ask(pipe, timer, rendered, question, max_new_tokens, recorder):
    """One pipeline call: answer, prefill and decode seconds, kept indices (None for Full)."""
    if recorder is not None:
        recorder.reset()
    timer.start()
    out = pipe(rendered.context_text, question=question, press=recorder,
               max_new_tokens=max_new_tokens, cache=DynamicCache())
    prefill_s, decode_s = timer.stop()
    return out["answer"], prefill_s, decode_s, (dict(recorder.kept) if recorder is not None else None)


def q2_aspect_for(ctx) -> str:
    """Planted: its aspect. Control: the aspect its restaurant gets at slot 15 (a yes is a false positive)."""
    return ctx.plant_aspect if ctx.condition == "plant" else config.ASPECTS[(int(ctx.context_id[1:3]) + 1) % 3]


def load_pipe():
    return pipeline("kv-press-text-generation", model=config.MODEL_NAME, device=config.DEVICE,
                    dtype=getattr(torch, config.DTYPE), model_kwargs={"attn_implementation": "sdpa"})


def run(contexts, methods, out_path, ask_q2=True):
    pipe = load_pipe()
    timer = PrefillTimer(pipe.model.model)
    model_revision = getattr(pipe.model.config, "_commit_hash", None)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT, text=True).strip()
    done = {json.loads(l)["run_id"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ratio = config.COMPRESSION_RATIO

    for c_index, ctx in enumerate(contexts):
        rendered = render(ctx, pipe.tokenizer)
        expected = pipe.preprocess(rendered.context_text, questions=[""], answer_prefix="",
                                   max_context_length=10**9)["context_ids"]
        assert torch.equal(expected, rendered.context_ids), "render() disagrees with kvpress preprocessing"

        seed = config.SEED + c_index
        for method in methods:
            run_id = f"{ctx.context_id}__{method}__{ratio if method != 'full' else 0.0}"
            if run_id in done:
                continue
            press = make_press(method, ratio, seed)
            recorder = RetentionRecorder(press) if press is not None else None

            q1, prefill_s, decode_s, kept = ask(pipe, timer, rendered, q1_prompt(), config.Q1_MAX_NEW_TOKENS, recorder)
            q2, q2_aspect = None, q2_aspect_for(ctx)
            if ask_q2:
                q2, _, _, kept_q2 = ask(pipe, timer, rendered, q2_prompt(q2_aspect), config.Q2_MAX_NEW_TOKENS, recorder)
                if kept is not None:
                    assert all(torch.equal(kept[l], kept_q2[l]) for l in kept), f"{run_id}: Q1 and Q2 kept different tokens"

            by_review = ([1.0] * len(rendered.spans) if kept is None
                         else retention_by_review(kept, rendered.spans).round(4).tolist())
            row = {
                "run_id": run_id, "model": config.MODEL_NAME, "model_revision": model_revision,
                "kvpress_version": version("kvpress"), "press": method,
                "compression_ratio": ratio if method != "full" else 0.0,
                "context_id": ctx.context_id, "business_id": ctx.business_id, "condition": ctx.condition,
                "plant_aspect": ctx.plant_aspect, "plant_position": ctx.plant_position,
                "template_id": ctx.template_id, "n_context_tokens": rendered.context_ids.shape[1],
                "q1_answer": q1, "q2_answer": q2, "q2_aspect": q2_aspect,
                "retention_planted": by_review[ctx.plant_position - 1] if ctx.condition == "plant" else None,
                "retention_by_review": by_review,
                "prefill_s": round(prefill_s, 3), "decode_s": round(decode_s, 3), "seed": seed,
                "git_sha": sha, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{c_index + 1}/{len(contexts)}] {run_id}  prefill {prefill_s:.1f}s  decode {decode_s:.1f}s")


def calibrate():
    contexts = [c for c in load_manifest() if c.condition == "plant" and int(c.context_id[1:3]) < 4]
    config.CALIBRATION_FILE.unlink(missing_ok=True)
    run(contexts, ["full"], config.CALIBRATION_FILE, ask_q2=False)
    rows = [json.loads(l) for l in config.CALIBRATION_FILE.read_text().splitlines()]
    kw = sum(keyword_mention(r["q1_answer"], r["plant_aspect"]) for r in rows)
    cite = sum(citation_mention(r["q1_answer"], r["plant_position"]) for r in rows)
    print(f"\nN_RECOMMENDATIONS={config.N_RECOMMENDATIONS}: keyword {kw}/{len(rows)}, citation {cite}/{len(rows)}. "
          "Rule: below 30% or above 95% -> change it once (to 8 or 3), log it, then launch.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    if parser.parse_args().calibrate:
        calibrate()
    else:
        contexts = [c for c in load_manifest() if int(c.context_id[1:3]) < config.N_RESTAURANTS]
        run(contexts, list(config.METHODS), config.RAW_RUNS_FILE)
