"""Optimise the DSPy planner's few-shot demos against execution accuracy on the TRAINING set.

    python -m analytics_agent.evals.optimize                         # local Ollama qwen2.5:7b, logs to MLflow
    python -m analytics_agent.evals.optimize --teacher cohere/command-a-03-2025   # bigger model writes the demos

Data hygiene: the optimiser sees only `train.yaml` (split into train/val here). `golden.yaml`, the reported number,
is never loaded by this script, so evaluating the saved program on golden is a held-out score.

Metric: the same check as the eval harness. A predicted plan is validated and executed by the agent's own
`validate`/`execute` nodes (`AnalyticsAgent.run_plan`) and its rows are compared with the hand-written reference SQL;
refuse cases pass only on refusal. No partial credit, no LLM judge.

Optimiser: BootstrapFewShotWithRandomSearch. It runs the program (or a teacher LM) over the train split, keeps the
traces that pass the metric as demos, builds several candidate demo sets and picks the best on the val split.
It does not rewrite the instructions, so the catalog and the security rules stay as written.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import dspy
import yaml

from ..agent import AnalyticsAgent, build_agent
from ..config import ROOT, Settings
from ..dspy_planner import DSPyLLM, make_lm, prediction_to_plan
from ..llm import HeuristicLLM, PlanError
from .run import check

TRAIN = Path(__file__).with_name("train.yaml")


def split(cases: list[dict], val_every: int = 2) -> tuple[list[dict], list[dict]]:
    """Deterministic stratified split: within each category (sorted by id), every `val_every`-th case goes to val."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for c in sorted(cases, key=lambda c: c["id"]):
        by_cat[c["category"]].append(c)
    train, val = [], []
    for group in by_cat.values():
        for i, c in enumerate(group):
            (val if i % val_every == 1 else train).append(c)
    return train, val


def to_examples(agent: AnalyticsAgent, cases: list[dict]) -> list[dspy.Example]:
    return [dspy.Example(context="\n".join(agent.index.context_for(c["question"])), question=c["question"],
                         feedback="none", case=c).with_inputs("context", "question", "feedback") for c in cases]


def make_metric(agent: AnalyticsAgent):
    def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
        try:
            plan = prediction_to_plan(pred)
        except PlanError:
            return False
        state = agent.run_plan(plan)
        rows = state["result"].rows if state.get("result") is not None else []
        passed, _ = check(agent, example.case, state["status"], rows)
        return passed
    return metric


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lm", default=None, help="student LM (LiteLLM id); default ANALYTICS_DSPY_LM / qwen2.5:7b on Ollama")
    p.add_argument("--teacher", default=None, help="optional teacher LM that generates the demos")
    p.add_argument("--train", type=Path, default=TRAIN)
    p.add_argument("--out", type=Path, default=None, help="where to save the program (default programs/planner_<tag>.json)")
    p.add_argument("--demos", type=int, default=4, help="max bootstrapped demos per candidate")
    p.add_argument("--candidates", type=int, default=6, help="random-search candidate programs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args(argv)

    settings = Settings(llm="heuristic")  # planner unused here; we only need the warehouse, layer and index
    agent = build_agent(settings)
    student_lm = make_lm(args.lm or settings.dspy_lm)
    from .. import agent as agent_mod
    planner = DSPyLLM(agent.layer, agent_mod._schema_text(agent.warehouse, agent.layer), lm=student_lm)
    assert isinstance(agent.llm, HeuristicLLM)

    cases = yaml.safe_load(args.train.read_text(encoding="utf-8"))
    train_cases, val_cases = split(cases)
    trainset, valset = to_examples(agent, train_cases), to_examples(agent, val_cases)
    metric = make_metric(agent)
    tag = (args.lm or settings.dspy_lm).split("/")[-1].replace(":", "-").replace(".", "")
    tag += f"_teacher-{args.teacher.split('/')[-1]}" if args.teacher else ""
    out = args.out or ROOT / "programs" / f"planner_{tag}.json"

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        import mlflow
        from .tracking import file_sha, git_state, setup
        setup()
        mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=False, silent=True)
        run = mlflow.start_run(run_name=f"optimize_{tag}")
        mlflow.set_tags({"kind": "optimize"})
        mlflow.log_params({"student_lm": student_lm.model, "teacher_lm": args.teacher or "none",
                           "optimizer": "BootstrapFewShotWithRandomSearch", "max_bootstrapped_demos": args.demos,
                           "num_candidate_programs": args.candidates, "seed": args.seed,
                           "train_file": args.train.name, "train_sha": file_sha(args.train),
                           "n_train": len(trainset), "n_val": len(valset), **git_state()})

    evaluate = dspy.Evaluate(devset=valset, metric=metric, num_threads=1, display_progress=True)
    t0 = time.time()
    with dspy.context(lm=student_lm):
        base = evaluate(planner.program)
        print(f"zero-shot val score: {base.score:.1f}%  ({time.time() - t0:.0f}s)")
        teacher_settings = {"lm": make_lm(args.teacher)} if args.teacher else None
        opt = dspy.BootstrapFewShotWithRandomSearch(
            metric=metric, max_bootstrapped_demos=args.demos, max_labeled_demos=0,
            num_candidate_programs=args.candidates, num_threads=1, teacher_settings=teacher_settings)
        compiled = opt.compile(planner.program, trainset=trainset, valset=valset)
        tuned = evaluate(compiled)
    elapsed = time.time() - t0
    print(f"optimised val score: {tuned.score:.1f}%  (zero-shot {base.score:.1f}%, {elapsed / 60:.1f} min total)")

    out.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(out))
    demos = [{"question": d.get("question"), "kind": d.get("kind")} for pr in compiled.predictors() for d in pr.demos]
    meta = {"student_lm": student_lm.model, "teacher_lm": args.teacher, "val_score_zero_shot": base.score,
            "val_score_optimised": tuned.score, "n_train": len(trainset), "n_val": len(valset), "demos": demos,
            "minutes": round(elapsed / 60, 1)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

    if use_mlflow:
        mlflow.log_metrics({"val_score_zero_shot": base.score, "val_score_optimised": tuned.score,
                            "n_demos": len(demos), "optimize_minutes": elapsed / 60})
        mlflow.log_artifact(str(out), artifact_path="program")
        mlflow.log_artifact(str(out.with_suffix(".meta.json")), artifact_path="program")
        mlflow.end_run()
        print(f"MLflow run: {run.info.run_id}")
    agent.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
