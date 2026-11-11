"""Generic agent runner with concurrency, a cost budget -- and a live LLM path.

The live path uses the owner's Z.ai Coding Plan as THE documented endpoint
for all live runs of this library:

    base URL: https://api.z.ai/api/coding/paas/v4   (OpenAI-compatible)
    model:    glm-5.3-flash  (pinned; no other model is offered)
    key:      RELIAGATE_ZAI_KEY environment variable

The live path is strictly opt-in behind the env var: nothing in reliagate
calls a network by default, and all core statistics + tests are offline
and key-free.  On any billing/endpoint error the runner STOPS and raises
-- it never falls back to another provider or to pay-per-call endpoints.
"""

from __future__ import annotations

import concurrent.futures
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ..model import EvalRun, Run

ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
ZAI_MODEL = "glm-5.3-flash"  # pinned by owner; do not add alternatives
ZAI_KEY_ENV = "RELIAGATE_ZAI_KEY"


# ---------------------------------------------------------------------------
# generic runner


AgentFn = Callable[[str], Optional[bool]]
"""fn(task_id or task payload) -> True/False, or None for abstain/infra error.
May instead return (success, cost) where cost is a float in budget units."""


@dataclass
class RunnerReport:
    run: EvalRun
    cost_spent: float
    attempts_run: int
    attempts_skipped_budget: int


def run_agent(
    fn: AgentFn,
    tasks: Sequence[str],
    k: int = 1,
    max_workers: int = 4,
    cost_budget: Optional[float] = None,
    estimated_cost_per_call: Optional[float] = None,
    version: Optional[str] = None,
    ordered: bool = True,
) -> RunnerReport:
    """Run ``fn`` on every task k times, with a thread pool and cost budget.

    ``cost_budget`` plus ``estimated_cost_per_call``: an attempt is only
    launched when ``cost_spent + estimated_cost_per_call <= budget``.
    Without an estimate, the budget is checked before each batch and a
    batch may overshoot by up to (batch size - 1) observed costs.

    Attempts not launched because the budget ran out are recorded as
    ``success=None`` with metadata ``{"reason": "budget_exhausted"}`` --
    abstained, never counted as failure.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    def over_budget(spent: float) -> bool:
        if cost_budget is None:
            return False
        if estimated_cost_per_call is not None:
            return spent + estimated_cost_per_call > cost_budget
        return spent >= cost_budget

    jobs = [(task, attempt) for task in tasks for attempt in range(k)]
    cost_spent = 0.0
    skipped = 0
    results: list[Run] = []

    def execute(job):
        task, attempt = job
        out = fn(task)
        if isinstance(out, tuple) and len(out) == 2:
            success, cost = out
        else:
            success, cost = out, 0.0
        return task, attempt, success, float(cost)

    # sequential honouring of the budget; small batches to keep order stable
    pending = list(jobs)
    if not ordered:
        pending.reverse()  # executor order is unpredictable anyway

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        batch: list = []
        while pending:
            task, attempt = pending.pop(0)
            committed = cost_spent + len(batch) * (
                estimated_cost_per_call or 0.0
            )
            if over_budget(committed):
                skipped += 1 + len(pending)
                results.extend(
                    Run(
                        task_id=t,
                        attempt=a,
                        success=None,
                        version=version,
                        metadata={"reason": "budget_exhausted"},
                    )
                    for t, a in [ (task, attempt) ] + pending
                )
                break
            batch.append((task, attempt))
            if len(batch) >= max_workers or not pending:
                for future in concurrent.futures.as_completed(
                    [pool.submit(execute, j) for j in batch]
                ):
                    t, a, success, cost = future.result()
                    cost_spent += cost
                    results.append(
                        Run(
                            task_id=t,
                            attempt=a,
                            success=success,
                            version=version,
                            metadata={"cost": cost},
                        )
                    )
                batch = []

    results.sort(key=lambda r: (r.task_id, r.attempt))
    return RunnerReport(
        run=EvalRun(name=version or "run", runs=results),
        cost_spent=cost_spent,
        attempts_run=len(jobs) - skipped,
        attempts_skipped_budget=skipped,
    )


# ---------------------------------------------------------------------------
# live Z.ai endpoint (opt-in behind RELIAGATE_ZAI_KEY)


class ZaiBillingError(RuntimeError):
    """Billing/endpoint failure of the Z.ai Coding Plan endpoint.

    Raised instead of falling back: the owner's rule is STOP on billing or
    endpoint errors -- never switch providers, never go pay-per-call.
    """


def make_zai_agent_fn(
    prompt_builder: Callable[[str], str],
    *,
    model: str = ZAI_MODEL,
    base_url: str = ZAI_BASE_URL,
    api_key_env: str = ZAI_KEY_ENV,
    timeout_s: float = 120.0,
    cost_per_call: float = 0.0,
    judge: Callable[[str], bool] | None = None,
) -> AgentFn:
    """Build an AgentFn that calls the Z.ai Coding Plan endpoint.

    Only stdlib (urllib) is used.  The call happens ONLY if the key env
    var is set, and this factory is only called from live scripts -- the
    library core never touches it.

    ``judge`` maps the model's completion text to a boolean verdict
    (task-specific); omit it to make the agent abstain-friendly by
    returning None on unparseable output.
    """
    import json as _json
    import urllib.error
    import urllib.request

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"live runs are opt-in: set {api_key_env} to use the Z.ai Coding "
            f"Plan endpoint ({base_url}, model {model})"
        )
    if model != ZAI_MODEL:
        # owner rule: glm-5.3-flash is the only supported model
        raise ValueError(
            f"only the pinned model {ZAI_MODEL!r} may be used (got {model!r})"
        )

    def call(task: str) -> Optional[bool]:
        url = base_url.rstrip("/") + "/chat/completions"
        payload = _json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt_builder(task)}],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = _json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # 401/402/429 = auth/billing/rate-limit -> STOP, no fallback
            raise ZaiBillingError(
                f"Z.ai endpoint error (HTTP {exc.code}) after "
                f"{time.monotonic() - started:.1f}s: {exc.reason}; "
                "stopping -- no provider fallback per owner policy"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ZaiBillingError(
                f"Z.ai endpoint unreachable: {exc!r}; "
                "stopping -- no provider fallback per owner policy"
            ) from exc
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        if judge is None:
            return None
        try:
            return bool(judge(text))
        except Exception:
            return None

    return call
