"""Stage 4: 3-second hook generation (5 ranked variants <= 9 words) & 2nd-pass transcript truth verification."""

from __future__ import annotations

import re
from typing import List, Tuple

from shortsforge.config import AppConfig
from shortsforge.llm import LLMClient
from shortsforge.models import (
    ClipCandidate,
    HookSet,
    HookVariant,
    HookVerification,
)


HOOK_GENERATOR_SYSTEM = """You write punchy, truthful on-screen hooks for vertical short-form clips.

HARD TRUTH RULES (NON-NEGOTIABLE):
1. Every hook may ONLY describe what actually happens or is said in the provided clip transcript.
2. NEVER invent events, quotes, dates, scandals, diagnoses, conflicts, or misleading implications.
3. Build curiosity purely through sharp framing (e.g., "He was asked about quitting. His answer stunned everyone.").
4. Maximum 9 words per hook (STRICT LIMIT - count carefully).
5. If the subject is deceased, maintain a respectful, dignified tone: zero mockery, sensationalism, or exploitation of their death.
6. Return exactly 5 distinct hook variants ranked 1 (highest predicted retention) to 5.

Return strict JSON:
{
  "variants": [
    {"rank": 1, "text": "Hook text <= 9 words", "predicted_retention": 93},
    {"rank": 2, "text": "...", "predicted_retention": 89},
    {"rank": 3, "text": "...", "predicted_retention": 85},
    {"rank": 4, "text": "...", "predicted_retention": 81},
    {"rank": 5, "text": "...", "predicted_retention": 77}
  ]
}"""


HOOK_VERIFIER_SYSTEM = """You are a strict fact-checking editor verifying whether a short video hook is 100% faithful to a transcript.

Evaluate the hook against the transcript and return FAIL if ANY of the following occur:
- Mentions an event, quote, number, date, person, or claim NOT supported by the transcript
- Creates a clickbait implication contradicted by or absent from the transcript
- Exceeds 9 words
- If subject_is_deceased=true: uses disrespectful, mocking, or exploitative language about their death

Return strict JSON:
{
  "passed": true,
  "reason": "One-sentence explanation.",
  "violations": []
}"""


_DISRESPECTFUL_DECEASED_PATTERNS = [
    r"\bshocking\s+death\b",
    r"\blast\s+words\s+before\s+dying\b",
    r"\bhaunting\s+final\b",
    r"\bpredicted\s+(?:his|her|their)\s+own\s+death\b",
    r"\btragic\s+end\b",
    r"\bdied\s+shortly\s+after\b",
]


def count_hook_words(text: str) -> int:
    """Count words in a hook consistently."""
    return len([tok for tok in re.split(r"\s+", text.strip()) if tok])


def deterministic_hook_checks(
    hook_text: str,
    max_words: int = 9,
    deceased_subject: bool = False,
) -> Tuple[bool, List[str]]:
    """Run deterministic guardrails (word limit, non-empty, deceased respectful-tone check)."""
    violations: List[str] = []
    cleaned = hook_text.strip()
    if not cleaned:
        violations.append("Hook text is empty.")
    wc = count_hook_words(cleaned)
    if wc > max_words:
        violations.append(f"Hook has {wc} words, exceeding max limit of {max_words} words.")
    if deceased_subject:
        for pat in _DISRESPECTFUL_DECEASED_PATTERNS:
            if re.search(pat, cleaned, re.IGNORECASE):
                violations.append("Disrespectful or exploitative framing regarding a deceased person.")
                break
    return len(violations) == 0, violations


def verify_hook_against_transcript(
    hook_text: str,
    transcript_text: str,
    llm: LLMClient,
    config: AppConfig,
    deceased_subject: bool = False,
    cache_namespace: str = "hook_verify",
) -> HookVerification:
    """Run deterministic checks plus an independent second-pass LLM verification against the transcript."""
    det_ok, det_violations = deterministic_hook_checks(
        hook_text,
        max_words=config.hook.max_words,
        deceased_subject=deceased_subject,
    )
    if not det_ok:
        return HookVerification(
            passed=False,
            reason="; ".join(det_violations),
            violations=det_violations,
        )

    user_prompt = (
        f"subject_is_deceased: {str(deceased_subject).lower()}\n"
        f"max_words: {config.hook.max_words}\n"
        f"CANDIDATE HOOK:\n{hook_text.strip()}\n\n"
        f"CLIP TRANSCRIPT:\n{transcript_text.strip()}\n"
    )
    resp = llm.complete_json(
        namespace=cache_namespace,
        system_prompt=HOOK_VERIFIER_SYSTEM,
        user_prompt=user_prompt,
    )
    passed = bool(resp.get("passed", False)) if isinstance(resp, dict) else False
    reason = str(resp.get("reason", "")).strip() if isinstance(resp, dict) else "Invalid verifier response"
    violations = list(resp.get("violations", [])) if isinstance(resp, dict) else ["Malformed JSON"]
    if not passed and not reason:
        reason = "; ".join(violations) or "Failed transcript truth verification."
    return HookVerification(passed=passed, reason=reason, violations=violations)


def generate_verified_hooks(
    candidate: ClipCandidate,
    person: str,
    llm: LLMClient,
    config: AppConfig,
    deceased_subject: bool = False,
) -> HookSet:
    """Generate 5 ranked hook variants and verify via a 2nd LLM pass; regenerate up to 3 attempts, then flag."""
    max_attempts = max(1, config.hook.max_verification_attempts)
    feedback = ""
    last_variants: List[HookVariant] = []
    last_failure_reason = ""

    for attempt in range(1, max_attempts + 1):
        deceased_note = (
            "IMPORTANT: The subject is deceased. Keep tone dignified and respectful; never exploit their death.\n"
            if deceased_subject
            else ""
        )
        retry_note = (
            f"\nPREVIOUS ATTEMPT FAILED VERIFICATION - FIX THIS:\n{feedback}\n"
            if feedback
            else ""
        )
        user_prompt = (
            f"Subject: {person}\n"
            f"{deceased_note}"
            f"Key quote: {candidate.key_quote}\n"
            f"Clip transcript:\n{candidate.transcript_text}\n"
            f"{retry_note}\n"
            f"Generate {config.hook.variants_count} truthful hook variants (each <= {config.hook.max_words} words), "
            "ranked #1 to #5 by predicted retention."
        )
        resp = llm.complete_json(
            namespace=f"hook_generate_{candidate.id}_a{attempt}",
            system_prompt=HOOK_GENERATOR_SYSTEM,
            user_prompt=user_prompt,
        )
        raw_variants = resp.get("variants", []) if isinstance(resp, dict) else []
        verified_variants: List[HookVariant] = []
        failed_notes: List[str] = []

        for idx, item in enumerate(raw_variants[: config.hook.variants_count], start=1):
            text = str(item.get("text") or "").strip().strip('"').strip("'")
            retention = int(max(0, min(100, item.get("predicted_retention", max(60, 95 - idx * 4)))))
            ver = verify_hook_against_transcript(
                hook_text=text,
                transcript_text=candidate.transcript_text,
                llm=llm,
                config=config,
                deceased_subject=deceased_subject,
                cache_namespace=f"hook_verify_{candidate.id}_a{attempt}_v{idx}",
            )
            verified_variants.append(
                HookVariant(
                    rank=idx,
                    text=text,
                    predicted_retention=retention,
                    verification=ver,
                )
            )
            if not ver.passed:
                failed_notes.append(f"Variant #{idx} ('{text}'): {ver.reason}")

        passing = [v for v in verified_variants if v.verification.passed]
        failing = [v for v in verified_variants if not v.verification.passed]
        ordered = sorted(passing, key=lambda v: v.predicted_retention, reverse=True) + sorted(
            failing, key=lambda v: v.predicted_retention, reverse=True
        )
        for rank_i, v in enumerate(ordered, start=1):
            v.rank = rank_i
        last_variants = ordered

        if passing and len(passing) >= min(3, max(1, len(ordered))):
            return HookSet(
                selected_index=0,
                variants=ordered,
                attempts=attempt,
                flagged_for_review=False,
                flag_reason="",
            )

        last_failure_reason = "; ".join(failed_notes) or "Insufficient hooks passed transcript truth check."
        feedback = last_failure_reason

    return HookSet(
        selected_index=0,
        variants=last_variants,
        attempts=max_attempts,
        flagged_for_review=True,
        flag_reason=(
            f"Hook verification failed after {max_attempts} attempts: {last_failure_reason}"
        ),
    )
