"""Tests for hook 9-word limit, deceased-person respectful tone, 2nd-pass verification, and 3-failure flagging."""

from __future__ import annotations

from typing import Any, List

from shortsforge.config import AppConfig
from shortsforge.hooks import (
    count_hook_words,
    deterministic_hook_checks,
    generate_verified_hooks,
)
from shortsforge.llm import LLMClient
from shortsforge.models import ClipCandidate


class _ScriptedLLM(LLMClient):
    """Test double returning scripted responses in sequence to test verification & retries."""

    def __init__(self, responses: List[Any], config: AppConfig) -> None:
        super().__init__(config=config, workspace=None)
        self._responses = list(responses)
        self.calls: List[str] = []

    def complete_json(self, namespace: str, system_prompt: str, user_prompt: str, **kwargs: Any) -> Any:
        self.calls.append(namespace)
        if not self._responses:
            raise AssertionError(f"Unexpected extra LLM call: {namespace}")
        return self._responses.pop(0)


def _sample_candidate() -> ClipCandidate:
    return ClipCandidate(
        id="clip_01",
        start=0.0,
        end=28.0,
        score=88,
        reason="Strong emotional turning point.",
        key_quote="I almost quit until my mentor called me back.",
        transcript_text=(
            "I was ready to walk away from the whole project. "
            "I almost quit until my mentor called me back and reminded me why we started."
        ),
        peak_start=10.0,
        peak_end=12.5,
    )


def test_word_limit_enforced_strictly() -> None:
    assert count_hook_words("He almost quit before everything changed.") == 6
    ok_short, viol_short = deterministic_hook_checks("He almost quit before everything changed.", max_words=9)
    assert ok_short is True
    assert viol_short == []

    too_long = "This hook deliberately contains ten separate words which violates limits."
    assert count_hook_words(too_long) == 10
    ok_long, viol_long = deterministic_hook_checks(too_long, max_words=9)
    assert ok_long is False
    assert any("exceeding max limit" in v for v in viol_long)


def test_deceased_person_disrespectful_framing_blocked() -> None:
    ok, violations = deterministic_hook_checks(
        "His shocking death changed Hollywood forever",
        max_words=9,
        deceased_subject=True,
    )
    assert ok is False
    assert any("Disrespectful" in v for v in violations)

    ok_dignified, _ = deterministic_hook_checks(
        "His honest advice on never giving up",
        max_words=9,
        deceased_subject=True,
    )
    assert ok_dignified is True


def test_failed_hooks_regenerate_and_pass_on_second_attempt() -> None:
    cfg = AppConfig()
    cfg.llm.provider = "mock"
    cfg.hook.max_verification_attempts = 3
    cfg.hook.variants_count = 3

    scripted = [
        # Attempt 1: generator invents a scandal -> verifier fails all 3
        {
            "variants": [
                {"rank": 1, "text": "He got fired after a secret scandal.", "predicted_retention": 95},
                {"rank": 2, "text": "Why the network banned him live.", "predicted_retention": 90},
                {"rank": 3, "text": "The secret million dollar lawsuit exposed.", "predicted_retention": 85},
            ]
        },
        {"passed": False, "reason": "No scandal in transcript.", "violations": ["Invented scandal"]},
        {"passed": False, "reason": "No network ban in transcript.", "violations": ["Invented ban"]},
        {"passed": False, "reason": "No lawsuit in transcript.", "violations": ["Invented lawsuit"]},
        # Attempt 2: generator produces truthful hooks -> verifier passes all 3
        {
            "variants": [
                {"rank": 1, "text": "He almost quit until one phone call.", "predicted_retention": 93},
                {"rank": 2, "text": "The call that stopped him from quitting.", "predicted_retention": 89},
                {"rank": 3, "text": "Why he chose not to walk away.", "predicted_retention": 84},
            ]
        },
        {"passed": True, "reason": "Faithful to transcript.", "violations": []},
        {"passed": True, "reason": "Faithful to transcript.", "violations": []},
        {"passed": True, "reason": "Faithful to transcript.", "violations": []},
    ]

    llm = _ScriptedLLM(scripted, cfg)
    hook_set = generate_verified_hooks(_sample_candidate(), "Jane Doe", llm, cfg)

    assert hook_set.attempts == 2
    assert hook_set.flagged_for_review is False
    assert hook_set.active_hook_text() == "He almost quit until one phone call."
    assert len(hook_set.variants) == 3
    assert all(v.verification.passed for v in hook_set.variants)


def test_hooks_flagged_after_three_failed_verification_attempts() -> None:
    cfg = AppConfig()
    cfg.llm.provider = "mock"
    cfg.hook.max_verification_attempts = 3
    cfg.hook.variants_count = 2

    bad_gen = {
        "variants": [
            {"rank": 1, "text": "He revealed a shocking secret lawsuit.", "predicted_retention": 90},
            {"rank": 2, "text": "Why he walked out mid interview.", "predicted_retention": 85},
        ]
    }
    bad_ver = {"passed": False, "reason": "Claim not in transcript.", "violations": ["Hallucination"]}

    scripted = [bad_gen, bad_ver, bad_ver] * 3
    llm = _ScriptedLLM(scripted, cfg)
    hook_set = generate_verified_hooks(_sample_candidate(), "Jane Doe", llm, cfg)

    assert hook_set.attempts == 3
    assert hook_set.flagged_for_review is True
    assert "failed after 3 attempts" in hook_set.flag_reason.lower()
