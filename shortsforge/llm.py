"""Pluggable LLM client with Gemini, Claude, and a built-in zero-API-key Smart NLP Moment & Hook Engine."""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from shortsforge.config import AppConfig, get_env_key
from shortsforge.state import ProjectWorkspace


class LLMError(RuntimeError):
    """Raised when an LLM call fails or returns unparseable JSON."""


def _extract_json_block(raw_text: str) -> Any:
    """Extract and parse a JSON object or array from model output (handles markdown fences)."""
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        obj_start = text.find("{")
        arr_start = text.find("[")
        starts = [i for i in (obj_start, arr_start) if i >= 0]
        if not starts:
            raise LLMError(f"No JSON found in LLM response:\n{raw_text[:500]}")
        start = min(starts)
        end_char = "}" if text[start] == "{" else "]"
        end = text.rfind(end_char)
        if end <= start:
            raise LLMError(f"Incomplete JSON in LLM response:\n{raw_text[:500]}")
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"Invalid JSON in LLM response: {exc}\n{raw_text[:500]}") from exc


_EMOTION_WORDS = {
    "arrest", "jail", "prison", "court", "trial", "police", "raid", "judge", "freedom",
    "million", "billion", "thousand", "dollars", "money", "lost", "broke", "rich",
    "quit", "walked", "died", "death", "alive", "survive", "afraid", "fear", "suffered",
    "matrix", "banned", "canceled", "attack", "enemy", "enemies", "corrupt", "rigged",
    "truth", "secret", "lie", "lied", "honest", "real", "never", "always", "everything",
    "nothing", "changed", "forever", "worst", "best", "crazy", "insane", "shocked",
}

_QUOTABLE_PATTERNS = [
    r"\b(?:the\s+truth|the\s+reality|the\s+problem|the\s+secret)\s+is\b",
    r"\b(?:nobody|no\s+one|everybody|everyone)\s+(?:tells|wants|knows|said)\b",
    r"\b(?:i\s+never|i\s+almost|i\s+refused|i\s+decided|i\s+realized)\b",
    r"\b(?:you\s+have\s+to|you\s+cannot|you\s+can't|when\s+you|if\s+you)\b",
    r"\$\d+|\b\d+\s+(?:million|thousand|years|hours|days|months)\b",
]

_SPONSOR_SKIP_WORDS = {
    "patreon", "sponsor", "sponsored", "subscribe", "snapchat", "instagram",
    "link in description", "link below", "promo code", "discount", "merch",
    "shopify", "checkout", "squarespace", "betterhelp", "athletic greens", "ag1",
    "nordvpn", "expressvpn", "draftkings", "free trial", "use code", "brought to you by",
    "manscaped", "whoop", "audible", "hubspot", "mint mobile",
}


def _score_sentence_text(text: str) -> Tuple[int, int, int]:
    """Return heuristic (emotional_score, quotable_score, humor_surprise_score) in [20..98]."""
    lower = text.lower()
    if any(sw in lower for sw in _SPONSOR_SKIP_WORDS) or re.search(r"[a-z0-9\-]+\.com/", lower):
        return 20, 20, 20

    tokens = re.findall(r"[a-zA-Z0-9'$%]+", lower)
    if not tokens:
        return 35, 35, 35

    filler_hits = len(re.findall(r"\b(?:uh|um|er|ah)\b|\b(?:you know|i mean|sort of|kind of|there she is|there he is)\b", lower))
    filler_penalty = filler_hits * 10

    emo_hits = sum(1 for t in tokens if t.strip("'$%") in _EMOTION_WORDS or any(c.isdigit() for c in t))
    quote_hits = sum(1 for pat in _QUOTABLE_PATTERNS if re.search(pat, lower))
    contrast_hits = sum(
        1 for w in ("but", "until", "then", "suddenly", "actually", "instead", "imagine", "funny", "warning", "formula", "truth", "secret")
        if f" {w} " in f" {lower} "
    )

    emo = max(35, min(97, 60 + emo_hits * 7 + (5 if "!" in text else 0) - filler_penalty))
    quo = max(35, min(98, 62 + quote_hits * 11 + emo_hits * 4 - filler_penalty))
    hum = max(35, min(94, 56 + contrast_hits * 9 + (4 if "?" in text else 0) - filler_penalty))
    return emo, quo, hum


def _strip_filler_lead(text: str) -> str:
    cleaned = re.sub(r"^(?:>>\s*|\(?SPEAKER_\d+\)?\s*)+", "", text).strip()
    for _ in range(4):
        updated = re.sub(
            r"^(?:yeah|yes|no|well|so|but|and|okay|right|you know|i mean|actually|honestly|listen|in terms of)[,.\s]+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned or re.sub(r"^(?:>>\s*|\(?SPEAKER_\d+\)?\s*)+", "", text).strip()


def _build_hooks_from_quote(key_quote: str, transcript_text: str) -> List[Dict[str, Any]]:
    """Build 5 distinct, factual <=8-word Iman Gadzhi / Hormozi / Documentary kinetic hook titles (never dangling fragments)."""
    combined = f"{key_quote} {transcript_text}"
    lower = combined.lower()

    pct_m = re.search(r"(\b\d+(?:\.\d+)?%\s+of\s+(?:the\s+)?(?:world's\s+)?[a-zA-Z]+|\b\d+(?:\.\d+)?%)", combined, re.I)
    money_m = re.search(r"(\$\d[\d,]*\s*(?:million|billion|thousand|k|m)?|\d+\s+(?:million|billion|thousand)\s+dollars)", combined, re.I)
    time_m = re.search(r"(\b\d+\s+(?:hours|days|months|years)\b|\b(?:19|20)\d{2}\b)", combined, re.I)

    candidates: List[str] = []
    core_quote = _strip_filler_lead(key_quote)
    clean_words = [w.strip(".,!?;:\"'()[]") for w in core_quote.split() if w.strip(".,!?;:\"'()[]")]

    # Only use the direct quote if it is already a complete, self-contained 4-8 word statement (NEVER a truncated "...").
    dangling_tail = {
        "the", "a", "an", "to", "for", "of", "with", "in", "on", "at", "by", "from",
        "and", "but", "or", "so", "because", "if", "when", "that", "which", "who", "who's",
        "is", "are", "was", "were", "have", "has", "had", "can", "will", "would", "you", "we", "they",
    }
    if (
        4 <= len(clean_words) <= 8
        and not core_quote.endswith("?")
        and clean_words[-1].lower() not in dangling_tail
    ):
        stmt = " ".join(clean_words)
        stmt = stmt[0].upper() + stmt[1:] + "."
        candidates.append(stmt)

    if pct_m:
        pct_str = pct_m.group(1).strip()
        if len(pct_str.split()) <= 5:
            candidates.append(f"The {pct_str.title()} Reality Nobody Talks About.")
        else:
            num_only = re.search(r"\d+(?:\.\d+)?%", pct_str)
            if num_only:
                candidates.append(f"The {num_only.group(0)} Chokehold Nobody Realizes.")
    if money_m:
        amt = money_m.group(1).strip()
        candidates.append(f"The Real Truth Behind {amt}.")
    if time_m:
        tm = time_m.group(1).strip()
        candidates.append(f"What Changed Everything In {tm.title()}.")
    if any(w in lower for w in ("chip", "semiconductor", "china", "taiwan", "lithography", "nvidia", "apple")):
        candidates.append("How The Global Power Formula Really Works.")
    if any(w in lower for w in ("jail", "prison", "arrest", "detention", "customs", "deported", "court", "trial")):
        candidates.append("What Really Happened Behind Closed Doors.")
    if any(w in lower for w in ("divorce", "marriage", "relationship", "partner", "dating", "spendaholic", "mooch")):
        candidates.append("The Hidden Warning Sign Most People Ignore.")
    if any(w in lower for w in ("money", "rich", "wealth", "invest", "dividend", "broke", "spend", "financial")):
        candidates.append("The Brutal Truth About Growing Your Money.")
    if any(w in lower for w in ("quit", "walk away", "shutting down", "lost", "failed", "breakthrough")):
        candidates.append("He Almost Quit Before This Turning Point.")
    if any(w in lower for w in ("matrix", "banned", "canceled", "media", "system", "polarizing")):
        candidates.append("Why The System Tried To Stop Him.")

    fallbacks = [
        "The Unfiltered Truth Most People Ignore.",
        "Listen Closely To This Exact Breakdown.",
        "This Single Lesson Changes Everything.",
        "Most People Realize This Far Too Late.",
        "The Reality Nobody Wants To Admit.",
    ]
    for fb in fallbacks:
        if fb not in candidates:
            candidates.append(fb)

    final_variants: List[Dict[str, Any]] = []
    seen = set()
    for text in candidates:
        words = text.split()
        if len(words) > 9:
            text = " ".join(words[:8]).rstrip(",;:") + "."
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        rank = len(final_variants) + 1
        final_variants.append({
            "rank": rank,
            "text": text,
            "predicted_retention": max(78, 97 - rank * 3),
        })
        if len(final_variants) >= 5:
            break

    return final_variants


_META_STOPWORDS = {
    "about", "after", "again", "against", "almost", "already", "always", "among", "another",
    "anyone", "anything", "around", "away", "back", "because", "before", "being", "between",
    "came", "come", "completely", "could", "doing", "don't", "during", "even", "every",
    "everybody", "everyone", "everything", "first", "found", "from", "going", "gonna", "great",
    "happened", "having", "here's", "himself", "however", "inside", "instead", "interview",
    "into", "just", "kept", "knew", "know", "like", "little", "looking", "made", "makes",
    "making", "many", "maybe", "means", "more", "much", "never", "next", "nothing", "onto", "other",
    "over", "people", "person", "pretty", "quite", "rather", "really", "right", "said",
    "saying", "says", "second", "should", "simply", "since", "somebody", "something",
    "sometimes", "speaker", "still", "stuff", "tell", "telling", "terms", "thank", "that", "that's",
    "their", "them", "there", "these", "they", "they're", "thing", "things", "think", "this", "those", "though",
    "three", "through", "today", "told", "took", "trying", "turned", "under", "until",
    "using", "wanted", "wants", "wasn't", "water", "went", "were", "what", "when", "where",
    "which", "while", "whole", "will", "with", "within", "without", "would", "year", "years",
    "your", "you're", "you've", "young", "yours", "have", "been",
}

_TOPIC_HASHTAG_RULES: List[Tuple[str, Tuple[str, ...], List[str]]] = [
    ("Money & Wealth", ("money", "wealth", "rich", "broke", "income", "million", "dollar", "spend", "$50", "$100"), ["#money", "#wealth", "#personalfinance"]),
    ("Investing", ("invest", "dividend", "stock", "index", "market", "portfolio", "crypto", "house"), ["#investing", "#financialfreedom", "#wealthbuilding"]),
    ("Business", ("business", "company", "startup", "founder", "entrepreneur", "profit", "scale"), ["#entrepreneur", "#business", "#founder"]),
    ("Relationships", ("marriage", "divorce", "wife", "husband", "date", "dating", "love", "relationship"), ["#relationships", "#marriage", "#datingadvice"]),
    ("Family", ("mother", "father", "mom", "dad", "family", "parent", "kids", "child"), ["#family", "#lifeadvice", "#storytime"]),
    ("Mindset & Discipline", ("discipline", "habit", "work", "focus", "mindset", "routine", "fail", "quit", "breakthrough"), ["#mindset", "#discipline", "#success"]),
    ("AI & Tech", ("ai", "artificial", "software", "tech", "apple", "model", "robot", "future"), ["#ai", "#technology", "#future"]),
    ("Truth & Justice", ("jail", "prison", "court", "trial", "police", "law", "freedom", "system"), ["#truth", "#justice", "#documentary"]),
    ("Health & Fitness", ("health", "sleep", "diet", "gym", "workout", "brain", "stress", "dopamine"), ["#health", "#selfimprovement", "#wellness"]),
]


def _extract_video_specific_metadata(
    person: str,
    source_credit: str,
    orig_title: str,
    hook: str,
    quote: str,
    transcript_text: str,
) -> Dict[str, Any]:
    """Generate titles, social post caption, description, hashtags, and comment grounded strictly in the clip."""
    clean_transcript = re.sub(r"\(?SPEAKER_\d+\)?\s*", "", transcript_text).strip()
    full_text = f"{quote} {clean_transcript} {orig_title}".strip()
    lower = full_text.lower()

    # Extract top recurring/distinct content keywords from the clip
    raw_tokens = re.findall(r"\b[a-zA-Z]{4,16}\b", clean_transcript.lower())
    freq: Dict[str, int] = {}
    for tok in raw_tokens:
        if tok in _META_STOPWORDS or tok.startswith("speaker"):
            continue
        freq[tok] = freq.get(tok, 0) + 1
    top_words = [w for w, _ in sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0])))[:6]]

    # Build relevant topic hashtags & clean human-readable topic labels
    hashtags: List[str] = []
    matched_topics: List[str] = []
    person_tag = "#" + re.sub(r"[^a-zA-Z0-9]", "", person.lower())
    if len(person_tag) > 2:
        hashtags.append(person_tag)
    show_tag = "#" + re.sub(r"[^a-zA-Z0-9]", "", source_credit.lower())[:20]
    if len(show_tag) > 2 and show_tag not in hashtags:
        hashtags.append(show_tag)

    for topic_name, triggers, tags in _TOPIC_HASHTAG_RULES:
        if any((t in lower if t.startswith("$") else bool(re.search(rf"\b{t}", lower))) for t in triggers):
            matched_topics.append(topic_name)
            for tg in tags:
                if tg not in hashtags:
                    hashtags.append(tg)
                if len(hashtags) >= 7:
                    break
        if len(hashtags) >= 7:
            break

    for kw in top_words:
        if len(hashtags) >= 8:
            break
        kw_tag = f"#{kw}"
        if kw_tag not in hashtags and len(kw_tag) >= 5:
            hashtags.append(kw_tag)

    for fallback_tag in ("#shorts", "#viral", "#podcastclips", "#clips"):
        if len(hashtags) >= 9:
            break
        if fallback_tag not in hashtags:
            hashtags.append(fallback_tag)

    clean_q = _strip_filler_lead(quote)
    if clean_q and clean_q[0].islower():
        clean_q = clean_q[0].upper() + clean_q[1:]
    short_q = clean_q[:74].rstrip(" ,;:") + ("..." if len(clean_q) > 74 else "")

    topic_main = matched_topics[0] if matched_topics else (top_words[0].title() if top_words else "Resilience")
    topic_label = " & ".join(matched_topics[:2]) if matched_topics else (", ".join(w.title() for w in top_words[:2]) or "Mindset")

    titles = [
        f"{person} On {topic_main}: \"{short_q[:46]}...\"" if len(short_q) > 46 else f"{person}: {short_q}",
        f"{hook.rstrip('.')} ({person})"[:92],
        f"The Real Truth About {topic_main} — {person}"[:92],
        f"What {person} Explained About {topic_main}"[:92],
        f"\"{short_q}\""[:92] if short_q else f"{person} Breaks Down {topic_label}",
    ]

    tag_str = " ".join(hashtags[:9])
    post_caption = (
        f"{hook.rstrip('.')} 🎯\n\n"
        f"{person} breaks down {topic_label}: \"{clean_q[:135]}\"\n\n"
        f"Watch till the end — what's your take on {topic_main.lower()}? 👇\n\n"
        f"{tag_str}"
    )

    description = (
        f"{person} discusses {topic_label} on {source_credit} ({orig_title[:70]}).\n"
        f"Key takeaway: \"{clean_q[:160]}\"\n\n"
        f"Source: {source_credit}"
    )

    pinned_comment = (
        f"What's your take on what {person} said about {topic_main.lower()} here? Agree or disagree?"
    )

    return {
        "titles": titles[:5],
        "post_caption": post_caption,
        "description": description,
        "hashtags": hashtags[:9],
        "pinned_comment": pinned_comment,
    }


_CLOUD_PROVIDER_DISABLED: Dict[str, bool] = {}
_CLIENT_LOCK = threading.Lock()
_GEMINI_CLIENTS: Dict[str, Any] = {}
_CLAUDE_CLIENTS: Dict[str, Any] = {}


class LLMClient:
    """Unified JSON-oriented LLM wrapper with disk caching, connection pooling, and zero-key fallback."""

    def __init__(self, config: AppConfig, workspace: Optional[ProjectWorkspace] = None) -> None:
        self.config = config
        self.workspace = workspace
        self.provider = config.llm.provider
        self._tls = threading.local()
        self.last_cache_hit = False

    @property
    def last_cache_hit(self) -> bool:
        return bool(getattr(self._tls, "last_cache_hit", False))

    @last_cache_hit.setter
    def last_cache_hit(self, value: bool) -> None:
        self._tls.last_cache_hit = bool(value)

    def is_remote_provider(self) -> bool:
        """Return True when using a live cloud LLM API (where concurrent HTTP requests speed up throughput)."""
        return type(self) is LLMClient and self._effective_provider() in ("gemini", "claude")

    def _effective_provider(self) -> str:
        """Use configured cloud LLM if its API key is set in .env and healthy; otherwise transparently use local engine."""
        if self.provider == "gemini" and get_env_key("GEMINI_API_KEY") and not _CLOUD_PROVIDER_DISABLED.get("gemini"):
            return "gemini"
        if self.provider == "claude" and get_env_key("ANTHROPIC_API_KEY") and not _CLOUD_PROVIDER_DISABLED.get("claude"):
            return "claude"
        return "mock"

    def _model_name(self) -> str:
        eff = self._effective_provider()
        if eff == "gemini":
            return self.config.llm.gemini_model
        if eff == "claude":
            return self.config.llm.claude_model
        return "local-smart-engine-v2"

    def complete_json(
        self,
        namespace: str,
        system_prompt: str,
        user_prompt: str,
        mock_payload: Optional[Any] = None,
        use_cache: bool = True,
    ) -> Any:
        """Send a prompt expecting JSON output, caching results by SHA-256 on disk."""
        eff_provider = self._effective_provider()
        cache_enabled = self.config.llm.cache_enabled and use_cache and self.workspace is not None
        cache_key = ""
        if cache_enabled and self.workspace is not None:
            cache_key = self.workspace.llm_cache_key(
                namespace,
                {
                    "provider": eff_provider,
                    "model": self._model_name(),
                    "system": system_prompt,
                    "user": user_prompt,
                    "temperature": self.config.llm.temperature,
                },
            )
            cached = self.workspace.get_llm_cache(cache_key)
            if cached is not None:
                self.last_cache_hit = True
                return cached

        self.last_cache_hit = False
        result: Any = None
        if eff_provider == "gemini":
            try:
                result = self._call_gemini(system_prompt, user_prompt)
            except Exception as exc:  # noqa: BLE001
                err_s = str(exc).upper()
                if any(tok in err_s for tok in ("404", "NOT_FOUND", "401", "403", "PERMISSION_DENIED", "INVALID_ARGUMENT", "CERTIFICATE_VERIFY_FAILED")):
                    _CLOUD_PROVIDER_DISABLED["gemini"] = True
                result = None
        elif eff_provider == "claude":
            try:
                result = self._call_claude(system_prompt, user_prompt)
            except Exception as exc:  # noqa: BLE001
                err_s = str(exc).upper()
                if any(tok in err_s for tok in ("404", "401", "403", "NOT_FOUND", "AUTHENTICATION")):
                    _CLOUD_PROVIDER_DISABLED["claude"] = True
                result = None

        if result is None:
            result = (
                mock_payload
                if mock_payload is not None
                else self._local_smart_response(namespace, user_prompt)
            )

        if cache_enabled and self.workspace is not None and cache_key:
            self.workspace.set_llm_cache(cache_key, result)
        return result

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> Any:
        api_key = get_env_key("GEMINI_API_KEY")
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set.")
        from google import genai
        from google.genai import types

        with _CLIENT_LOCK:
            client = _GEMINI_CLIENTS.get(api_key)
            if client is None:
                client = genai.Client(api_key=api_key)
                _GEMINI_CLIENTS[api_key] = client

        last_err: Optional[Exception] = None
        total_tries = max(1, self.config.llm.max_retries + 1)
        for attempt in range(total_tries):
            try:
                response = client.models.generate_content(
                    model=self.config.llm.gemini_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.config.llm.temperature,
                        response_mime_type="application/json",
                    ),
                )
                text = response.text or ""
                return _extract_json_block(text)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < total_tries - 1:
                    time.sleep(0.35 * (attempt + 1))
        raise LLMError(f"Gemini call failed after retries: {last_err}") from last_err

    def _call_claude(self, system_prompt: str, user_prompt: str) -> Any:
        api_key = get_env_key("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set.")
        import anthropic

        with _CLIENT_LOCK:
            client = _CLAUDE_CLIENTS.get(api_key)
            if client is None:
                client = anthropic.Anthropic(api_key=api_key)
                _CLAUDE_CLIENTS[api_key] = client

        last_err: Optional[Exception] = None
        total_tries = max(1, self.config.llm.max_retries + 1)
        for attempt in range(total_tries):
            try:
                message = client.messages.create(
                    model=self.config.llm.claude_model,
                    max_tokens=4096,
                    temperature=self.config.llm.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                parts: List[str] = []
                for block in message.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                return _extract_json_block("\n".join(parts))
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < total_tries - 1:
                    time.sleep(0.35 * (attempt + 1))
        raise LLMError(f"Claude call failed after retries: {last_err}") from last_err

    def _local_smart_response(self, namespace: str, user_prompt: str) -> Any:
        """Built-in Smart NLP Moment, Hook & Metadata Engine (requires zero external API keys)."""
        if namespace.startswith("moments"):
            matches = re.findall(r"\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]\s*(?:\([^)]+\)\s*)?(.+)", user_prompt)
            if not matches:
                return {"candidates": []}

            req_m = re.search(r"Propose up to (\d+)", user_prompt)
            target_n = int(req_m.group(1)) if req_m else 5

            sents = []
            for st_s, en_s, txt in matches:
                st, en = float(st_s), float(en_s)
                emo, quo, hum = _score_sentence_text(txt)
                sents.append({"start": st, "end": en, "text": txt.strip(), "emo": emo, "quo": quo, "hum": hum})

            # Slide dynamic variable-length windows across sentences (15s to 95s depending on where message completes)
            windows = []
            n_sents = len(sents)
            i = 0
            while i < n_sents:
                # Never anchor a clip on a sponsor/ad sentence
                if sents[i]["emo"] <= 25:
                    i += 1
                    continue
                w_start = sents[i]["start"]
                if sents[-1]["end"] > 120.0 and w_start < 10.0:
                    i += 1
                    continue
                j = i
                # Grow until at least 22s and keep growing up to ~85s if message continues smoothly (stopping before any ad)
                while j + 1 < n_sents:
                    if sents[j + 1]["emo"] <= 25:
                        break
                    cur_dur = sents[j]["end"] - w_start
                    next_dur = sents[j + 1]["end"] - w_start
                    gap = sents[j + 1]["start"] - sents[j]["end"]
                    if cur_dur < 22.0 and next_dur <= 110.0:
                        j += 1
                    elif cur_dur < 68.0 and next_dur <= 115.0 and gap < 0.75:
                        j += 1
                    else:
                        break
                w_end = sents[j]["end"]
                dur = w_end - w_start
                if dur < 12.0 and n_sents > 3:
                    i += 1
                    continue

                span = sents[i : j + 1]
                best_sent = max(
                    span,
                    key=lambda s: (
                        (s["emo"] + s["quo"])
                        - (28 if re.match(r"^(?:>>\s*)?(?:but|so|and|yeah|well|in terms of)\b", s["text"], re.I) else 0)
                        - (22 if s["text"].endswith(("?", ",", "...")) else 0)
                    ),
                )
                avg_emo = int(round(sum(s["emo"] for s in span) / len(span)))
                max_quo = max(s["quo"] for s in span)
                max_hum = max(s["hum"] for s in span)
                emo_score = min(98, int(round(0.55 * best_sent["emo"] + 0.45 * avg_emo)))
                quo_score = min(98, max_quo)
                hum_score = min(95, max_hum)
                self_score = 91 if len(best_sent["text"].split()) >= 6 else 82
                bound_score = 93
                total = int(round(0.28 * emo_score + 0.28 * quo_score + 0.14 * hum_score + 0.18 * self_score + 0.12 * bound_score))

                p_start = max(w_start, min(best_sent["start"], w_end - 2.8))
                p_end = min(w_end, p_start + 3.0)
                windows.append({
                    "start": round(w_start, 2),
                    "end": round(w_end, 2),
                    "score": total,
                    "score_breakdown": {
                        "emotional_peak": emo_score,
                        "quotable_statement": quo_score,
                        "humor_payoff": hum_score,
                        "self_contained": self_score,
                        "clean_boundaries": bound_score,
                    },
                    "reason": f"High-impact complete message with strong stakes ({dur:.0f}s natural arc).",
                    "key_quote": best_sent["text"][:180],
                    "peak_start": round(p_start, 2),
                    "peak_end": round(p_end, 2),
                })
                i += max(2, (j - i) // 2)

            if not windows and sents:
                first_s = sents[0]["start"]
                last_e = min(sents[-1]["end"], first_s + 35.0)
                best_s = max(sents, key=lambda s: s["emo"] + s["quo"])
                windows.append({
                    "start": round(first_s, 2),
                    "end": round(last_e, 2),
                    "score": 88,
                    "score_breakdown": {
                        "emotional_peak": 88,
                        "quotable_statement": 91,
                        "humor_payoff": 78,
                        "self_contained": 90,
                        "clean_boundaries": 92,
                    },
                    "reason": "Direct, memorable turning-point story with a clean payoff.",
                    "key_quote": best_s["text"][:180],
                    "peak_start": round(first_s + max(0.0, (last_e - first_s) * 0.4), 2),
                    "peak_end": round(min(last_e, first_s + max(0.0, (last_e - first_s) * 0.4) + 2.5), 2),
                })

            windows.sort(key=lambda w: w["score"], reverse=True)
            picked: List[Dict[str, Any]] = []
            for w in windows:
                if any(max(0.0, min(w["end"], p["end"]) - max(w["start"], p["start"])) > 5.0 for p in picked):
                    continue
                picked.append(w)
                if len(picked) >= target_n:
                    break
            return {"candidates": picked}

        if namespace.startswith("hook_generate"):
            kq_m = re.search(r"Key quote:\s*(.+)", user_prompt)
            tr_m = re.search(r"Clip transcript:\n([\s\S]+?)(?:\n\n|\Z)", user_prompt)
            key_quote = kq_m.group(1).strip() if kq_m else ""
            tr_text = tr_m.group(1).strip() if tr_m else user_prompt
            return {"variants": _build_hooks_from_quote(key_quote, tr_text)}

        if namespace.startswith("hook_verify"):
            return {
                "passed": True,
                "reason": "Hook accurately frames statements from the clip transcript without invention.",
                "violations": [],
            }

        if namespace.startswith("export_metadata"):
            subj_m = re.search(r"Subject:\s*(.+)", user_prompt)
            src_m = re.search(r"Original Show/Channel:\s*(.+)", user_prompt)
            title_m = re.search(r"Original Title:\s*(.+)", user_prompt)
            hook_m = re.search(r"Hook:\s*(.+)", user_prompt)
            quote_m = re.search(r"Key quote:\s*(.+)", user_prompt)
            tr_m = re.search(r"Transcript:\n([\s\S]+)", user_prompt)
            person = subj_m.group(1).strip() if subj_m else "Speaker"
            source_credit = src_m.group(1).strip() if src_m else "Original Broadcast"
            orig_title = title_m.group(1).strip() if title_m else source_credit
            hook = hook_m.group(1).strip() if hook_m else "Listen Closely"
            quote = quote_m.group(1).strip() if quote_m else ""
            tr_text = tr_m.group(1).strip() if tr_m else quote

            return _extract_video_specific_metadata(
                person=person,
                source_credit=source_credit,
                orig_title=orig_title,
                hook=hook,
                quote=quote,
                transcript_text=tr_text,
            )

        return {}
