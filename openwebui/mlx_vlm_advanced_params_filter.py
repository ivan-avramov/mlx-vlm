"""
title: MLX-VLM Advanced Params
author: ivan-avramov
version: 0.1.0
required_open_webui_version: 0.9.2
description: |
  Per-chat advanced parameter overrides for the local mlx-vlm server.
  Exposes thinking, sampling, and loop-mitigation knobs as UserValves so
  each user (or each chat, when scoped) can tune them in the UI without
  touching server config or the model's Custom Parameters JSON.

  Knobs the mlx-vlm server actually applies (verified against server.py):
    - enable_thinking, thinking_budget, thinking_start_token
    - temperature, top_p, top_k, min_p, max_tokens, seed
    - repetition_penalty (also accepted as repeat_penalty alias)
    - logit_bias

  Anything left at None / empty is NOT injected, so the server's defaults
  (or the model's Custom Parameters) keep applying. This filter only
  *overrides* — it never silently sets a value the user didn't ask for.
"""

from typing import Optional

from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        """Admin-side global toggles for this filter."""

        enabled: bool = Field(
            True,
            description="Master switch. Disable to make the filter a no-op without uninstalling.",
        )
        target_model_substring: str = Field(
            "",
            description=(
                "If non-empty, only apply this filter when the request's model id "
                "contains this substring (case-insensitive). Useful when this filter "
                "is attached globally but you only want it active for the mlx-vlm "
                "models, e.g. 'gemma'. Leave empty to apply to all models."
            ),
        )

    class UserValves(BaseModel):
        """Per-user / per-chat overrides. Settable from the chat UI."""

        # --- Thinking ---
        enable_thinking: Optional[bool] = Field(
            None,
            description="Enable the model's <think> reasoning channel. Leave unset to defer to server default.",
        )
        thinking_budget: Optional[int] = Field(
            None,
            description=(
                "Hard cap on thinking tokens. Server auto-budget is 80%% of "
                "max_tokens; set this to override. Recommend 8000-12000 to "
                "break self-reverification loops on multi-constraint prompts."
            ),
            ge=1,
        )
        thinking_start_token: Optional[str] = Field(
            None,
            description="Override the chat-template's thinking start token. Advanced: leave unset unless you know you need it.",
        )

        # --- Sampling ---
        temperature: Optional[float] = Field(
            None,
            description="Sampling temperature. 0.3-0.5 reduces loop risk on multi-constraint prompts; 0.7-1.0 is more creative.",
            ge=0.0,
            le=2.0,
        )
        top_p: Optional[float] = Field(
            None, description="Nucleus sampling threshold (0,1].", gt=0.0, le=1.0
        )
        top_k: Optional[int] = Field(
            None, description="Top-K sampling. 0 disables.", ge=0
        )
        min_p: Optional[float] = Field(
            None,
            description="Min-P sampling threshold. 0 disables.",
            ge=0.0,
            le=1.0,
        )
        max_tokens: Optional[int] = Field(
            None,
            description="Max output tokens (includes thinking). Caps total work.",
            ge=1,
        )
        seed: Optional[int] = Field(
            None,
            description=(
                "Random seed for reproducibility. Best-effort under continuous "
                "batching — interleaved batches share PRNG state."
            ),
        )

        # --- Repetition ---
        repetition_penalty: Optional[float] = Field(
            None,
            description=(
                "Down-weights tokens already in the recent context. 1.08-1.15 "
                "is the loop-mitigation knob — directly attacks 'one last "
                "thing' / 'wait, let me re-verify' phrase reinforcement on "
                "multi-constraint prompts. Leave unset (or 1.0) for no penalty."
            ),
            gt=0.0,
            le=2.0,
        )

    def __init__(self):
        self.valves = self.Valves()

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """Mutate the outbound chat-completion body before it hits the server.

        OpenWebUI calls inlet() with the request body and the current user
        context. We copy each non-None UserValve into the body. Existing
        keys take precedence — so if the user set something via the model's
        Custom Parameters JSON, this filter won't clobber it.
        """
        if not self.valves.enabled:
            return body

        # Optional model-id gate so the filter can be attached globally
        # without affecting non-mlx-vlm targets.
        substring = self.valves.target_model_substring.strip().lower()
        if substring:
            model_id = str(body.get("model", "")).lower()
            if substring not in model_id:
                return body

        # Pull the per-user valves. OpenWebUI provides them on __user__.
        user_valves = None
        if __user__ and isinstance(__user__, dict):
            user_valves = __user__.get("valves")
        if user_valves is None:
            return body

        # Normalize to a dict regardless of whether OpenWebUI hands us a
        # Pydantic instance or a plain dict.
        if hasattr(user_valves, "model_dump"):
            overrides = user_valves.model_dump(exclude_none=True)
        elif hasattr(user_valves, "dict"):
            overrides = user_valves.dict(exclude_none=True)
        elif isinstance(user_valves, dict):
            overrides = {k: v for k, v in user_valves.items() if v is not None}
        else:
            return body

        # Don't overwrite values the user already pinned via Custom Params.
        for key, value in overrides.items():
            body.setdefault(key, value)

        return body
