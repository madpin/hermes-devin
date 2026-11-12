"""Devin CLI (Cognition) as a Hermes model provider.

Hermes keeps its own agent loop, tools, memory and skills. Devin only supplies the model: every request is one
``devin -p`` run with Devin's tools, MCP servers, skills and rules switched off, and Hermes' system prompt and
tool schemas delivered as session context. See README.md.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

try:  # a package when Hermes loads it, flat modules when tests or scripts do
    from . import devin_client, devin_models
except ImportError:  # pragma: no cover
    import devin_client
    import devin_models

logger = logging.getLogger(__name__)

DEVIN_MARKER_BASE_URL = devin_client.MARKER_BASE_URL
DEFAULT_CONTEXT = 200_000


class DevinProfile(ProviderProfile):
    """Devin CLI as an external process: no REST endpoint, Hermes talks to a child per request."""

    def _command(self) -> str | None:
        try:
            from hermes_cli.auth import resolve_external_process_provider_credentials

            return resolve_external_process_provider_credentials(self.name).get("command")
        except Exception:
            return None

    def create_client(self, **client_kwargs: Any) -> Any:
        return devin_client.DevinClient(**client_kwargs)

    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None,
                     timeout: float = 30.0) -> list[str] | None:
        try:
            return devin_models.picker_ids(devin_models.catalog(self._command() or "devin"))
        except Exception as exc:  # never break the picker
            logger.debug("devin fetch_models failed: %s", exc)
            return devin_models.picker_ids(devin_models.PINNED)

    def get_model_context_length(self, model: str) -> int | None:
        try:
            override = int(os.environ.get("HERMES_DEVIN_CONTEXT", "") or 0)
        except ValueError:
            override = 0
        if override > 0:
            return override
        try:
            return devin_models.context_for(model, devin_models.catalog(self._command() or "devin")) or DEFAULT_CONTEXT
        except Exception:
            return DEFAULT_CONTEXT


devin = DevinProfile(
    name="devin",
    aliases=("devin-cli", "cognition-devin"),
    display_name="Devin CLI (Cognition)",
    description="Run Hermes on the models of your Devin account through the Devin CLI",
    api_mode="chat_completions",
    env_vars=(),
    base_url=DEVIN_MARKER_BASE_URL,
    auth_type="external_process",
    process_command="devin",
    process_args=(),
    process_command_env_vars=("HERMES_DEVIN_COMMAND", "DEVIN_CLI_PATH"),
    default_aux_model="claude-haiku-5-5-low",
    fallback_models=tuple(devin_models.picker_ids(devin_models.PINNED)),
    supports_streaming=False,
)

register_provider(devin)
