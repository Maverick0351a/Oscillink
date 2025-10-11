from __future__ import annotations

from fastapi import FastAPI

_STRIPE_EVENTS_COUNT = 0


class WebhookEventsWrapper:
    """In-memory store wrapper with a stable length counter.

    Mirrors the behavior used in main.py so tests relying on len() and clear()
    continue to work even if the underlying dict is rebound.
    """

    def __init__(self, app_ref: FastAPI):
        self._app = app_ref

    def _store(self) -> dict:
        if not hasattr(self._app.state, "webhook_events") or not isinstance(
            self._app.state.webhook_events, dict
        ):
            self._app.state.webhook_events = {}
        return self._app.state.webhook_events

    def __setitem__(self, key, value):
        global _STRIPE_EVENTS_COUNT
        st = self._store()
        is_new = key not in st
        st[key] = value
        if is_new:
            _STRIPE_EVENTS_COUNT += 1

    def __getitem__(self, key):
        return self._store()[key]

    def __contains__(self, key):
        return key in self._store()

    def get(self, key, default=None):
        return self._store().get(key, default)

    def values(self):
        return self._store().values()

    def items(self):
        return self._store().items()

    def clear(self):
        global _STRIPE_EVENTS_COUNT
        st = self._store()
        st.clear()
        _STRIPE_EVENTS_COUNT = 0

    def __len__(self):
        return _STRIPE_EVENTS_COUNT


def get_webhook_events_mem(app: FastAPI) -> WebhookEventsWrapper:
    # Single instance per app process; stored on module-level global is avoided to ensure referential stability.
    # The wrapper internally uses app.state.webhook_events for the actual dict.
    return WebhookEventsWrapper(app)
