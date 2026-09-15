"""Anonymous contact ID — stable per (device_sn, session_id), mirroring AnonymousContactId.java."""

from __future__ import annotations

import uuid

ANON_PREFIX = "anon-"


def derive_anonymous_contact_id(device_sn: str, session_id: str) -> str:
    """Derive a stable anonymous contactId from (device_sn, session_id).

    Same input -> same output (uuid3), so multi-turn conversations keep
    a consistent identity without persisting anything.
    """
    seed = f"{device_sn}:{session_id}"
    return ANON_PREFIX + str(uuid.uuid3(uuid.NAMESPACE_DNS, seed))


def is_anonymous(contact_id: str | None) -> bool:
    """True when contact_id is an anonymous marker (anon- prefix)."""
    return bool(contact_id) and contact_id.startswith(ANON_PREFIX)
