"""Tool permissions: the whitelist, and the ways privilege must not escalate."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from rag_agent.storage import BusinessRepository
from rag_agent.tools import DEVICE_LOOKUP, device_lookup
from rag_agent.tools.errors import PermissionDeniedError
from rag_agent.tools.models import DeviceLookupArgs
from rag_agent.tools.permissions import (
    ROLE_ALLOWED_TOOLS,
    Role,
    ToolPermissions,
    assign_role,
)


@contextmanager
def seeded_repository() -> Iterator[BusinessRepository]:
    with BusinessRepository(":memory:") as repository:
        repository.seed_from_file("data/business/seed.json")
        yield repository


def test_default_role_is_the_least_privileged_one() -> None:
    permissions = ToolPermissions(user_id="U1001")

    assert permissions.role is Role.END_USER
    assert permissions.may_read_other_users is False


def test_the_whitelist_names_the_tools_that_actually_exist() -> None:
    """The whitelist spells out tool names, so it must be checked against the real ones.

    The names cannot be imported into the permissions module without creating an import
    cycle, so this test is the thing that stops a rename from turning into "every call is
    denied" or, worse, a stale name that quietly stops matching.
    """

    from rag_agent.tools import TOOL_NAMES

    for role in Role:
        assert ROLE_ALLOWED_TOOLS[role] == frozenset(TOOL_NAMES)


def test_a_tool_outside_the_whitelist_is_refused_before_anything_runs() -> None:
    permissions = ToolPermissions(user_id="U1001")

    with pytest.raises(PermissionDeniedError) as excinfo:
        permissions.require_tool("device.delete_all")

    assert excinfo.value.code == "permission_denied"
    assert excinfo.value.details == {"role": "end_user", "tool": "device.delete_all"}


def test_an_end_user_may_not_read_someone_elses_record() -> None:
    permissions = ToolPermissions(user_id="U1001")

    with pytest.raises(PermissionDeniedError):
        permissions.require_own_record(owner_user_id="U1002", tool_name=DEVICE_LOOKUP)


def test_a_support_agent_may_read_across_users() -> None:
    permissions = ToolPermissions(user_id="U9001", role=Role.SUPPORT_AGENT)

    permissions.require_own_record(owner_user_id="U1002", tool_name=DEVICE_LOOKUP)

    assert permissions.may_read_other_users is True


def test_an_anonymous_end_user_may_not_read_any_record() -> None:
    """No declared identity means no owned records, which must not mean "all of them"."""

    permissions = ToolPermissions(user_id=None)

    with pytest.raises(PermissionDeniedError):
        permissions.require_own_record(owner_user_id="U1001", tool_name=DEVICE_LOOKUP)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (None, Role.END_USER),
        ("", Role.END_USER),
        ("   ", Role.END_USER),
        ("end_user", Role.END_USER),
        ("support_agent", Role.END_USER),  # granted 是 end_user，只能持平或下调
        ("SUPPORT_AGENT", Role.END_USER),
        ("administrator", Role.END_USER),  # 未知角色回落到 granted，而不是报错或放行
        ("../root", Role.END_USER),
    ],
)
def test_assign_role_never_exceeds_the_granted_role(requested: str | None, expected: Role) -> None:
    """The whole point of this function: a requested role can never Escalate."""

    assert assign_role(requested) is expected


def test_assign_role_can_lower_a_granted_role() -> None:
    assert assign_role("end_user", granted=Role.SUPPORT_AGENT) is Role.END_USER
    assert assign_role("support_agent", granted=Role.SUPPORT_AGENT) is Role.SUPPORT_AGENT


def test_permissions_render_for_an_audit_line() -> None:
    permissions = ToolPermissions(user_id="U1001", role=Role.SUPPORT_AGENT)

    assert permissions.as_dict() == {
        "user_id": "U1001",
        "role": "support_agent",
        "cross_user_read": True,
    }


# ---------------------------------------------------------------- enforcement

FOREIGN_DEVICE = DeviceLookupArgs(device_id="D2003", user_id="U1001")


def test_an_end_user_is_denied_a_foreign_device_through_the_real_tool() -> None:
    """The whitelist and the owner rule are enforced by the tool, not only by this module."""

    with seeded_repository() as repository:
        result = device_lookup(
            FOREIGN_DEVICE,
            repository=repository,
            permissions=ToolPermissions(user_id="U1001"),
        )

    assert result.status == "error"
    assert result.error_code == "permission_denied"


def test_a_support_agent_may_read_a_foreign_device_through_the_real_tool() -> None:
    """The role has to mean something, or the permission model is decoration."""

    with seeded_repository() as repository:
        result = device_lookup(
            FOREIGN_DEVICE,
            repository=repository,
            permissions=ToolPermissions(user_id="U9001", role=Role.SUPPORT_AGENT),
        )

    assert result.status == "ok"
    assert (result.data or {}).get("device", {}).get("device_id") == "D2003"


def test_omitting_permissions_stays_strict() -> None:
    """A caller that forgets to pass them must not accidentally gain the cross-user read."""

    with seeded_repository() as repository:
        result = device_lookup(FOREIGN_DEVICE, repository=repository)

    assert result.error_code == "permission_denied"
