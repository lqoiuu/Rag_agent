"""Tool permissions: who may call what, declared by the caller and never by the model.

Until now the only authorisation rule lived inside the tool bodies as a single owner check
(``_require_owner``). That is not a model of permissions, it is one rule per entity, and it
cannot express "this caller may read but not write" or "a support agent may look up somebody
else's device".

Two properties of this module carry the security weight:

1. **The role comes from the caller, never from model output.** ``assign_role`` exists only
   so a trusted entry point (a CLI flag, a session value) can *lower* a role. There is no
   code path that raises one, because a tool that trusts the model's claim about its own
   privileges is not a permission system.
2. **The default is the least privilege that still works.** A caller that says nothing is an
   ``end_user`` who may read their own records and ask for a ticket, and nothing else.

Enforcement happens at the tool boundary, before any repository access, so a denied call
cannot cause a read, a write, or a side effect that a later check would have to undo.

Module layering rule: this module must not import ``tools.business``. ``business`` imports
these types to enforce the whitelist, so importing back would be a cycle. The tool-name
constants therefore live in ``tools.business`` and the sets below spell the names out; if a
tool is renamed, the test that compares these sets against ``TOOL_NAMES`` fails immediately
rather than letting a stale name silently deny every call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rag_agent.tools.errors import PermissionDeniedError


class Role(StrEnum):
    """Who is asking. Ordered from least to most privileged."""

    END_USER = "end_user"
    SUPPORT_AGENT = "support_agent"


#: The tools each role may call at all. This is the whitelist: a tool absent from the set is
#: refused before its body runs, regardless of arguments. The names are the values of the
#: constants in ``tools.business``; ``test_tool_permissions`` asserts they still match.
ROLE_ALLOWED_TOOLS: dict[Role, frozenset[str]] = {
    Role.END_USER: frozenset({"user.lookup", "device.lookup", "order.lookup", "ticket.create"}),
    Role.SUPPORT_AGENT: frozenset(
        {"user.lookup", "device.lookup", "order.lookup", "ticket.create"}
    ),
}

#: Whether a role may act on records that are not its own. ``end_user`` may not; the owner
#: check inside the tools stays as the second, argument-level barrier.
ROLES_WITH_CROSS_USER_READ: frozenset[Role] = frozenset({Role.SUPPORT_AGENT})


@dataclass(frozen=True, slots=True)
class ToolPermissions:
    """The caller's declared identity and role, validated before any tool runs."""

    user_id: str | None
    role: Role = Role.END_USER

    def allows_tool(self, tool_name: str) -> bool:
        allowed = ROLE_ALLOWED_TOOLS.get(self.role, frozenset())
        return tool_name in allowed

    @property
    def may_read_other_users(self) -> bool:
        return self.role in ROLES_WITH_CROSS_USER_READ

    def require_tool(self, tool_name: str) -> None:
        """Raise a classified denial when this role may not call the tool at all."""

        if not self.allows_tool(tool_name):
            raise PermissionDeniedError(
                f"role {self.role} may not call {tool_name}",
                details={"role": str(self.role), "tool": tool_name},
            )

    def require_own_record(self, *, owner_user_id: str, tool_name: str) -> None:
        """Raise when a role that may not read across users asks about somebody else.

        Kept separate from :meth:`require_tool` so the two failures stay distinguishable in
        an audit log: "this role never has this tool" is a different event from "this role
        has the tool but not for this subject".
        """

        if self.may_read_other_users:
            return
        if self.user_id is None or self.user_id != owner_user_id:
            raise PermissionDeniedError(
                f"role {self.role} may only read its own records",
                details={"tool": tool_name, "owner": owner_user_id},
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "role": str(self.role),
            "cross_user_read": self.may_read_other_users,
        }


def assign_role(requested: str | None, *, granted: Role = Role.END_USER) -> Role:
    """Resolve a requested role, never above the granted one.

    This is the only way a role is produced from outside. It can lower the granted role but
    never raise it, so a caller (or a model, if one ever influenced this value) cannot ask
    its way into more privilege than the entry point gave it.
    """

    if requested is None or not requested.strip():
        return granted
    try:
        candidate = Role(requested.strip().lower())
    except ValueError:
        return granted
    order = {Role.END_USER: 0, Role.SUPPORT_AGENT: 1}
    return candidate if order[candidate] <= order[granted] else granted


__all__ = [
    "ROLES_WITH_CROSS_USER_READ",
    "ROLE_ALLOWED_TOOLS",
    "Role",
    "ToolPermissions",
    "assign_role",
]
