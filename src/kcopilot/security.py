"""Role-based access control.

Access is enforced at *retrieval time*, before any text reaches the LLM. The model never sees a chunk
the user is not allowed to read, so a prompt injection cannot talk it into leaking one.

In production, replace UserDirectory with your identity provider (OIDC/SAML group claims).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class User:
    user_id: str
    name: str
    roles: tuple[str, ...]

    @property
    def effective_roles(self) -> list[str]:
        # Everybody can read documents marked "all".
        return sorted({*self.roles, "all"})


class UnknownUserError(KeyError):
    pass


class UserDirectory:
    def __init__(self, users: dict[str, User]):
        self._users = users

    @classmethod
    def from_file(cls, path: Path) -> UserDirectory:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        users = {
            u["user_id"]: User(u["user_id"], u.get("name", u["user_id"]), tuple(r.lower() for r in u["roles"]))
            for u in data["users"]
        }
        return cls(users)

    def get(self, user_id: str) -> User:
        try:
            return self._users[user_id]
        except KeyError as exc:
            raise UnknownUserError(user_id) from exc

    def all(self) -> list[User]:
        return list(self._users.values())


def can_read(user: User, allowed_roles: list[str]) -> bool:
    return bool(set(allowed_roles) & set(user.effective_roles))
