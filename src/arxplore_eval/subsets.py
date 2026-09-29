from collections.abc import Sequence
from typing import Protocol

SUBSETS = ("all", "ko", "en", "known_item", "llm_synth", "manual")
CATEGORY_PREFIX = "category:"
BEHAVIOR_PREFIX = "behavior:"


class SubsetMember(Protocol):
    @property
    def lang(self) -> str: ...

    @property
    def source(self) -> str: ...

    @property
    def category(self) -> str: ...

    @property
    def expected_behavior(self) -> str: ...


def in_subset(member: SubsetMember, subset: str) -> bool:
    if subset == "all":
        return True
    if subset in ("ko", "en"):
        return member.lang == subset
    if subset.startswith(CATEGORY_PREFIX):
        return member.category == subset[len(CATEGORY_PREFIX) :]
    if subset.startswith(BEHAVIOR_PREFIX):
        return member.expected_behavior == subset[len(BEHAVIOR_PREFIX) :]
    return member.source == subset


def build_subsets(members: Sequence[SubsetMember], base: Sequence[str] = SUBSETS) -> list[str]:
    categories = sorted({member.category for member in members if member.category})
    behaviors = sorted({member.expected_behavior for member in members})
    subsets = [*base, *(f"{CATEGORY_PREFIX}{name}" for name in categories)]
    if len(behaviors) > 1:
        subsets.extend(f"{BEHAVIOR_PREFIX}{name}" for name in behaviors)
    return subsets
