# app/schemas.py
"""Request / response models and the structured-output DTOs used by the agents.

Two kinds of models live here:

* API models (`QueryRequest`, `HistoryRequest`, `UserCreateRequest`) — validate
  what callers send us.
* LLM output models (`TripPlan`, guardrails, `SpecificSearchQuery`, ...) — their
  field descriptions are part of the prompt, and their validators repair the
  small, predictable ways a model's JSON drifts (nulls on required lists,
  a country listed next to its own city) instead of failing the whole turn.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "QueryRequest",
    "HistoryRequest",
    "UserCreateRequest",
    "Output_Format",
    "global_input_guardrail",
    "global_travel_guardrail",
    "TripDriver",
    "Pax",
    "TripPlan",
    "Feedback",
    "SpecificSearchQuery",
    "ExploreResponse",
    "dedupe_destinations",
]


# ─── Shared types ────────────────────────────────────────────────────

UserType = Literal["logged-in", "visitor"]
Priority = Literal["incidental", "preferred", "important", "primary", "exclusive"]


# ─── API request models ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Incoming chat request."""

    message: str
    user_id: str = Field(..., description="Logged-in user's userId, or the visitor's vId — see user_type.")
    user_type: Optional[UserType] = Field(None, description="Which kind of id user_id carries.")
    reference: str = Field(..., description="cId — client/site identifier (e.g. 'hiptraveler').")
    param: str
    # Required-but-nullable: callers must send the key, null starts a new thread.
    threadId: Optional[str]
    is_pro: bool = False
    plan: bool = Field(False, description="If True: save & retrieve conversation memory via Zep.")

    @field_validator("is_pro", "plan", mode="before")
    @classmethod
    def _null_flag_is_false(cls, v):
        """Older clients send explicit null for unset flags."""
        return False if v is None else v

    @model_validator(mode="after")
    def _check_identity(self) -> "QueryRequest":
        """user_id must be non-blank whenever the caller declares a user_type.

        The credits service requires exactly one of userId/vId, so a blank
        identifier here would come back as DENIED_INVALID_IDENTITY. Fail at our
        own boundary instead, with a message that says which field is wrong.
        """
        if self.user_type and not (self.user_id or "").strip():
            raise ValueError(f"user_id is required when user_type is '{self.user_type}'")
        return self

    @property
    def credits_identity(self) -> tuple[Optional[str], Optional[str]]:
        """Return (userId, vId) for the credits API — exactly one is ever set.

        Falls back to treating the caller as a visitor when user_type is absent,
        so requests from a not-yet-updated frontend still resolve to one identity.
        """
        if self.user_type == "logged-in":
            return self.user_id, None
        return None, self.user_id


class HistoryRequest(BaseModel):
    session_id: str
    last_n: int = Field(20, ge=1, le=500)


class UserCreateRequest(BaseModel):
    user_id: str
    email: str
    first_name: str
    last_name: str


# ─── Agent output models ─────────────────────────────────────────────

class Output_Format(BaseModel):  # noqa: N801 — name kept for existing imports
    """Output structure for the main agent."""

    answer: str


class global_input_guardrail(BaseModel):  # noqa: N801 — name kept for existing imports
    isValid: bool
    reason: str
    isTravelRelated: bool
    isMemoryQuery: bool
    isCustomerService: bool
    isRealtime: bool
    solution: str


class global_travel_guardrail(BaseModel):  # noqa: N801 — name kept for existing imports
    isValid: bool
    reason: str
    isTravelRelated: bool
    isPlanRelated: bool
    travel_type: str


# ─── Trip planner ────────────────────────────────────────────────────

class TripDriver(BaseModel):
    """Structured intent driver — why a theme matters and how central it is."""

    theme: str = Field(..., description="Travel theme (e.g. 'fishing', 'beaches', 'food', 'hiking')")
    priority: Priority = Field(..., description="How central this theme is to the trip")
    score: float = Field(
        ...,
        description="Priority score 0.0–1.0: incidental≈0.15, preferred≈0.3, important≈0.55, primary≈0.8, exclusive≈0.95",
    )
    confidence: float = Field(..., description="Confidence in this classification 0.0–1.0")
    user_evidence: str = Field(
        ..., description="Exact phrase from a user message that justifies this priority — never from assistant text"
    )
    destination_driver: bool = Field(
        ..., description="True if this theme is what caused the user to choose (or search for) the destination"
    )
    specific_activity: Optional[str] = Field(
        None,
        description=(
            "The exact activity named by the user when it is more specific than the theme "
            "(e.g. theme='adventure', specific_activity='paragliding'). "
            "Null when the activity and theme are the same word."
        ),
    )
    desired_frequency: Optional[str] = Field(
        None, description="How often: 'once', 'multiple_days', 'daily', 'throughout'"
    )

    @field_validator("score", "confidence", mode="before")
    @classmethod
    def _clamp_unit_interval(cls, v):
        """Models occasionally emit 1.2 or -0.1; clamp instead of failing the plan."""
        if v is None:
            return 0.0
        try:
            return min(1.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            return 0.0


class Pax(BaseModel):
    """Count of each traveler type."""

    adults: int = Field(
        0,
        description=(
            "Number of adults. Infer from context: 'solo'/'alone'/'just me'=1, "
            "'couple'/'romantic'/'two of us'=2, explicit number otherwise. "
            "0 if the user gave a group but never implied how many adults."
        ),
    )
    children: int = Field(0, description="Number of children.")
    infants: int = Field(0, description="Number of infants (babies).")
    elderly: int = Field(0, description="Number of elderly travelers.")

    @field_validator("adults", "children", "infants", "elderly", mode="before")
    @classmethod
    def _null_count_is_zero(cls, v):
        """The planner emits null (not 0) for counts the user never mentioned.

        Every count is contractually an integer for the frontend, so a null
        collapses to 0 rather than being rejected or dropped: a validation
        error here loses the whole trip plan over a field the user never
        talked about, and exclude_none would strip a null key entirely.
        Negative counts are nonsense and collapse to 0 as well.
        """
        if v is None:
            return 0
        try:
            return max(0, int(v))
        except (TypeError, ValueError):
            return 0


DESTINATIONS_DESCRIPTION = (
    "One entry per distinct destination the user wants to visit, correctly spelled even if "
    "the user misspelled them. Formats: US cities as 'City, ST, United States' "
    "(e.g., 'New York, NY, United States'); US states as 'State, United States' "
    "(e.g., 'California, United States'); other cities as 'City, Country' "
    "(e.g., 'Karachi, Pakistan', 'Da Nang, Vietnam'); regions as 'Region, Country' "
    "(e.g., 'Amalfi Coast, Italy'); countries bare (e.g., 'Japan', 'Hong Kong'). "
    "A country or state that only qualifies a city is part of that city's entry, NOT a "
    "separate destination: 'trip to Amman Jordan' → ['Amman, Jordan'], never "
    "['Amman, Jordan', 'Jordan']; 'San Francisco California' → "
    "['San Francisco, CA, United States']. List a country or state on its own only when "
    "the user wants to visit it with no city inside it named "
    "(e.g., 'Tokyo and then around Italy' → ['Tokyo, Japan', 'Italy'])."
)


def dedupe_destinations(destinations: list[str]) -> list[str]:
    """Drop blanks, case-insensitive duplicates, and bare parents of another entry.

        ['Amman, Jordan', 'Jordan']            → ['Amman, Jordan']
        ['Austin, TX, United States', 'TX']    → ['Austin, TX, United States']
        ['Tokyo, Japan', 'Italy']              → ['Tokyo, Japan', 'Italy']

    Text-only: a full state name next to its abbreviation ('California' vs
    'CA') is caught later, after geocoding, in preload_inventory.
    """
    cleaned = [d.strip() for d in destinations or [] if isinstance(d, str) and d.strip()]

    parents: set[str] = set()
    for d in cleaned:
        pieces = [p.strip().lower() for p in d.split(",") if p.strip()]
        parents.update(pieces[1:])

    seen: set[str] = set()
    out: list[str] = []
    for d in cleaned:
        key = d.lower()
        if key in seen or ("," not in d and key in parents):
            continue
        seen.add(key)
        out.append(d)
    return out


class TripPlan(BaseModel):
    """Structured DTO for an extracted trip plan."""

    startDate: Optional[str] = Field(
        None, description="The start date of the trip in MM-dd-yyyy format. Infer from context."
    )
    endDate: Optional[str] = Field(
        None, description="The end date of the trip in MM-dd-yyyy format. Infer from context."
    )
    numDays: Optional[int] = Field(
        None, description="Trip duration in days. Auto-calculated from startDate + endDate. Never included in feedback."
    )
    destinations: list[str] = Field(..., description=DESTINATIONS_DESCRIPTION)
    pax: Optional[Pax] = Field(None, description="Traveler counts. Null if not mentioned.")
    experienceTypes: Optional[list[str]] = Field(
        None,
        description="List of curated experience keywords (e.g., 'romantic', 'adventure', 'cultural', 'family friendly', 'relaxation').",
    )
    travelStyle: Optional[list[str]] = Field(
        None,
        description="List of travel styles (e.g., 'luxury', 'budget', 'backpacking', 'all inclusive', 'solo trip').",
    )
    activities: Optional[list[str]] = Field(
        None,
        description="List of requested activities, normalized to base form (e.g., 'hiking', 'wine tasting', 'snorkeling').",
    )
    weighted_activities: Optional[list[str]] = Field(
        None,
        description=(
            "The user's priority activities/interests, WEIGHT-ORDERED: index 0 = highest priority, "
            "last index = lowest. Extracted from everything the user emphasized or asked about across "
            "the WHOLE conversation (e.g. 'vegan restaurants', 'surfing', 'nightlife'). Preserve this "
            "order as-is — most important first, descending. Empty/null if none expressed."
        ),
    )
    themes: Optional[list[str]] = Field(
        None, description="List of themes from media or pop culture (e.g., 'James Bond', 'Midnight in Paris')."
    )
    pois: list[str] = Field(..., description="Explicitly mentioned Points of Interest.")
    feedback: Optional[list[str]] = Field(None, description="Array of missing field names that require user input.")
    month: Optional[str] = Field(None, description="Month extracted from conversation")
    summary: str = Field(
        "",
        description=(
            "A friendly, conversational acknowledgement of the current input followed by a "
            "question asking for the items in the 'feedback' list."
        ),
    )
    trip_drivers: Optional[list[TripDriver]] = Field(
        None, description="Structured intent drivers extracted from user messages only — see TRIP DRIVERS section"
    )

    @field_validator("destinations", "pois", mode="before")
    @classmethod
    def _null_list_is_empty(cls, v):
        """Explicit null on a required list means 'nothing named yet', not an error."""
        return [] if v is None else v

    @field_validator("summary", mode="before")
    @classmethod
    def _null_summary_is_blank(cls, v):
        """Callers only test summary for truthiness, so null and "" mean the same."""
        return "" if v is None else v

    @field_validator("destinations", mode="after")
    @classmethod
    def _dedupe_destinations(cls, v: list[str]) -> list[str]:
        """'Amman Jordan' can come back as ['Amman, Jordan', 'Jordan'] — keep one."""
        return dedupe_destinations(v)


# ─── Explore / specific search ───────────────────────────────────────

class Feedback(BaseModel):
    """Backend search trigger attached to a specific-search / explore response.

    One model for both SpecificSearchQuery and ExploreResponse — the file
    previously defined `Feedback` twice, so the two responses silently used
    different classes depending on definition order.
    """

    action: str = Field("fetch-search-results", description="Action to trigger backend search.")
    view: str = Field(..., description="Maps to UI screen: dine, stay, or play.")
    filters: list[str] = Field(default_factory=list, description="Explicit preferences or keywords mentioned by user.")

    @field_validator("filters", mode="before")
    @classmethod
    def _null_filters_is_empty(cls, v):
        return [] if v is None else v


class SpecificSearchQuery(BaseModel):
    category: str = Field("specific-search-query", description="Always 'specific-search-query'")
    intent: str = Field(..., description="Intent of the query: dine, stay, or play")
    destination: str = Field(..., description="Explicitly mentioned destination")
    feedback: Feedback = Field(..., description="Feedback object with action, view, filters")


class ExploreResponse(BaseModel):
    category: str
    intent: str
    destination: str
    feedback: Feedback