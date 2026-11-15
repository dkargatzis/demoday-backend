"""Who one personalized demo is for (request) and what the pipeline learned about them (job state).

Kept apart from schemas.py so that module can import these models without a cycle. Sender-typed fields are only
ever quoted as the sender's words; facts about the company come from their public website at run time, tagged
[site], [sender] or [assumed] so the planner, the cards and the dashboard know where every claim came from.
"""
import re
from typing import List, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, PrivateAttr, ValidationInfo, field_validator, model_validator

AudienceGoal = Literal["book_call", "start_trial", "reply"]
TEXT_LIMITS = {"company": 80, "persona": 300, "first_name": 40, "notes": 600}


def _plain(value: Optional[str], limit: int) -> Optional[str]:
    """Founder-typed text reaches prompts, title cards and TTS: strip control chars, collapse whitespace, hard cap."""
    text = re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]+", " ", value or "")).strip()
    return text[:limit] or None


def _http_url(value: str) -> str:
    """Same check and error sentence as schemas._http_url (copied: schemas imports this module, not the reverse)."""
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"{value!r} must be an http(s) URL")
    return value


_HOST = re.compile(r"^[\w-]+(\.[\w-]+)+$")  # letters, digits and hyphens around at least one dot, no spaces
_FIELD_NAMES = {"website": "Website", "cta_url": "Link"}


def _optional_http_url(value: Optional[str], field: str = "website") -> Optional[str]:
    """Blank -> None; a bare host typed in a hurry ("acme.com") becomes https://acme.com; otherwise http(s) only, and
    the host must be a web address (a person's name typed into the field is refused with a sentence that says so)."""
    text = (value or "").strip()
    if not text:
        return None
    candidate = _http_url(text if "://" in text else "https://" + text)  # the scheme first: ftp:// gets that sentence
    host = candidate.split("://", 1)[1].split("/", 1)[0].split("?", 1)[0].split("#", 1)[0].rsplit("@", 1)[-1]
    host = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    if not _HOST.match(host):
        raise ValueError(f"{_FIELD_NAMES.get(field, field)} {text!r} is not a web address; use one like hubstaff.com")
    return candidate


def _company_from_url(text: str) -> Optional[str]:
    """'https://www.hubstaff.com/' -> 'Hubstaff' (a URL typed into the company field); None for anything else.
    A bare "Booking.com" stays as typed: only a scheme or "www." marks it as a web address."""
    if not re.match(r"^(https?://|www\.)", text, re.IGNORECASE) or " " in text.strip():
        return None
    host = (urlparse(text if "://" in text else "https://" + text).hostname or "").lower()
    labels = [label for label in host.split(".") if label and label != "www"]
    if len(labels) < 2:
        return None
    second_level = len(labels) >= 3 and labels[-2] in ("co", "com", "org", "net", "ac", "gov") and len(labels[-1]) == 2
    name = labels[-3] if second_level else labels[-2]
    return name[:1].upper() + name[1:]


class Audience(BaseModel):
    """Who one personalized demo is for. Sender-typed fields are quoted as the sender's words; facts about the
    company come only from `website` at run time."""
    company: str = Field(..., min_length=1, max_length=80, description="Prospect company name as it appears on the cards")
    persona: str = Field(..., min_length=3, max_length=300,
                         description='Who will watch: "Head of Platform at a 40-person fintech, drowning in PR reviews"')
    website: Optional[str] = Field(default=None, description="Their public site, read for at most 15 s to ground the "
                                                             "brief; private addresses are refused at run time")
    first_name: Optional[str] = Field(default=None, max_length=40, description="Spoken at most twice")
    goal: AudienceGoal = Field(default="book_call", description="What the outro card asks for")
    cta_url: Optional[str] = Field(default=None, description="Link typeset on the outro card and used in the outreach; "
                                                             "defaults to app_url")
    notes: Optional[str] = Field(default=None, max_length=600,
                                 description="What the sender knows; quoted as the sender's notes, never as a site fact")

    @field_validator("company", "persona", "first_name", "notes")
    @classmethod
    def _plain_text(cls, value: Optional[str], info: ValidationInfo) -> Optional[str]:
        cleaned = _plain(value, TEXT_LIMITS[str(info.field_name)])
        if cleaned is None and info.field_name in ("company", "persona"):
            raise ValueError(f"audience.{info.field_name} must not be empty")
        return cleaned

    @field_validator("website", "cta_url")
    @classmethod
    def _http_or_none(cls, value: Optional[str], info: ValidationInfo) -> Optional[str]:
        return _optional_http_url(value, str(info.field_name))

    @model_validator(mode="after")
    def _company_not_a_url(self) -> "Audience":
        """A URL typed as the company ("https://hubstaff.com/") becomes the name on the cards ("Hubstaff") and, when
        no website was given, the site the research reads."""
        name = _company_from_url(self.company)
        if name:
            if not self.website:
                self.website = _optional_http_url(self.company)
            self.company = name
        return self


class AudienceBrief(BaseModel):
    """What the planner, the cards and the outreach know about the prospect. Tags: [site] [sender] [assumed]."""
    company: str
    persona: str
    website: Optional[str] = None
    first_name: Optional[str] = None
    goal: AudienceGoal = "book_call"
    cta_url: Optional[str] = None
    notes: Optional[str] = None
    what_they_do: str = Field(default="", description="[site] one sentence in the site's words; '' when the site was not read")
    facts: List[str] = Field(default_factory=list, description="[site] up to 4, each verified near-verbatim against the site")
    pains: List[str] = Field(default_factory=list, description="Up to 3, each prefixed '[site] ', '[sender] ' or '[assumed] '")
    talking_points: List[str] = Field(default_factory=list, description="Up to 3 outcomes to show this persona")
    assumptions: List[str] = Field(default_factory=list, description="Shown amber, spoken only as possibilities")
    tone: str = ""
    site_title: str = ""
    pages_read: List[str] = Field(default_factory=list, description="Paths actually fetched")
    site_error: Optional[str] = Field(default=None, description='Why the site was not read: "HTTP 403", "timed out after 15 s", ...')
    research_sec: float = 0.0
    source: str = Field(default="pending", description='"pending" | "acme.com + Gemini (model)" | "acme.com + template (reason)" '
                                                       '| "typed only (why) + ..."')
    coverage: Optional[int] = Field(default=None, description="Scenes naming the company, the role or a tagged fact (set after planning)")
    unsourced_numbers: List[str] = Field(default_factory=list,
                                         description="Numbers in the narration that are not on the site or in the notes (warned, never blocking)")
    _corpus: str = PrivateAttr(default="")  # the fetched site text, kept in memory for the post-plan number check


class OutreachEmail(BaseModel):
    subject: str
    body: str = Field(..., description="Plain text containing the literal {{DEMO_LINK}} exactly once")
