# app/schemas.py
from pydantic import BaseModel , Field
from typing import Optional, Literal


class Interaction(BaseModel):
    question: str
    answer: str
    
    
class QueryRequest(BaseModel):
    """Model for incoming chat requests."""
    message: str
    user_id: str
    reference: str
    param: str
    threadId: Optional[str]
    old_interactions: Optional[list[Interaction]] = None
    is_pro: Optional[bool] = False
    plan: Optional[bool] = False   # if True: save & retrieve conversation memory via Zep
    location: Optional[str] = None  # user location for destination-guide personalization
    filters: Optional[str] = None   # audience/style filter for destination-guide (e.g. "family")
    
    
class Output_Format(BaseModel):
    """Defines the desired output structure for the main agent."""
    answer: str
    
class HistoryRequest(BaseModel):
    session_id: str
    last_n: int = 20
    
    
class UserCreateRequest(BaseModel):
    user_id: str
    email: str
    first_name: str
    last_name: str
    
class global_input_guardrail(BaseModel):
        isValid: bool
        reason: str
        isTravelRelated: bool
        isMemoryQuery: bool
        isRealtime: bool
        solution: str
        
        
class global_travel_guardrail(BaseModel):
        isValid: bool
        reason: str
        isTravelRelated: bool
        isPlanRelated: bool
        travel_type: str
        
        
        
class TripDriver(BaseModel):
    """Structured intent driver — captures why a theme matters and how central it is."""
    theme: str = Field(..., description="Travel theme (e.g. 'fishing', 'beaches', 'food', 'hiking')")
    priority: Literal["incidental", "preferred", "important", "primary", "exclusive"] = Field(
        ..., description="How central this theme is to the trip"
    )
    score: float = Field(..., description="Priority score 0.0–1.0: incidental≈0.15, preferred≈0.3, important≈0.55, primary≈0.8, exclusive≈0.95")
    confidence: float = Field(..., description="Confidence in this classification 0.0–1.0")
    user_evidence: str = Field(..., description="Exact phrase from a user message that justifies this priority — never from assistant text")
    destination_driver: bool = Field(..., description="True if this theme is what caused the user to choose (or search for) the destination")
    desired_frequency: Optional[str] = Field(None, description="How often: 'once', 'multiple_days', 'daily', 'throughout'")


class Pax(BaseModel):
    """Represents the count of different types of travelers."""
    adults: int = Field(..., description="Number of adults. Infer from context: 'solo'/'alone'/'just me'=1, 'couple'/'romantic'/'two of us'=2, explicit number otherwise.")
    children: int = Field(0, description="Number of children.")
    infants: int = Field(0, description="Number of infants (babies).")
    elderly: int = Field(0, description="Number of elderly travelers.")

class TripPlan(BaseModel):
    """The structured DTO for an extracted trip plan."""
    startDate: Optional[str] = Field(None, description="The start date of the trip in MM-dd-yyyy format. Infer from context.")
    endDate: Optional[str] = Field(None, description="The end date of the trip in MM-dd-yyyy format. Infer from context.")
    numDays: Optional[int] = Field(None, description="Trip duration in days. Auto-calculated from startDate + endDate. Never included in feedback.")
    destinations: list[str] = Field(..., description="A list of cities, countries, or regions (e.g., 'Amalfi Coast', 'NorCal').")
    pax: Optional[Pax] = Field(None, description="Traveler counts. Null if not mentioned.")    
    experienceTypes: Optional[list[str]] = Field(None, description="list of curated experience keywords (e.g., 'romantic', 'adventure', 'cultural', 'family friendly', 'relaxation').")
    travelStyle: Optional[list[str]] = Field(None, description="list of travel styles (e.g., 'luxury', 'budget', 'backpacking', 'all inclusive', 'solo trip').")
    activities: Optional[list[str]] = Field(None, description="list of requested activities, normalized to base form (e.g., 'hiking', 'wine tasting', 'snorkeling').")
    weighted_activities: Optional[list[str]] = Field(None, description="The user's priority activities/interests, WEIGHT-ORDERED: index 0 = highest priority, last index = lowest. Extracted from everything the user emphasized or asked about across the WHOLE conversation (e.g. 'vegan restaurants', 'surfing', 'nightlife'). Preserve this order as-is — most important first, descending. Empty/null if none expressed.")
    themes: Optional[list[str]] = Field(None, description="list of themes from media or pop culture (e.g., 'James Bond', 'Midnight in Paris').")
    pois: list[str] = Field(..., description="Explicitly mentioned Points of Interest.")  
    feedback: Optional[list[str]] = Field(None,description="Array of missing field names that require user input.")
    month: Optional[str] = Field(None, description="Month extracted from conversation")
    summary: str = Field(None, description="A friendly, conversational acknowledgement of the current input followed by a question asking for the items in the 'feedback' list.")
    trip_drivers: Optional[list[TripDriver]] = Field(None, description="Structured intent drivers extracted from user messages only — see TRIP DRIVERS section")
    
    
    
    
class Feedback(BaseModel):
    action: str = Field("fetch-search-results", description="Action to trigger backend search.")
    view: str = Field(..., description="Maps to UI screen: dine, stay, or play.")
    filters: list[str] = Field([], description="Explicit preferences or keywords mentioned by user.")

class SpecificSearchQuery(BaseModel):
    category: str = Field("specific-search-query", description="Always 'specific-search-query'")
    intent: str = Field(..., description="Intent of the query: dine, stay, or play")
    destination: str = Field(..., description="Explicitly mentioned destination")
    feedback: Feedback = Field(..., description="Feedback object with action, view, filters")
    
    
class Feedback(BaseModel):
    action: str
    view: str
    filters: list[str]

class ExploreResponse(BaseModel):
    category: str
    intent: str
    destination: str
    feedback: Feedback
