# app/schemas.py
from pydantic import BaseModel , Field
from typing import Optional

class QueryRequest(BaseModel):
    """Model for incoming chat requests."""
    message: str
    user_id: str
    reference: str
    param: str
    threadId: Optional[str]
    
    
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
        solution: str
        
        
class global_travel_guardrail(BaseModel):
        isValid: bool
        reason: str
        isTravelRelated: bool
        isPlanRelated: bool
        travel_type: str
        
        
        
class Pax(BaseModel):
    """Represents the count of different types of travelers."""
    adults: int = Field(..., description="Number of adults. Default to 2 for romantic trips or if unspecified.")
    children: int = Field(0, description="Number of children.")
    infants: int = Field(0, description="Number of infants (babies).")
    elderly: int = Field(0, description="Number of elderly travelers.")

class TripPlan(BaseModel):
    """The structured DTO for an extracted trip plan."""
    startDate: Optional[str] = Field(None, description="The start date of the trip in MM-dd-yyyy format. Infer from context.")
    endDate: Optional[str] = Field(None, description="The end date of the trip in MM-dd-yyyy format. Infer from context.")
    numDays: Optional[int] = Field(None, description="The total duration of the trip in days. Use this if specific dates are not present.")
    destinations: list[str] = Field(..., description="A list of cities, countries, or regions (e.g., 'Amalfi Coast', 'NorCal').")
    pax: Optional[Pax] = Field(None, description="Traveler counts. Null if not mentioned.")    
    experienceTypes: Optional[list[str]] = Field(None, description="list of curated experience keywords (e.g., 'romantic', 'adventure', 'cultural', 'family friendly', 'relaxation').")
    travelStyle: Optional[list[str]] = Field(None, description="list of travel styles (e.g., 'luxury', 'budget', 'backpacking', 'all inclusive', 'solo trip').")
    activities: Optional[list[str]] = Field(None, description="list of requested activities, normalized to base form (e.g., 'hiking', 'wine tasting', 'snorkeling').")
    themes: Optional[list[str]] = Field(None, description="list of themes from media or pop culture (e.g., 'James Bond', 'Midnight in Paris').") 
    pois: list[str] = Field(..., description="Explicitly mentioned Points of Interest.")  
    feedback: Optional[list[str]] = Field(None,description="Array of missing field names that require user input.")
    
    
    
    
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
