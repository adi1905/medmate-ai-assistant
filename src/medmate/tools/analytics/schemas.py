from pydantic import BaseModel, Field
from typing import Optional, Literal


class NoShowRateInput(BaseModel):
    start: str = Field(..., description="Start date, ISO format YYYY-MM-DD")
    end: str =  Field(..., description="End date, ISO format YYYY-MM-DD")

class AppointmentVolumeTrendInput(BaseModel):
    start: str = Field(..., description="Start date, ISO format YYYY-MM-DD")
    end: str = Field(..., description="End date, ISO format YYYY-MM-DD")
    bucket: str = Field("day", description="Bucketing granularity: day, week, or month")

class TopMedicationsInput(BaseModel):
    limit: int = Field(10, description="Number of top medications to return")

class FlaggedLabSummaryInput(BaseModel):
    start: str = Field(..., description="Start date, ISO format YYYY-MM-DD")
    end: str = Field(..., description="End date, ISO format YYYY-MM-DD")

class ProviderUtilizationInput(BaseModel):
    provider_id: str = Field(..., description="Exact provider_id, e.g. PR-001")
    start: str = Field(..., description="Start date, ISO format YYYY-MM-DD")
    end: str = Field(..., description="End date, ISO format YYYY-MM-DD")

class CustomAnalyticsQueryInput(BaseModel):
    question: str = Field(..., description="The staff member's natural-language analytics question")

class GenerateChartInput(BaseModel):
    data: list[dict] = Field(..., description="Rows of data, e.g. from a prior tool's result")
    x_key: str = Field(..., description="Dict key to use for the x-axis / labels")
    y_key: str = Field(..., description="Dict key to use for the y-axis / values")
    chart_type: str = Field("bar", description="One of: bar, line, pie")
    title: str = Field("", description="Chart title")
