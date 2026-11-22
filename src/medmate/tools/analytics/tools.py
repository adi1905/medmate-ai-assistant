from langchain.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated, Optional
from ...auth import rbac
from ...db.repositories import analytics_repo
from ...guards import sql_guard
from ...llm.text_to_sql import generate_sql
from .schemas import (NoShowRateInput, AppointmentVolumeTrendInput, 
                      FlaggedLabSummaryInput, TopMedicationsInput, 
                      ProviderUtilizationInput, 
                      CustomAnalyticsQueryInput, GenerateChartInput)
from ...graph.state import MedMateState
from .charts import _generate_chart
from ..common.tools import _current_identity, ToolResult

GENERIC_QUERY_FAILURE  = (
    "I wasn't able to answer that with the data available to me. "
    "Try rephrasing, or ask about appointments, prescriptions, lab results, or provider activity."
)



@tool(args_schema=NoShowRateInput)
def no_show_rate_by_period(start: str, end: str, state: Annotated[MedMateState, InjectedState]) ->dict:
    """Get the no-show rate and appointment status breakdown for a date range. STAFF ONLY."""
    role, _ = _current_identity(state)
    try:
        rbac.require_staff(role)
        data = analytics_repo.get_no_show_rate(start, end)
        return ToolResult(success=True, data=data).model_dump()
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error = str(e)).model_dump()
    
@tool(args_schema=AppointmentVolumeTrendInput)
def appointment_volume_trend(start: str, end: str, bucket: str, state: Annotated[MedMateState, InjectedState]) -> dict:
    """Get appointment volume over time, bucketed by day/week/month. STAFF ONLY."""
    role, _ = _current_identity(state)
    try:
        rbac.require_staff(role)
        data = analytics_repo.get_appointment_volume_trend(start, end, bucket)
        return ToolResult(success=True, data={"trend": data}).model_dump()
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error=str(e)).model_dump()
    except ValueError as e:
        return ToolResult(success=False, error=str(e)).model_dump()
    
@tool(args_schema=TopMedicationsInput)
def top_medications(limit: int, 
                    state: Annotated[MedMateState, InjectedState],) -> dict:
    """Get the most prescribed active medications, ranked by patient count. STAFF ONLY."""
    role, _ = _current_identity(state)
    try:
        rbac.require_staff(role)
        data = analytics_repo.get_top_medications(limit)
        return ToolResult(success=True, data={"results": data}).model_dump()
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error=str(e)).model_dump()
    
@tool(args_schema=FlaggedLabSummaryInput)
def flagged_lab_summary(start: str, end: str, state: Annotated[MedMateState, InjectedState]) -> dict:
    """Get a count and breakdown of flagged lab results in a date range. STAFF ONLY."""
    role, _ = _current_identity(state)
    try:
        rbac.require_staff(role)
        data = analytics_repo.get_flagged_lab_summary(start, end)
        return ToolResult(success=True, data=data).model_dump()
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error=str(e)).model_dump()

    
@tool(args_schema=ProviderUtilizationInput)
def provider_utilization(provider_id: str, start: str, end: str, state: Annotated[MedMateState, InjectedState]) -> dict:
    """Get appointment status breakdown for a single provider in a date range. STAFF ONLY."""
    role, _ = _current_identity(state)
    try:
        rbac.require_staff(role)
        data = analytics_repo.get_provider_utilization(provider_id, start, end)
        return ToolResult(success=True, data=data).model_dump()
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error=str(e)).model_dump()
    
@tool(args_schema=CustomAnalyticsQueryInput)
def answer_custom_query(question: str, state: Annotated[MedMateState, InjectedState]) -> dict:
    """Answer a staff analytics question that doesn't match a specific metric tool
    (no_show_rate_by_period, appointment_volume_trend, top_medications, flagged_lab_summary,
    provider_utilization). Use only when none of those fit. Generates and safely executes
    a read-only SQL query. STAFF ONLY."""
    role, _ = _current_identity(state)
    try:
        rbac.require_staff(role)
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error=str(e)).model_dump()

    schema = analytics_repo.get_schema_snapshot()
    gen = generate_sql(question, schema)
    print(f"[DEBUG] Generated SQL (attempt 1): {gen.sql}")
    is_valid, safe_sql, reason = sql_guard.guard_and_prepare(
        gen.sql, analytics_repo.ALLOWED_TABLES
    )
    print(f"[DEBUG] Valid: {is_valid}, Reason: {reason}")

    if not is_valid:
        # one retry with the failure reason fed back into generation
        gen = generate_sql(question, schema, prior_error=reason)
        is_valid, safe_sql, reason = sql_guard.guard_and_prepare(
            gen.sql, analytics_repo.ALLOWED_TABLES
        )

    if not is_valid:
        # reason is internal-only (logged), never shown to the user
        return ToolResult(success=False, error=GENERIC_QUERY_FAILURE).model_dump()
    
    if not gen.answerable:
        return ToolResult(success=False, error=GENERIC_QUERY_FAILURE).model_dump()

    try:
        columns, rows = analytics_repo.run_readonly_query(safe_sql)
    except Exception:
        return ToolResult(success=False, error=GENERIC_QUERY_FAILURE).model_dump()

    return ToolResult(
        success=True,
        data={"columns": columns, "rows": rows},
    ).model_dump()


@tool(args_schema=GenerateChartInput)
def generate_chart(data: list[dict], x_key: str, y_key: str, 
                   state: Annotated[MedMateState, InjectedState],
                   chart_type: str = "bar", title: str = "",) -> dict:
    """Render tabular data (e.g. from another analytics tool's result) as a chart.
    chart_type: 'bar', 'line', or 'pie'. STAFF ONLY."""
    role, _ = _current_identity(state)
    try:
        rbac.require_staff(role)
        image_b64 = _generate_chart(data, x_key, y_key, chart_type, title)
        return ToolResult(success=True, data={"image_base64": image_b64, "format": "png"}).model_dump()
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error=str(e)).model_dump()
    except (KeyError, ValueError) as e:
        return ToolResult(success=False, error=f"Could not generate chart: {e}").model_dump()


