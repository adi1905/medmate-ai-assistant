from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))


def now_ist() -> datetime:
    """Return current datetime in IST."""
    return datetime.now(IST)


def now_ist_str() -> str:
    """Return current IST time as a formatted string."""
    return now_ist().strftime("%Y-%m-%d %H:%M:%S")