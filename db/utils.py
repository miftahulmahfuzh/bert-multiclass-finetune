import datetime

def get_current_timestamp(timezone_offset: int = 7) -> str:
    """
    Get current timestamp in ISO format with specified timezone offset.

    Args:
        timezone_offset: Hours offset from UTC (default: 7 for Jakarta)

    Returns:
        ISO formatted datetime string
    """
    tz = datetime.timezone(datetime.timedelta(hours=timezone_offset))
    return datetime.datetime.now(tz).isoformat()
