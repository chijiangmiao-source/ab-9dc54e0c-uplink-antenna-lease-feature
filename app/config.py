"""Runtime configuration.

All values are read from environment variables so the same image can run as
the API service or the one-shot ``verify`` acceptance service.
"""

import os


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    user = os.environ.get("POSTGRES_USER", "satctl")
    password = os.environ.get("POSTGRES_PASSWORD", "satctl")
    db = os.environ.get("POSTGRES_DB", "satctl")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


# SQLAlchemy connection URL (psycopg2 driver).
DATABASE_URL: str = _database_url()

# Inclusive lease duration bounds in seconds.
MIN_LEASE_SECONDS: int = 5
MAX_LEASE_SECONDS: int = 120

# Inclusive bounds for a single renewal extension, in seconds. A renewal adds
# on top of the lease's current expiry; each individual addition stays within
# the same window as an initial acquisition.
MIN_RENEW_SECONDS: int = 5
MAX_RENEW_SECONDS: int = 120
