import os
import logging

logger = logging.getLogger(__name__)


def get_env_var(name: str, default=None) -> str:
    """Return the environment variable value, logging the result.

    If the variable is not set, returns the provided default value.
    """
    value = os.getenv(name, default)
    logger.info(f"ENV {name} = {value}")
    return value
