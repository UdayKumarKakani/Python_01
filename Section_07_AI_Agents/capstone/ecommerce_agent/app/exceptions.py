"""Service-level exceptions. Each one maps to an HTTP status in app/main.py."""


class AgentServiceError(Exception):
    status_code: int = 500
    error_code: str = "internal_error"
    public_message: str = "Something went wrong on our side. Please try again shortly."

    def __init__(self, detail: str | None = None) -> None:
        # `detail` is for logs only; clients always receive `public_message`.
        super().__init__(detail or self.public_message)


class LLMConfigurationError(AgentServiceError):
    status_code = 500
    error_code = "llm_misconfigured"
    public_message = "The assistant is not configured correctly. Please contact support."


class LLMTimeoutError(AgentServiceError):
    status_code = 504
    error_code = "llm_timeout"
    public_message = "The assistant took too long to respond. Please try again."


class LLMProviderError(AgentServiceError):
    status_code = 503
    error_code = "llm_unavailable"
    public_message = "The assistant is temporarily unavailable. Please try again in a moment."


class VectorStoreError(AgentServiceError):
    """Raised inside the knowledge base tool; converted to a tool message, never surfaced directly."""

    status_code = 503
    error_code = "knowledge_base_unavailable"
    public_message = "Our knowledge base is temporarily unavailable."


class StoreError(AgentServiceError):
    """Raised by the SQLite store; converted to a tool message, never surfaced directly."""

    status_code = 503
    error_code = "store_unavailable"
    public_message = "Our store database is temporarily unavailable."
