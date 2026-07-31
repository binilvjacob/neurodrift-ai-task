import json
import logging

_STANDARD_LOG_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Renders each log record as one JSON line, merging in whatever fields were passed via `extra=`."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {key: value for key, value in record.__dict__.items() if key not in _STANDARD_LOG_RECORD_ATTRS}
        )
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Attaches a JSON handler to the "rag" logger namespace, isolated from uvicorn's own loggers."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("rag")
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False
