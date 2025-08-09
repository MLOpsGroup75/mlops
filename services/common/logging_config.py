import structlog
import logging
import sys
import sqlite3
import asyncio
import aiosqlite
import json
from datetime import datetime
from typing import Dict, Any, Optional
from config.settings import settings


class SQLiteHandler(logging.Handler):
    """Custom logging handler that writes logs to SQLite database"""

    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize the SQLite database with logs table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                logger_name TEXT,
                message TEXT,
                service TEXT,
                request_id TEXT,
                extra_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()
        conn.close()

    def emit(self, record):
        """Emit a log record to SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Extract structured data
            extra_data = {}
            if hasattr(record, "request_id"):
                extra_data["request_id"] = record.request_id
            if hasattr(record, "service"):
                extra_data["service"] = record.service
            if hasattr(record, "endpoint"):
                extra_data["endpoint"] = record.endpoint
            if hasattr(record, "method"):
                extra_data["method"] = record.method
            if hasattr(record, "status_code"):
                extra_data["status_code"] = record.status_code
            if hasattr(record, "duration"):
                extra_data["duration"] = record.duration

            cursor.execute(
                """
                INSERT INTO logs (timestamp, level, logger_name, message, service, request_id, extra_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.fromtimestamp(record.created).isoformat(),
                    record.levelname,
                    record.name,
                    record.getMessage(),
                    getattr(record, "service", "unknown"),
                    getattr(record, "request_id", None),
                    json.dumps(extra_data) if extra_data else None,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            # Don't let logging errors crash the application
            print(f"Error writing to log database: {e}", file=sys.stderr)


def setup_logging(service_name: str = "mlops-service"):
    """Setup structured logging with console and SQLite output"""

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Setup standard logging
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, settings.log_level.upper()),
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Add SQLite handler if log_to_file is enabled
    if settings.log_to_file:
        sqlite_handler = SQLiteHandler(settings.log_db_path)
        sqlite_handler.setLevel(getattr(logging, settings.log_level.upper()))
        logging.getLogger().addHandler(sqlite_handler)

    # Get the logger
    logger = structlog.get_logger(service_name)

    return logger


def get_logger(name: str = None):
    """Get a structured logger instance"""
    return structlog.get_logger(name or settings.service_name)
