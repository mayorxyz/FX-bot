"""Environment-driven settings for FXB (loaded from ``.env`` via python-dotenv).

All runtime configuration lives here; nothing else reads environment variables
directly. Secrets (MT5 password) are never logged — :meth:`Settings.redacted`
exists precisely so logs/reports can print config safely.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from fxb.core.paths import DATA_DIR, PROJECT_ROOT

#: Validated log levels accepted in ``LOG_LEVEL``.
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigError(ValueError):
    """Raised for missing/invalid environment configuration (fail fast)."""


def _split_list(raw: str) -> List[str]:
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


@dataclass(frozen=True)
class Settings:
    """Immutable application settings built from the environment."""

    db_path: Path
    symbols: List[str]
    timeframes: List[str]
    account_currency: str
    risk_pct: float
    max_portfolio_heat: float
    mt5_login: Optional[int]
    mt5_password: Optional[str]
    mt5_server: Optional[str]
    mt5_server_utc_offset_hours: float
    d1_anchor: str
    log_level: str

    def redacted(self) -> dict:
        """Plain dict of all settings safe to log (password replaced by ***)."""
        return {
            "db_path": str(self.db_path),
            "symbols": list(self.symbols),
            "timeframes": list(self.timeframes),
            "account_currency": self.account_currency,
            "risk_pct": self.risk_pct,
            "max_portfolio_heat": self.max_portfolio_heat,
            "mt5_login": self.mt5_login,
            "mt5_password": "***" if self.mt5_password else None,
            "mt5_server": self.mt5_server,
            "mt5_server_utc_offset_hours": self.mt5_server_utc_offset_hours,
            "d1_anchor": self.d1_anchor,
            "log_level": self.log_level,
        }


def load_settings(env_file: Optional[os.PathLike | str] = None) -> Settings:
    """Load ``.env`` (if present) and return validated :class:`Settings`.

    Raises :class:`ConfigError` with a clear message on any invalid value.
    """
    path = Path(env_file) if env_file else PROJECT_ROOT / ".env"
    if path.exists():
        load_dotenv(path, override=False)

    def get(key: str, default: str = "") -> str:
        val = os.environ.get(key, default)
        return val.strip() if val is not None else default

    db_raw = get("DB_PATH") or str(DATA_DIR / "fxb.db")
    db_path = Path(db_raw)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    symbols = _split_list(get("SYMBOLS", "EURUSD"))
    if not symbols:
        raise ConfigError("SYMBOLS must contain at least one symbol")
    timeframes = _split_list(get("TIMEFRAMES", "M15,H1,H4,D1"))
    if not timeframes:
        raise ConfigError("TIMEFRAMES must contain at least one timeframe")

    account_currency = get("ACCOUNT_CURRENCY", "USD").upper()
    if not (2 <= len(account_currency) <= 5):
        raise ConfigError(f"ACCOUNT_CURRENCY looks invalid: {account_currency!r}")

    try:
        risk_pct = float(get("RISK_PCT", "0.01"))
    except ValueError:
        raise ConfigError("RISK_PCT must be a number") from None
    if not 0.0 < risk_pct <= 0.05:
        raise ConfigError(f"RISK_PCT must be in (0, 0.05], got {risk_pct}")

    try:
        heat = float(get("MAX_PORTFOLIO_HEAT", "0.06"))
    except ValueError:
        raise ConfigError("MAX_PORTFOLIO_HEAT must be a number") from None
    if not 0.0 < heat <= 1.0:
        raise ConfigError(f"MAX_PORTFOLIO_HEAT must be in (0, 1], got {heat}")
    if heat < risk_pct:
        raise ConfigError(
            f"MAX_PORTFOLIO_HEAT ({heat}) must be >= RISK_PCT ({risk_pct})"
        )

    login_raw = get("MT5_LOGIN")
    mt5_login: Optional[int] = None
    if login_raw:
        try:
            mt5_login = int(login_raw)
        except ValueError:
            raise ConfigError(f"MT5_LOGIN must be an integer, got {login_raw!r}") from None
    mt5_password = get("MT5_PASSWORD") or None
    mt5_server = get("MT5_SERVER") or None

    try:
        offset = float(get("MT5_SERVER_UTC_OFFSET_HOURS", "0"))
    except ValueError:
        raise ConfigError("MT5_SERVER_UTC_OFFSET_HOURS must be a number") from None
    if not -14.0 <= offset <= 14.0:
        raise ConfigError(
            f"MT5_SERVER_UTC_OFFSET_HOURS must be within [-14, 14], got {offset}"
        )

    d1_anchor = get("D1_ANCHOR", "17:00 America/New_York")

    log_level = get("LOG_LEVEL", "INFO").upper()
    if log_level not in _LOG_LEVELS:
        raise ConfigError(
            f"LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}, got {log_level!r}"
        )

    return Settings(
        db_path=db_path,
        symbols=symbols,
        timeframes=timeframes,
        account_currency=account_currency,
        risk_pct=risk_pct,
        max_portfolio_heat=heat,
        mt5_login=mt5_login,
        mt5_password=mt5_password,
        mt5_server=mt5_server,
        mt5_server_utc_offset_hours=offset,
        d1_anchor=d1_anchor,
        log_level=log_level,
    )
