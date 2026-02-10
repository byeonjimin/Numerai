

import os, re, time, json, argparse, warnings, traceback, sys, threading, random, gc
from collections import deque
from importlib import import_module
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
warnings.filterwarnings("ignore")

# ----- soft imports -----------------------------------------------------------

def _try_import(name):
    try:
        return import_module(name)
    except Exception:
        return None

numerapi = _try_import("numerapi")
yf = _try_import("yfinance")
lgb = _try_import("lightgbm")
xgb = _try_import("xgboost")
requests = _try_import("requests")
try:
    from numerai_tools import scoring as nt_scoring, signals as nt_signals
except Exception:
    nt_scoring = None
    nt_signals = None

NewsApiClient = None
_newsapi_mod = _try_import("newsapi")
if _newsapi_mod:
    NewsApiClient = getattr(_newsapi_mod, "NewsApiClient", None)

TrendReq = None
_pytrends = _try_import("pytrends.request")
if _pytrends:
    TrendReq = getattr(_pytrends, "TrendReq", None)

SentimentIntensityAnalyzer = None
_vader = _try_import("vaderSentiment.vaderSentiment")
if _vader:
    SentimentIntensityAnalyzer = getattr(_vader, "SentimentIntensityAnalyzer", None)

zipline = _try_import("zipline") or _try_import("zipline_reloaded")
dotenv = _try_import("dotenv")
rich_logging = _try_import("rich.logging")

_opensignals_provider_mod = _try_import("opensignals.data.provider")
_opensignals_yahoo_mod = _try_import("opensignals.data.yahoo")
_opensignals_features_mod = _try_import("opensignals.features")
OpenSignalsProviderBase = getattr(_opensignals_provider_mod, "Provider", None) if _opensignals_provider_mod else None
OpenSignalsYahooCls = getattr(_opensignals_yahoo_mod, "Yahoo", None) if _opensignals_yahoo_mod else None

plt = None
_mpl = _try_import("matplotlib.pyplot")
if _mpl:
    plt = _mpl

# optional rich progress (legacy call sites)
track = None
try:
    from rich.progress import track as _rich_track
    track = _rich_track
except Exception:
    track = None

# progress helpers
def _iter_with_progress(iterable, description: str, total: Optional[int]=None, log_obj=None):
    """Yield iterable while logging periodic percentage updates."""
    if total is None and hasattr(iterable, "__len__"):
        try:
            total = len(iterable)  # type: ignore
        except Exception:
            total = None
    if log_obj is None or total is None or total <= 0:
        return iterable

    def _generator():
        step = max(1, total // 20)
        for idx, item in enumerate(iterable):
            if idx % step == 0:
                pct = (idx / total) * 100
                log_obj.info(f"{description} progress: {pct:.1f}% ({idx}/{total})")
            yield item
        log_obj.info(f"{description} progress: 100% ({total}/{total})")
    return _generator()

def _start_heartbeat(interval_sec: int = 120):
    """Emit periodic stdout to avoid CI log timeouts."""
    stop = threading.Event()

    def _beat():
        while not stop.wait(interval_sec):
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[heartbeat] {ts} UTC", flush=True)

    t = threading.Thread(target=_beat, name="heartbeat", daemon=True)
    t.start()
    return stop

from scipy import stats
from sklearn.linear_model import Ridge


def _parse_start_date(value: Optional[str], default: str) -> pd.Timestamp:
    """Best-effort parser that falls back to default when input is invalid."""
    candidate = value.strip() if isinstance(value, str) else default
    if not candidate:
        candidate = default
    try:
        ts = pd.Timestamp(candidate)
    except Exception:
        ts = pd.Timestamp(default)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()

# ----- logging ---------------------------------------------------------------
import logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_STDOUT = bool(int(os.getenv("LOG_STDOUT", "0"))) or os.getenv("GITHUB_ACTIONS", "").lower() == "true"
handlers = []
if rich_logging:
    handlers.append(rich_logging.RichHandler(rich_tracebacks=True, show_time=False, keywords=None))
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
handlers.append(file_handler)
fmt = "%(message)s" if rich_logging else "%(asctime)s %(levelname)s: %(message)s"
if LOG_STDOUT:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(fmt))
    handlers.append(stream_handler)
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO").upper(),
                    format=fmt,
                    handlers=handlers)
log = logging.getLogger("signals")
log.info(f"[logging] Streaming to {log_file}")


def _get_opensignals_yahoo():
    """Instantiate and memoize the OpenSignals Yahoo downloader."""
    global _OPENSIGNALS_YAHOO_CLIENT
    if _OPENSIGNALS_YAHOO_CLIENT is None and USE_OPENSIGNALS_YAHOO and OpenSignalsYahooCls:
        try:
            _OPENSIGNALS_YAHOO_CLIENT = OpenSignalsYahooCls()
            log.info("[opensignals] Yahoo downloader initialized.")
        except Exception as err:
            log.warning(f"[opensignals] Yahoo init failed: {err}")
            _OPENSIGNALS_YAHOO_CLIENT = None
    return _OPENSIGNALS_YAHOO_CLIENT


def _get_opensignals_generators() -> List[Any]:
    """Lazily build OpenSignals feature generators for reuse."""
    global _OPENSIGNALS_FEATURE_GENERATORS
    if _OPENSIGNALS_FEATURE_GENERATORS or not ENABLE_OPENSIGNALS_FEATURES:
        return _OPENSIGNALS_FEATURE_GENERATORS
    if not _opensignals_features_mod:
        return []
    try:
        VarChange = getattr(_opensignals_features_mod, "VarChange", None)
        RSI = getattr(_opensignals_features_mod, "RSI", None)
        SMA = getattr(_opensignals_features_mod, "SMA", None)
    except Exception as err:
        log.warning(f"[opensignals] Feature import failed: {err}")
        return []
    gens: List[Any] = []
    try:
        if RSI:
            gens.append(RSI(num_days=5, interval=14, variable='adj_close'))
            gens.append(RSI(num_days=5, interval=21, variable='adj_close'))
        if SMA:
            gens.append(SMA(num_days=5, interval=14, variable='adj_close'))
            gens.append(SMA(num_days=5, interval=21, variable='adj_close'))
        if VarChange:
            gens.append(VarChange(num_days=5, variable='adj_close'))
    except Exception as err:
        log.warning(f"[opensignals] Feature generator build failed: {err}")
        gens = []
    _OPENSIGNALS_FEATURE_GENERATORS = gens
    if gens:
        log.info(f"[opensignals] Feature generators enabled ({len(gens)} blocks).")
    return _OPENSIGNALS_FEATURE_GENERATORS

# ----- env & defaults --------------------------------------------------------
if dotenv:
    dotenv.load_dotenv()

DATA_VERSION = "v2.1"
YEARS_BACK = int(os.getenv("YEARS_BACK", "10"))
HIST_START_DEFAULT = (datetime.utcnow() - timedelta(days=365*YEARS_BACK)).strftime("%Y-%m-%d")
DATA_START_TS = _parse_start_date(os.getenv("DATA_START_DATE"), HIST_START_DEFAULT)
DATA_START_DATE = DATA_START_TS.strftime("%Y-%m-%d")
PIPE_START_DEFAULT = "2019-01-01"
PIPE_START_TS = _parse_start_date(os.getenv("PIPE_START_DATE"), PIPE_START_DEFAULT)
PIPE_START_DATE = PIPE_START_TS.strftime("%Y-%m-%d")
HOLDOUT_START_DEFAULT = os.getenv("HOLDOUT_START_DATE", "2023-01-01")
HOLDOUT_START_TS = _parse_start_date(os.getenv("HOLDOUT_START_DATE"), HOLDOUT_START_DEFAULT)
TRAIN_START_TS = _parse_start_date(os.getenv("TRAIN_START_DATE"), DATA_START_DATE)
TRAIN_START_DATE = TRAIN_START_TS.strftime("%Y-%m-%d")

SOCIAL_MEDIA_TICKER_COUNT = int(os.getenv("SOCIAL_MEDIA_TICKER_COUNT","300"))
GOOGLE_TRENDS_TICKER_COUNT = int(os.getenv("GOOGLE_TRENDS_TICKER_COUNT","300"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS","8"))
MODEL_CV_FOLDS = int(os.getenv("MODEL_CV_FOLDS","5"))
MODEL_MIN_SPEARMAN = float(os.getenv("MODEL_MIN_SPEARMAN","0.02"))
MODEL_FOLD_KEEP_FRAC = float(os.getenv("MODEL_FOLD_KEEP_FRAC","0.6"))
DEFAULT_COMBOS = "25x25,50x50,50x25"
STACK_META_ALPHA = float(os.getenv("STACK_META_ALPHA","0.0"))
STACK_META_MIN_MODELS = int(os.getenv("STACK_META_MIN_MODELS","3"))
STACK_META_MIN_SPEARMAN = float(os.getenv("STACK_META_MIN_SPEARMAN","0.01"))
EMBARGO_WEEKS = int(os.getenv("EMBARGO_WEEKS","4"))
FAST_MODE = bool(int(os.getenv("FAST_MODE", "0")))
if FAST_MODE:
    MODEL_CV_FOLDS = min(MODEL_CV_FOLDS, 2)

NUMERAI_API_MAX_RETRIES = max(int(os.getenv("NUMERAI_API_MAX_RETRIES", "3")), 1)
NUMERAI_API_RETRY_BACKOFF = float(os.getenv("NUMERAI_API_RETRY_BACKOFF", "2.0"))
USE_OPENSIGNALS_UNIVERSE = bool(int(os.getenv("USE_OPENSIGNALS_UNIVERSE", "0")))
USE_OPENSIGNALS_YAHOO = bool(int(os.getenv("USE_OPENSIGNALS_YAHOO", "0")))
ENABLE_OPENSIGNALS_FEATURES = bool(int(os.getenv("ENABLE_OPENSIGNALS_FEATURES", "0")))
OPENSIGNALS_FEATURE_PREFIX = os.getenv("OPENSIGNALS_FEATURE_PREFIX", "os")
_OPENSIGNALS_YAHOO_CLIENT = None
_OPENSIGNALS_FEATURE_GENERATORS: List[Any] = []


# diagnostics config
DIAG_LAST_ERAS = int(os.getenv("DIAG_LAST_ERAS","300"))
DIAG_TARGET_ERAS_COUNT = int(os.getenv("DIAG_TARGET_ERAS_COUNT","100"))
DIAG_FULL_UNIVERSE = bool(int(os.getenv("DIAG_FULL_UNIVERSE","1")))
DIAG_MIN_TICKERS_PER_ERA = int(os.getenv("DIAG_MIN_TICKERS_PER_ERA","2000"))
DIAG_UPLOAD_TRIMMED = bool(int(os.getenv("DIAG_UPLOAD_TRIMMED","1")))
MISSING_RATIO_DROP = float(os.getenv("MISSING_RATIO_DROP","0.3"))

# neutralization
NEUTRALIZE_FEATURES = True
NEUTRALIZE_PREDICTIONS = False  # B: feature-neutral only (skip prediction neutralization)
RISK_NEUTRALIZE = False
SIZE_NEUTRALIZE_STRENGTH = float(os.getenv("SIZE_NEUTRALIZE_STRENGTH","0.7"))  # partial to avoid overkill
VOL_NEUTRALIZE_STRENGTH  = float(os.getenv("VOL_NEUTRALIZE_STRENGTH","0.7"))

# blending & smoothing
EMA_SPAN = int(os.getenv("EMA_SPAN","16"))
PREDICTION_BLEND_WEIGHT = float(os.getenv("PRED_BLEND","0.3"))  # blend fraction for new predictions
ERA_DECAY_HALFLIFE = int(os.getenv("ERA_DECAY_HALFLIFE","26"))
RECENT_VALIDATION_ERAS = int(os.getenv("RECENT_VALIDATION_ERAS","60"))
# payout scaling: Numerai Signals payout = corr / 0.04 clipped to [-1, 2]
PAYOUT_CORR_DIVISOR = float(os.getenv("PAYOUT_CORR_DIVISOR","0.04"))
PAYOUT_CLIP_LOW = float(os.getenv("PAYOUT_CLIP_LOW","-1.0"))
PAYOUT_CLIP_HIGH = float(os.getenv("PAYOUT_CLIP_HIGH","2.0"))
PAYOUT_BACKTEST_START_CASH = float(os.getenv("PAYOUT_BACKTEST_START_CASH","1"))
PAYOUT_STAKE_FRACTION = float(os.getenv("PAYOUT_STAKE_FRACTION","0.1"))
PAYOUT_SETTLEMENT_LAG = int(os.getenv("PAYOUT_SETTLEMENT_LAG","4"))
PAYOUT_METRICS_LOG = Path(os.getenv("PAYOUT_METRICS_LOG", LOG_DIR / "run_metrics_log.csv"))
RISK_TILT_MIN_SECTOR_ROWS = int(os.getenv("RISK_TILT_MIN_SECTOR_ROWS","200"))

# winsorization
WINSORIZE_P1, WINSORIZE_P99 = 0.01, 0.99

# social weights (base)
WEIGHTS_BASE = {
    "reddit_volume":  0.10,
    "wiki_views":     0.15,
    "gdelt_mentions": 0.15,
    "twitter_counts": 0.10,
    "google_trend":   0.20,
}

# social toggles
ENABLE_PUSHSHIFT = True
ENABLE_GDELT = True
ENABLE_TWITTER = False
ENABLE_WIKIPEDIA = True
ENABLE_GOOGLE_TRENDS = True
ENABLE_SOCIAL_FEATURES = bool(int(os.getenv("ENABLE_SOCIAL_FEATURES","0")))
SOCIAL_FEATURE_WEIGHT = float(os.getenv("SOCIAL_FEATURE_WEIGHT","0.3"))
SOCIAL_FEATURE_WHITELIST = {
    part.strip() for part in os.getenv(
        "SOCIAL_FEATURE_WHITELIST",
        "social_index_rank,google_trend_rank,analyst_buzz_rank,analyst_sentiment_rank,"
        "mom20_x_social,mom60_x_trend,oversold_attention,bullish_rating_momentum"
    ).split(",") if part.strip()
}
ENABLE_XGB_MODELS = bool(int(os.getenv("ENABLE_XGB_MODELS","1")))
SOCIAL_RELIABILITY_MIN = float(os.getenv("SOCIAL_RELIABILITY_MIN","0.5"))
SOCIAL_NOISE_THRESHOLD = float(os.getenv("SOCIAL_NOISE_THRESHOLD","2.2"))
SOCIAL_ONLY_UNIVERSE = bool(int(os.getenv("SOCIAL_ONLY_UNIVERSE","0")))

# crypto hook (disabled)
ENABLE_CRYPTO_META = bool(int(os.getenv("ENABLE_CRYPTO_META","0")))

CACHE_DIR = Path("cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
PRICE_CACHE = CACHE_DIR / "prices.parquet"
SOCIAL_CACHE = CACHE_DIR / "hist_social.parquet"
META_CACHE = CACHE_DIR / "meta.parquet"
PREV_PRED_CACHE = CACHE_DIR / "prev_predictions.parquet"
SUBMISSION_STATE_FILE = CACHE_DIR / "submission_state.json"

np.random.seed(42)

# ----- credentials -----------------------------------------------------------
DEFAULTS = {
    "NUMERAI_PUBLIC_ID": os.getenv("NUMERAI_PUBLIC_ID",""),
    "NUMERAI_SECRET_KEY": os.getenv("NUMERAI_SECRET_KEY",""),
    "NUMERAI_MODEL_NAME": os.getenv("NUMERAI_MODEL_NAME","signals-model"),
    "REDDIT_CLIENT_ID": os.getenv("REDDIT_CLIENT_ID",""),
    "REDDIT_CLIENT_SECRET": os.getenv("REDDIT_CLIENT_SECRET",""),
    "REDDIT_USER_AGENT": os.getenv("REDDIT_USER_AGENT","signals-bot"),
    "NEWSAPI_KEY": os.getenv("NEWSAPI_KEY",""),
    "TWITTER_BEARER_TOKEN": os.getenv("TWITTER_BEARER_TOKEN",""),
}

# ----- utils ----------------------------------------------------------------

def weekly_friday(dt_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt_series).dt.tz_localize(None)
    return dt.dt.to_period("W-FRI").dt.end_time.dt.tz_localize(None).dt.normalize()


def pct_rank_stable(values: pd.Series) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.array([], dtype=float)
    base = pd.Series(values).astype(float).reset_index(drop=True)
    jitter = base.index.to_series().apply(lambda i: ((i*2654435761) % 997)/1e8 - 5e-9)
    v = base + jitter
    rank = v.rank(method="average").astype(float)
    pct = (rank - 0.5) / n
    return np.clip(pct.values, 1e-6, 1-1e-6)


def _cs_rank(series: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank preserving original index."""
    if series.empty:
        return pd.Series(index=series.index, dtype=float)
    arr = series.astype(float)
    mask = arr.isna()
    filled = arr.fillna(arr.mean())
    ranked = pd.Series(pct_rank_stable(filled), index=series.index)
    if mask.any():
        ranked[mask] = np.nan
    return ranked


def _cs_zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score (mean 0, std 1) with safe fallback."""
    if series.empty:
        return pd.Series(index=series.index, dtype=float)
    arr = series.astype(float)
    std = arr.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=series.index)
    mean = arr.mean()
    return (arr - mean) / (std + 1e-9)


def winsorize_zscore_per_ticker(df: pd.DataFrame, value_col: str,
                                p1: float=WINSORIZE_P1, p99: float=WINSORIZE_P99) -> pd.Series:
    eps = 1e-9
    def _wz(g):
        v = g[value_col].astype(float)
        if len(v.dropna()) < 5:
            return (v - v.mean()) / (v.std(ddof=0) + eps)
        lo, hi = v.quantile(p1), v.quantile(p99)
        vv = v.clip(lo, hi)
        mu, sd = vv.mean(), vv.std(ddof=0)
        return (vv - mu) / (sd + eps)
    return df.groupby("numerai_ticker", group_keys=False).apply(_wz)


def compute_dynamic_source_weights(work: pd.DataFrame,
                                   base_weights: Dict[str,float],
                                   months: int=18) -> Dict[str,float]:
    if work.empty or 'friday_date' not in work.columns:
        return base_weights.copy()
    end_dt = pd.to_datetime(work['friday_date']).max()
    start_dt = end_dt - pd.DateOffset(months=months)
    sub = work[(work['friday_date']>=start_dt)&(work['friday_date']<=end_dt)].copy()
    reliab = {c: float(sub[c].notna().mean()) for c in
              ['reddit_volume','wiki_views','gdelt_mentions','twitter_counts','google_trend'] if c in sub.columns}
    dyn = {}
    for k, base_w in base_weights.items():
        r = reliab.get(k, 0.0)
        if r < SOCIAL_RELIABILITY_MIN:
            dyn[k] = 0.0
        else:
            dyn[k] = base_w * max(1e-9, r)
    s = sum(dyn.values()) or 1.0
    return {k: (v/s if s else 0.0) for k,v in dyn.items()}


def build_query_terms(ticker: str, long_name: Optional[str]) -> str:
    if not long_name:
        return f'({ticker} OR ${ticker})'
    clean = re.sub(r'[^A-Za-z0-9 ]+', ' ', str(long_name))
    clean = re.sub(r'\s+', ' ', clean).strip()
    # bias toward equity context to reduce false positives
    return f'("{clean}" OR {ticker} OR ${ticker}) AND (stock OR shares OR Inc OR Corp OR company)'


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
        return data if isinstance(data, dict) else {}
    except Exception as err:
        log.warning(f"[auto] Failed to read {path.name}: {err}")
        return {}


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def get_latest_live_era(live_df: pd.DataFrame) -> Optional[str]:
    if live_df is None or live_df.empty:
        return None
    try:
        if "friday_date" in live_df.columns:
            era_series = pd.to_datetime(live_df["friday_date"], errors="coerce")
        elif "date" in live_df.columns:
            era_series = weekly_friday(live_df["date"])
        else:
            return None
        era_series = pd.to_datetime(era_series, errors="coerce").dt.tz_localize(None).dt.normalize()
        latest = era_series.max()
        if pd.isna(latest):
            return None
        return latest.strftime("%Y-%m-%d")
    except Exception as err:
        log.warning(f"[auto] Failed to derive latest era: {err}")
        return None


def _get_model_state(state: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    models = state.get("models", {})
    if isinstance(models, dict):
        return models.get(model_name, {}) if model_name else {}
    return {}


def already_submitted_for_era(model_name: str, latest_era: Optional[str]) -> bool:
    if not latest_era:
        return False
    state = _read_json(SUBMISSION_STATE_FILE)
    model_state = _get_model_state(state, model_name)
    return model_state.get("last_era") == latest_era


def update_submission_state(model_name: str, model_id: str, latest_era: str, live_path: str) -> None:
    state = _read_json(SUBMISSION_STATE_FILE)
    models = state.get("models")
    if not isinstance(models, dict):
        models = {}
        state["models"] = models
    models[model_name] = {
        "last_era": latest_era,
        "last_upload_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "last_file": live_path,
        "model_id": model_id,
        "data_version": DATA_VERSION,
    }
    _write_json_atomic(SUBMISSION_STATE_FILE, state)


# ----- API setup -------------------------------------------------------------

def setup_apis() -> Tuple[object, object, object, object, object]:
    napi = numerapi.SignalsAPI(DEFAULTS["NUMERAI_PUBLIC_ID"], DEFAULTS["NUMERAI_SECRET_KEY"]) if numerapi else None
    if napi:
        last_err: Optional[Exception] = None
        for attempt in range(NUMERAI_API_MAX_RETRIES):
            try:
                acct = napi.get_account()
                if not isinstance(acct, dict) or not acct:
                    raise RuntimeError("empty account payload")
                log.info("[numerai] API connected")
                break
            except Exception as err:
                last_err = err
                wait = min(10.0, NUMERAI_API_RETRY_BACKOFF * float(attempt + 1))
                log.warning(f"[numerai] account handshake failed (attempt {attempt+1}/{NUMERAI_API_MAX_RETRIES}): {err}")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Numerai API unavailable after {NUMERAI_API_MAX_RETRIES} attempts: {last_err}")
    else:
        raise RuntimeError("numerapi is required.")

    reddit = None
    praw = _try_import("praw")
    if praw and DEFAULTS["REDDIT_CLIENT_ID"] and DEFAULTS["REDDIT_CLIENT_SECRET"]:
        try:
            reddit = praw.Reddit(client_id=DEFAULTS["REDDIT_CLIENT_ID"],
                                 client_secret=DEFAULTS["REDDIT_CLIENT_SECRET"],
                                 user_agent=DEFAULTS["REDDIT_USER_AGENT"])  
            _ = reddit.user.me()
            log.info("[reddit] API connected (recent sentiment)")
        except Exception as e:
            log.warning(f"Reddit init failed: {e}")

    newsapi = None
    if NewsApiClient and DEFAULTS["NEWSAPI_KEY"]:
        try:
            newsapi = NewsApiClient(api_key=DEFAULTS["NEWSAPI_KEY"])
            log.info("[newsapi] Client ready")
        except Exception as e:
            log.warning(f"NewsAPI init failed: {e}")

    pytrends = None
    if TrendReq:
        try:
            pytrends = TrendReq(hl='en-US', tz=360)
            log.info("[pytrends] Client ready")
        except Exception as e:
            log.warning(f"PyTrends init failed: {e}")

    analyzer = None
    if SentimentIntensityAnalyzer:
        try:
            analyzer = SentimentIntensityAnalyzer()
            log.info("[vader] Analyzer ready")
        except Exception as e:
            log.warning(f"VADER init failed: {e}")

    return napi, reddit, newsapi, pytrends, analyzer


# ----- datasets --------------------------------------------------------------

def _read_parquet_columns(local_path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if not columns:
        return pd.read_parquet(local_path)
    try:
        import pyarrow.parquet as pq
        schema = pq.ParquetFile(local_path).schema
        available = set(schema.names)
        cols = [c for c in columns if c in available]
        if not cols:
            return pd.read_parquet(local_path)
        return pd.read_parquet(local_path, columns=cols)
    except Exception as err:
        log.warning(f"[data] Column-select read failed for {local_path.name}: {err}. Falling back to full read.")
        return pd.read_parquet(local_path)


def _load_or_download_dataset(napi, remote: str, local: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    local_path = Path(local)
    if local_path.exists():
        log.info(f"  Using cached dataset: {local_path.name}")
        return _read_parquet_columns(local_path, columns=columns)
    if not napi:
        raise RuntimeError(f"Dataset {remote} missing locally and Numerai API unavailable.")
    log.info(f"  Downloading {remote} -> {local_path.name}")
    napi.download_dataset(remote, local)
    return _read_parquet_columns(local_path, columns=columns)


def download_live_data(napi) -> pd.DataFrame:
    return _load_or_download_dataset(
        napi,
        f"{DATA_VERSION}/live.parquet",
        f"{DATA_VERSION.replace('/','_')}_live.parquet",
        columns=["numerai_ticker", "date", "friday_date"]
    )


def download_numerai_data(napi):
    log.info("[1/10] Preparing Numerai datasets...")
    train = _load_or_download_dataset(
        napi,
        f"{DATA_VERSION}/train.parquet",
        f"{DATA_VERSION.replace('/','_')}_train.parquet",
        columns=["numerai_ticker", "date", "target"]
    )
    valid = _load_or_download_dataset(
        napi,
        f"{DATA_VERSION}/validation.parquet",
        f"{DATA_VERSION.replace('/','_')}_validation.parquet",
        columns=["numerai_ticker", "date", "target"]
    )
    live = download_live_data(napi)
    t_weights = _load_or_download_dataset(
        napi,
        f"{DATA_VERSION}/train_sample_weights.parquet",
        f"{DATA_VERSION.replace('/','_')}_train_sample_weights.parquet",
        columns=["numerai_ticker", "date", "sample_weight", "sample_weights"]
    )
    try:
        v_weights = _load_or_download_dataset(
            napi,
            f"{DATA_VERSION}/validation_sample_weights.parquet",
            f"{DATA_VERSION.replace('/','_')}_validation_sample_weights.parquet",
            columns=["numerai_ticker", "date", "sample_weight", "sample_weights"]
        )
    except Exception:
        v_weights = pd.DataFrame()

    # Merge train + validation for a unified historical set; hold out recent eras.
    def _prep(df):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['numerai_ticker'] = df['numerai_ticker'].astype(str)
        return df
    train = _prep(train)
    valid = _prep(valid)
    all_df = pd.concat([train, valid], ignore_index=True)

    def _prep_w(df):
        if df is None or len(df) == 0:
            return pd.DataFrame()
        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        df['numerai_ticker'] = df['numerai_ticker'].astype(str)
        if 'sample_weights' in df.columns and 'sample_weight' not in df.columns:
            df = df.rename(columns={'sample_weights':'sample_weight'})
        return df
    all_weights = pd.concat([_prep_w(t_weights), _prep_w(v_weights)], ignore_index=True) if not isinstance(t_weights, int) else pd.DataFrame()

    holdout_cut = HOLDOUT_START_TS
    train_mask = all_df['date'] < holdout_cut
    holdout_mask = all_df['date'] >= holdout_cut
    train_hist = all_df[train_mask].copy()
    holdout_hist = all_df[holdout_mask].copy()
    w_train = all_weights[all_weights['date'] < holdout_cut] if not all_weights.empty else pd.DataFrame()
    w_holdout = all_weights[all_weights['date'] >= holdout_cut] if not all_weights.empty else pd.DataFrame()

    log.info(f"  Unified dataset rows: {len(all_df):,} (train={len(train_hist):,}, holdout={len(holdout_hist):,})")
    if holdout_hist.empty:
        log.warning("[data] Holdout window empty; consider adjusting HOLDOUT_START_DATE.")
    return train_hist, holdout_hist, live, w_train, w_holdout


def create_ticker_map(live_df: pd.DataFrame) -> pd.DataFrame:
    """Build yahoo<->Numerai ticker map, optionally using OpenSignals universe."""
    log.info("[2/10] Creating ticker map...")
    if USE_OPENSIGNALS_UNIVERSE and OpenSignalsProviderBase:
        try:
            ticker_map = OpenSignalsProviderBase.get_tickers()
            if not ticker_map.empty and {'yahoo', 'bloomberg_ticker'}.issubset(ticker_map.columns):
                tmap = ticker_map.rename(columns={'yahoo': 'yahoo_ticker'}).copy()
                tmap['yahoo_ticker'] = tmap['yahoo_ticker'].astype(str).str.replace('/', '-', regex=False)
                tmap['numerai_ticker'] = ticker_map['bloomberg_ticker'].astype(str)
                tmap = tmap[['numerai_ticker', 'yahoo_ticker']].dropna().drop_duplicates('yahoo_ticker')
                log.info(f"[opensignals] Universe map loaded: {len(tmap):,} tickers.")
                return tmap
        except Exception as err:
            log.warning(f"[opensignals] Universe map failed, falling back (err={err})")
    rows = []
    for nt in live_df['numerai_ticker'].unique():
        parts = str(nt).split()
        if not parts:
            continue
        sym = parts[0]
        country = 'US' if ('US' in parts) else None
        if country == 'US':
            rows.append({'numerai_ticker': nt, 'yahoo_ticker': sym.replace('/', '-')})
    tmap = pd.DataFrame(rows).drop_duplicates('yahoo_ticker')
    log.info(f"[ticker-map] Mapped {len(tmap):,} US tickers")
    return tmap


# ----- caching


# ----- caching ---------------------------------------------------------------

def _load_price_cache() -> pd.DataFrame:
    if PRICE_CACHE.exists():
        try:
            return pd.read_parquet(PRICE_CACHE)
        except Exception:
            pass
    return pd.DataFrame(columns=['date','yahoo_ticker','Close','Volume'])


def _save_price_cache(df: pd.DataFrame):
    if not df.empty:
        df.sort_values(['yahoo_ticker','date'], inplace=True)
        df.to_parquet(PRICE_CACHE, index=False)


def _download_prices_via_opensignals(symbols: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
    client = _get_opensignals_yahoo()
    if client is None:
        return None
    start_dt = pd.to_datetime(start).to_pydatetime()
    end_dt = pd.to_datetime(end).to_pydatetime()
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        try:
            _, data = client.download_ticker(sym, start=start_dt, end=end_dt)
        except Exception as err:
            log.warning(f"[opensignals] Yahoo download failed for {sym}: {err}")
            continue
        if data is None or data.empty:
            continue
        clean = data.rename(columns={'bloomberg_ticker': 'yahoo_ticker', 'adj_close': 'Close', 'volume': 'Volume'}).copy()
        if 'date' not in clean.columns or 'Close' not in clean.columns:
            continue
        clean['date'] = pd.to_datetime(clean['date']).dt.tz_localize(None).dt.normalize()
        for col in ['Close', 'Volume']:
            if col not in clean.columns:
                clean[col] = np.nan
        frames.append(clean[['date', 'yahoo_ticker', 'Close', 'Volume']])
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def get_price_data_cached(tmap: pd.DataFrame) -> pd.DataFrame:
    if yf is None and not (USE_OPENSIGNALS_YAHOO and OpenSignalsYahooCls):
        raise RuntimeError("yfinance is required.")
    log.info("[3/10] Fetching price data (incremental cache)...")
    cached = _load_price_cache()
    need = set(tmap['yahoo_ticker'])
    last_by = {}
    if not cached.empty:
        last_by = cached.groupby('yahoo_ticker')['date'].max().to_dict()
    hit_full = 0
    failed_tickers: List[str] = []
    groups = {}
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    end_str = (today + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    max_batch = int(os.getenv("YF_MAX_BATCH","50"))
    max_retries = int(os.getenv("YF_MAX_RETRIES","3"))
    backoff_base = float(os.getenv("YF_BACKOFF_BASE","2.0"))
    backoff_cap = float(os.getenv("YF_BACKOFF_CAP","30.0"))
    batch_sleep = float(os.getenv("YF_BATCH_SLEEP", "0"))
    use_threads = bool(int(os.getenv("YF_THREADS", "1")))
    for t in need:
        if t in last_by:
            start_dt = (pd.to_datetime(last_by[t]) + pd.Timedelta(days=1)).tz_localize(None).normalize()
            if start_dt >= today:
                hit_full += 1
                continue
            start = start_dt.strftime('%Y-%m-%d')
        else:
            start = DATA_START_DATE
        groups.setdefault(start, []).append(t)
    new_frames = []
    start_keys = list(groups.keys())

    def _download_chunk(symbols, start):
        last_err = None
        if USE_OPENSIGNALS_YAHOO:
            data = _download_prices_via_opensignals(symbols, start, end_str)
            if data is not None:
                return data
            if yf is None:
                return None
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    symbols, start=start, end=end_str,
                    progress=False, auto_adjust=True,
                    group_by='column', threads=use_threads
                )
                if data is not None and not data.empty:
                    return data
                last_err = RuntimeError("yfinance returned empty frame")
            except Exception as err:
                last_err = err
            sleep_for = min(backoff_cap, backoff_base * (2 ** attempt))
            sleep_for += random.uniform(0, 0.5)
            time.sleep(sleep_for)
        log.warning(f"yfinance download failed for batch start={start} (n={len(symbols)}): {last_err}")
        return None

    iterable = track(start_keys, description="YF batches") if track else start_keys
    for k in iterable:
        batch = groups[k]
        for i in range(0, len(batch), max_batch):
            sub = batch[i:i+max_batch]
            data = _download_chunk(sub, k)
            if data is None or data.empty:
                failed_tickers.extend(sub)
                continue
            try:
                if {'date','yahoo_ticker','Close','Volume'}.issubset(data.columns):
                    price_df = data[['date','yahoo_ticker','Close','Volume']].copy()
                else:
                    price_df = data.stack(level=1).reset_index()
                    ticker_col = price_df.columns[1]
                    price_df.rename(columns={ticker_col:'yahoo_ticker','Date':'date'}, inplace=True)
                    price_df['date'] = pd.to_datetime(price_df['date']).dt.tz_localize(None).dt.normalize()
                    price_df = price_df[['date','yahoo_ticker','Close','Volume']]
                new_frames.append(price_df)
            except Exception as err:
                log.warning(f"Failed to process price frame (start={k}, n={len(sub)}): {err}")
                continue
            if batch_sleep > 0:
                time.sleep(batch_sleep)
    out = cached.copy()
    if new_frames:
        out = pd.concat([cached] + new_frames, ignore_index=True)
    out = out.drop_duplicates(['yahoo_ticker','date'], keep='last')
    _save_price_cache(out)
    hit_ratio = hit_full / max(len(need), 1)
    log.info(f"[prices] Cached data: {out['yahoo_ticker'].nunique():,} tickers, {len(out):,} rows | cache full-hit {hit_full}/{len(need)} ({hit_ratio:.1%})")
    if failed_tickers:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        miss_path = LOG_DIR / f"missing_prices_{ts}.csv"
        pd.DataFrame({"yahoo_ticker": failed_tickers}).drop_duplicates().to_csv(miss_path, index=False)
        log.warning(f"[prices] Missing/failed tickers logged: {miss_path} (n={len(set(failed_tickers))})")
    return out


# ----- social caches & meta --------------------------------------------------

def _load_social_cache() -> pd.DataFrame:
    if SOCIAL_CACHE.exists():
        try:
            return pd.read_parquet(SOCIAL_CACHE)
        except Exception:
            pass
    return pd.DataFrame(columns=['friday_date','numerai_ticker'])


def _save_social_cache(df: pd.DataFrame):
    if not df.empty:
        df.sort_values(['numerai_ticker','friday_date'], inplace=True)
        df.to_parquet(SOCIAL_CACHE, index=False)


def _load_meta_cache() -> pd.DataFrame:
    if META_CACHE.exists():
        try:
            return pd.read_parquet(META_CACHE)
        except Exception:
            pass
    return pd.DataFrame(columns=['yahoo_ticker','long_name','sector','market_cap'])


def _save_meta_cache(df: pd.DataFrame):
    if not df.empty:
        df = df.drop_duplicates('yahoo_ticker')
        df.to_parquet(META_CACHE, index=False)


def add_sector_mcap_meta(tmap: pd.DataFrame) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is required.")
    cached = _load_meta_cache()
    need = [t for t in tmap['yahoo_ticker'] if t not in set(cached['yahoo_ticker'])]
    if need:
        for i in (track(range(len(need)), description="Meta (yfinance)") if track else range(len(need))):
            t = need[i]
            try:
                info = yf.Ticker(t).info
            except Exception:
                info = {}
            cached = pd.concat([
                cached,
                pd.DataFrame([{
                    'yahoo_ticker': t,
                    'long_name': info.get('longName') or info.get('shortName'),
                    'sector': info.get('sector') or 'UNK',
                    'market_cap': info.get('marketCap') or np.nan
                }])
            ], ignore_index=True)
            time.sleep(0.03)
        _save_meta_cache(cached)

    meta = cached.copy()
    meta['sector'] = meta['sector'].fillna('UNK').replace('','UNK')
    return meta[['yahoo_ticker','long_name','sector','market_cap']]


# ----- social signals (historical, cached) -----------------------------------

def _yf_name(y: str) -> str:
    try:
        info = yf.Ticker(y).info
        return str(info.get('longName') or info.get('shortName') or y)
    except Exception:
        return y


def _wiki_search_title(q: str) -> str:
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action":"query","list":"search","srsearch":q,"format":"json","srlimit":1}
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code==200:
            hits = r.json().get('query',{}).get('search',[])
            if hits:
                return hits[0].get('title', q).replace(' ','_')
    except Exception:
        pass
    return q.replace(' ','_')


def _wiki_pageviews(title: str, start: str, end: str) -> pd.DataFrame:
    frames=[]
    try:
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{title}/daily/{start}/{end}"
        r = requests.get(url, timeout=20)
        if r.status_code==200:
            it = r.json().get('items',[])
            if it:
                df = pd.DataFrame(it)
                df['date'] = pd.to_datetime(df['timestamp'].str[:8])
                df['wiki_views'] = df['views']
                frames.append(df[['date','wiki_views']])
    except Exception: 
        pass
    try:
        url = f"https://wikimedia.org/api/rest_v1/metrics/legacy/pagecounts/per-article/en.wikipedia/all-sites/all-agents/{title}/daily/2008010100/2016070100"
        r = requests.get(url, timeout=20)
        if r.status_code==200:
            it = r.json().get('items',[])
            if it:
                df = pd.DataFrame(it)
                df['date'] = pd.to_datetime(df['timestamp'].str[:8])
                df['wiki_views'] = df['count']
                frames.append(df[['date','wiki_views']])
    except Exception: 
        pass
    if not frames: 
        return pd.DataFrame(columns=['date','wiki_views'])
    out = pd.concat(frames, ignore_index=True)
    out = out.groupby('date', as_index=False)['wiki_views'].sum()
    return out


def get_historical_social_signals_cached(tmap: pd.DataFrame,
                                         pytrends,
                                         meta: pd.DataFrame,
                                         refresh_social: bool=False) -> pd.DataFrame:
    log.info("[5/10] Collecting HISTORICAL Social Signals (composite, cached)...")
    cached = _load_social_cache()
    need_refresh = refresh_social or cached.empty
    if not cached.empty:
        log.info(f"  [social-cache] Hit: {cached['numerai_ticker'].nunique():,} tickers, {cached['friday_date'].nunique():,} eras, rows={len(cached):,}")

    if need_refresh:
        def since_map(col):
            if cached.empty or col not in cached.columns:
                return {}
            return cached.dropna(subset=[col]).groupby('numerai_ticker')['friday_date'].max().to_dict()

        since_reddit  = since_map('reddit_volume')
        since_wiki    = since_map('wiki_views')
        since_gdelt   = since_map('gdelt_mentions')
        since_trends  = since_map('google_trend')

        today = pd.Timestamp.utcnow().tz_localize(None).normalize()
        last_friday_target = today - pd.Timedelta(days=(today.weekday() - 4) % 7)
        tmap_rows = list(tmap.itertuples(index=False))

        def _coerce_last(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            try:
                ts = pd.Timestamp(val)
            except Exception:
                return None
            if pd.isna(ts):
                return None
            if ts.tzinfo is not None:
                ts = ts.tz_convert(None)
            return ts.normalize()

        def _pending_rows(since_dict, rows=None):
            rows = rows or tmap_rows
            pend = []
            for r in rows:
                nt = r.numerai_ticker
                last_dt = _coerce_last(since_dict.get(nt))
                if last_dt is None or last_dt < last_friday_target:
                    pend.append((r, last_dt))
            return pend

        reddit_df = pd.DataFrame()
        if ENABLE_PUSHSHIFT and requests is not None:
            pending = _pending_rows(since_reddit)
            if not pending:
                log.info("  Reddit cache up-to-date; skipping Pushshift fetch.")
            else:
                rows=[]
                for r, last_dt in (track(pending, description="Reddit (Pushshift)") if track else pending):
                    yt, nt = r.yahoo_ticker, r.numerai_ticker
                    long_name = meta.set_index('yahoo_ticker').get('long_name',{}).get(yt) if not meta.empty else None
                    after = (last_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d') if last_dt is not None else '2005-01-01'
                    base = "https://api.pushshift.io/reddit/search/submission/"
                    params = {"q": build_query_terms(yt, long_name), "after":after, "size":0, "aggs":"created_utc", "frequency":"week"}
                    try:
                        resp = requests.get(base, params=params, timeout=20)
                        if resp.status_code==200:
                            buckets = resp.json().get('aggs',{}).get('created_utc',[])
                            for b in buckets:
                                dt = datetime.utcfromtimestamp(b['key'])
                                rows.append({
                                    "friday_date": weekly_friday(pd.Series([dt]))[0],
                                    "numerai_ticker": nt,
                                    "reddit_volume": b.get('doc_count',0)
                                })
                    except Exception:
                        pass
                reddit_df = pd.DataFrame(rows)

        wiki_df = pd.DataFrame()
        if ENABLE_WIKIPEDIA and requests is not None:
            pending = _pending_rows(since_wiki)
            if not pending:
                log.info("  Wikipedia cache up-to-date; skipping fetch.")
            else:
                parts=[]
                now = datetime.utcnow().strftime("%Y%m%d00")
                for r, last_dt in (track(pending, description="Wikipedia views") if track else pending):
                    yt, nt = r.yahoo_ticker, r.numerai_ticker
                    long_name = meta.set_index('yahoo_ticker').get('long_name',{}).get(yt) if not meta.empty else _yf_name(yt)
                    start = (last_dt + pd.Timedelta(days=1)).strftime('%Y%m%d00') if last_dt is not None else "2008010100"
                    try:
                        title = _wiki_search_title(long_name)
                        df = _wiki_pageviews(title, start, now)
                        if not df.empty:
                            tmp = df.copy()
                            tmp['friday_date'] = weekly_friday(tmp['date'])
                            tmp = tmp.groupby('friday_date', as_index=False)['wiki_views'].sum()
                            tmp['numerai_ticker'] = nt
                            parts.append(tmp[['friday_date','numerai_ticker','wiki_views']])
                    except Exception:
                        pass
                if parts:
                    wiki_df = pd.concat(parts, ignore_index=True)

        gdelt_df = pd.DataFrame()
        if ENABLE_GDELT and requests is not None:
            pending = _pending_rows(since_gdelt)
            if not pending:
                log.info("  GDELT cache up-to-date; skipping fetch.")
            else:
                rows=[]
                for r, last_dt in (track(pending, description="GDELT timeline") if track else pending):
                    yt, nt = r.yahoo_ticker, r.numerai_ticker
                    long_name = meta.set_index('yahoo_ticker').get('long_name',{}).get(yt) if not meta.empty else _yf_name(yt)
                    startdt = (last_dt + pd.Timedelta(days=1)).strftime('%Y%m%d%H%M%S') if last_dt is not None else '20040101000000'
                    base = "https://api.gdeltproject.org/api/v2/doc/doc"
                    params = {"query": build_query_terms(yt, long_name), "mode":"TimelineVol", "format":"json","startdatetime":startdt}
                    try:
                        resp = requests.get(base, params=params, timeout=30)
                        if resp.status_code==200:
                            for it in resp.json().get('timeline', []):
                                dt = pd.to_datetime(it.get('date'))
                                cnt = int(it.get('value',0))
                                rows.append({"friday_date": weekly_friday(pd.Series([dt]))[0], "numerai_ticker": nt, "gdelt_mentions": cnt})
                    except Exception:
                        pass
                gdelt_df = pd.DataFrame(rows)

        gt_df = pd.DataFrame()
        if ENABLE_GOOGLE_TRENDS and TrendReq and pytrends is not None:
            rows=[]
            gt_candidates = list(tmap.head(min(GOOGLE_TRENDS_TICKER_COUNT, len(tmap))).itertuples(index=False))
            pending = _pending_rows(since_trends, gt_candidates)
            if not pending:
                log.info("  Google Trends cache up-to-date; skipping fetch.")
            else:
                for r, last_dt in (track(pending, description="Google Trends") if track else pending):
                    yt, nt = r.yahoo_ticker, r.numerai_ticker
                    if last_dt is not None:
                        past = (last_dt - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
                        timeframe = f"{past} {datetime.utcnow().strftime('%Y-%m-%d')}"
                    else:
                        timeframe = 'all'
                    try:
                        pytrends.build_payload([yt], timeframe=timeframe, geo='US')
                        df = pytrends.interest_over_time()
                        if not df.empty and yt in df.columns:
                            tmp = df[[yt]].rename(columns={yt:'google_trend'}).reset_index().rename(columns={'date':'friday_date'})
                            tmp['friday_date'] = weekly_friday(tmp['friday_date'])
                            tmp['numerai_ticker'] = nt
                            rows.append(tmp[['friday_date','numerai_ticker','google_trend']])
                        time.sleep(1.5)
                    except Exception:
                        time.sleep(0.5)
                        continue
                if rows:
                    gt_df = pd.concat(rows, ignore_index=True)

        parts = [d for d in [reddit_df, wiki_df, gdelt_df, gt_df] if isinstance(d, pd.DataFrame) and not d.empty]
        if parts:
            new_df = parts[0]
            for d in parts[1:]:
                new_df = pd.merge(new_df, d, on=['friday_date','numerai_ticker'], how='outer')
            agg = {c:'sum' for c in new_df.columns if c not in ['friday_date','numerai_ticker']}
            new_df = new_df.groupby(['friday_date','numerai_ticker'], as_index=False).agg(agg)
            work = pd.concat([cached, new_df], ignore_index=True).drop_duplicates(['friday_date','numerai_ticker'], keep='last')
        else:
            work = cached.copy()
    else:
        work = cached.copy()
        log.info("  Social cache found; skipping refresh (use --refresh-social-cache to force).")

    if work.empty:
        return work

    if 'friday_date' in work.columns:
        work['friday_date'] = pd.to_datetime(work['friday_date']).dt.tz_localize(None).dt.normalize()
    if need_refresh:
        _save_social_cache(work)
        log.info(f"  [social-cache] Updated: {work['numerai_ticker'].nunique():,} tickers, {work['friday_date'].nunique():,} eras, rows={len(work):,}")

    for col in ['reddit_volume','wiki_views','gdelt_mentions','twitter_counts','google_trend']:
        if col in work.columns:
            work[f'{col}_z'] = winsorize_zscore_per_ticker(work, col)

    rel_cols = [c for c in ['reddit_volume','wiki_views','gdelt_mentions','twitter_counts','google_trend'] if c in work.columns]
    if rel_cols:
        rel = work.groupby('numerai_ticker').agg({c:lambda s: s.notna().mean() for c in rel_cols})
        low_rel = set(rel.index[(rel.fillna(0) < SOCIAL_RELIABILITY_MIN).any(axis=1)])
        if low_rel:
            work = work[~work['numerai_ticker'].isin(low_rel)]

    dyn_w = compute_dynamic_source_weights(work, WEIGHTS_BASE, months=18)
    def _row_w(row):
        s=0.0; wsum=0.0
        for c in ['reddit_volume','wiki_views','gdelt_mentions','twitter_counts','google_trend']:
            zc = f"{c}_z"
            if zc in work.columns and not pd.isna(row.get(zc)):
                w = dyn_w.get(c, 0.0)
                s += w*row.get(zc,0.0); wsum += w
        return s/(wsum+1e-12) if wsum>0 else 0.0
    work['social_index'] = work.apply(_row_w, axis=1)
    work['social_index_roc1'] = work.groupby('numerai_ticker')['social_index'].diff(1)
    if 'google_trend' in work.columns:
        work['google_trend_roc1'] = work.groupby('numerai_ticker')['google_trend'].diff(1)

    work['social_index_ema8'] = work.groupby('numerai_ticker')['social_index'].transform(lambda x: x.ewm(span=8, adjust=False).mean())
    work['social_index_std8'] = work.groupby('numerai_ticker')['social_index'].transform(lambda x: x.rolling(8, min_periods=3).std())
    if 'social_index_std8' in work.columns:
        noisy = work['social_index_std8'] > SOCIAL_NOISE_THRESHOLD
        work.loc[noisy, 'social_index'] = work.loc[noisy, 'social_index_ema8']

    if 'google_trend' in work.columns:
        work['google_trend_ema8'] = work.groupby('numerai_ticker')['google_trend'].transform(lambda x: x.ewm(span=8, adjust=False).mean())
        work['google_trend_std8'] = work.groupby('numerai_ticker')['google_trend'].transform(lambda x: x.rolling(8, min_periods=3).std())
        noisy_gt = work['google_trend_std8'] > SOCIAL_NOISE_THRESHOLD
        work.loc[noisy_gt, 'google_trend'] = work.loc[noisy_gt, 'google_trend_ema8']

    keep = ['friday_date','numerai_ticker','social_index','social_index_roc1']
    for extra in ['social_index_ema8','social_index_std8']:
        if extra in work.columns:
            keep.append(extra)
    for c in ['reddit_volume','wiki_views','gdelt_mentions','twitter_counts','google_trend','google_trend_roc1']:
        if c in work.columns:
            keep.append(c)
            if f"{c}_z" in work.columns:
                keep.append(f"{c}_z")
            if c == 'google_trend':
                for extra in ['google_trend_ema8','google_trend_std8']:
                    if extra in work.columns:
                        keep.append(extra)
    out = work[keep].copy()
    log.info(f"  Composite rows (cached): {len(out):,}")
    return out


# ----- features --------------------------------------------------------------

def neutralize_features_cross_sectional(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df['sector'] = df['sector'].fillna('UNK')
    if 'size_bucket' not in df.columns:
        df['size_bucket'] = -1
    df['size_bucket'] = df['size_bucket'].fillna(-1)
    keys = ['friday_date','sector','size_bucket']
    for c in cols:
        if c in df.columns:
            df[c] = df[c] - df.groupby(keys)[c].transform('mean')
    return df


def _standardize(a: np.ndarray) -> np.ndarray:
    mu = np.nanmean(a, axis=0)
    sd = np.nanstd(a, axis=0, ddof=0)
    sd[sd==0] = 1.0
    return (a - mu) / sd


def neutralize_to_factors(df: pd.DataFrame, ycol: str, factor_cols: List[str], keys: List[str], strength: float=1.0) -> pd.Series:
    """OLS neutralization y ~ X by group keys; returns partially neutralized y.
    new_y = y - strength * (X_beta)
    """
    y_new = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby(keys):
        idx = g.index
        y = g[ycol].astype(float).values
        X = g[factor_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
        # add intercept
        X = np.column_stack([np.ones(len(g)), _standardize(X)])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            y_hat = X @ beta
            y_new.loc[idx] = y - strength*y_hat
        except Exception:
            y_new.loc[idx] = y
    return y_new


def add_advanced_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    def rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta>0,0)).rolling(period).mean()
        loss = (-delta.where(delta<0,0)).rolling(period).mean()
        rs = gain/(loss+1e-10)
        return 100-(100/(1+rs))
    df['rsi_14'] = df.groupby('numerai_ticker')['Close'].transform(lambda x: rsi(x,14))
    df['rsi_7']  = df.groupby('numerai_ticker')['Close'].transform(lambda x: rsi(x,7))
    df['rsi_21'] = df.groupby('numerai_ticker')['Close'].transform(lambda x: rsi(x,21))
    df['bb_mid'] = df.groupby('numerai_ticker')['Close'].transform(lambda x: x.rolling(20).mean())
    df['bb_std'] = df.groupby('numerai_ticker')['Close'].transform(lambda x: x.rolling(20).std())
    df['bb_up']  = df['bb_mid'] + 2*df['bb_std']
    df['bb_lo']  = df['bb_mid'] - 2*df['bb_std']
    df['bb_width'] = (df['bb_up']-df['bb_lo'])/(df['bb_mid']+1e-10)
    df['bb_position'] = (df['Close']-df['bb_lo'])/((df['bb_up']-df['bb_lo'])+1e-10)
    df['ema_12'] = df.groupby('numerai_ticker')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df['ema_26'] = df.groupby('numerai_ticker')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df.groupby('numerai_ticker')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['macd_diff'] = df['macd'] - df['macd_signal']
    for p in [3,5,10,20,60]:
        df[f'momentum_{p}'] = df.groupby('numerai_ticker')['Close'].transform(lambda x: x.pct_change(p))
    df['volume_ratio'] = df.groupby('numerai_ticker')['Volume'].transform(lambda x: x/(x.rolling(20).mean()+1e-10))
    df['volume_momentum'] = df.groupby('numerai_ticker')['Volume'].transform(lambda x: x.pct_change(5))
    df['pv_corr'] = df.groupby('numerai_ticker').apply(
        lambda g: g[['Close','Volume']].rolling(20).corr().unstack()['Close']['Volume']
    ).reset_index(level=0, drop=True)
    df['deviation_from_ma20'] = (df['Close']-df['bb_mid'])/(df['bb_mid']+1e-10)
    df['deviation_from_ma60'] = df.groupby('numerai_ticker')['Close'].transform(
        lambda x: (x-x.rolling(60).mean())/(x.rolling(60).mean()+1e-10)
    )
    df['volatility_20d'] = df.groupby('numerai_ticker')['Close'].transform(lambda x: x.pct_change().rolling(20).std())
    df['volatility_60d'] = df.groupby('numerai_ticker')['Close'].transform(lambda x: x.pct_change().rolling(60).std())
    df['volatility_ratio'] = df['volatility_20d']/(df['volatility_60d']+1e-10)
    df.drop(['bb_mid','bb_std','bb_up','bb_lo','ema_12','ema_26','macd','macd_signal'], axis=1, errors='ignore', inplace=True)
    return df


def _apply_opensignals_feature_templates(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Optionally add OpenSignals RSI/SMA/VarChange features to the price frame."""
    if not ENABLE_OPENSIGNALS_FEATURES:
        return df, []
    generators = _get_opensignals_generators()
    required = {'numerai_ticker', 'date', 'Close'}
    if not generators or not required.issubset(df.columns):
        return df, []
    os_df = df[['numerai_ticker', 'date', 'Close']].copy()
    os_df = os_df.rename(columns={'numerai_ticker': 'bloomberg_ticker'})
    os_df['date'] = pd.to_datetime(os_df['date']).dt.tz_localize(None).dt.normalize()
    os_df['adj_close'] = os_df['Close']
    for field in ['open', 'high', 'low', 'close']:
        os_df[field] = os_df['adj_close']
    if 'Volume' in df.columns:
        os_df['Volume'] = df['Volume'].values
    os_df['volume'] = os_df.get('Volume', np.nan)
    feature_cols: List[str] = []
    for gen in generators:
        try:
            os_df, feats = gen.generate_features(os_df, feature_prefix=OPENSIGNALS_FEATURE_PREFIX)
            if feats:
                feature_cols.extend(feats)
        except Exception as err:
            log.warning(f"[opensignals] Feature generation failed: {err}")
    feature_cols = [c for c in feature_cols if c in os_df.columns]
    if not feature_cols:
        return df, []
    merged = os_df.rename(columns={'bloomberg_ticker': 'numerai_ticker'})
    keep = ['numerai_ticker', 'date'] + feature_cols
    merged = merged[keep].drop_duplicates(['numerai_ticker', 'date'])
    df = df.merge(merged, on=['numerai_ticker', 'date'], how='left')
    return df, feature_cols



def neutralize_features_cross_sectional(df: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    """Standard Global Neutralization (Memory Intensive but Accurate)."""
    # Ensure keys exist
    if 'sector' not in df.columns:
        df['sector'] = 'UNK'
    else:
        df['sector'] = df['sector'].fillna('UNK')
        
    if 'size_bucket' not in df.columns:
        df['size_bucket'] = -1
    else:
        df['size_bucket'] = df['size_bucket'].fillna(0)
        
    # Standard Keys: Date (Daily) + Sector + Size (Correct Logic)
    neu_keys = ['date', 'sector', 'size_bucket']
    
    # Global Transformation (Uses Copy implicitly or explicitly)
    # This matches the canonical logic.
    grouped = df.groupby(neu_keys)
    
    for c in feats:
        if c in df.columns:
            try:
                df[c] = df[c] - grouped[c].transform('mean')
            except Exception:
                pass
                
    return df

# Removed _finalize_features global in-place version to avoid confusion.
# Logic moved back to merge_features or neutralize_features_cross_sectional.


def merge_features(price_df: pd.DataFrame, tmap: pd.DataFrame,
                   hist_social: pd.DataFrame, meta: pd.DataFrame,
                   recent_media: Optional[pd.DataFrame]=None,
                   finalize: bool=True) -> Tuple[pd.DataFrame, List[str]]:
    log.info("    [6a] Merging price/meta & generating base features...")
    df = price_df.merge(tmap, on='yahoo_ticker', how='inner').sort_values(['numerai_ticker','date'])
    meta_cols = [c for c in ['yahoo_ticker','sector'] if c in meta.columns]
    if meta_cols:
        df = df.merge(meta[meta_cols], on='yahoo_ticker', how='left')
    df['sector'] = df['sector'].fillna('UNK')
    # price/volume features
    df['return_5d']  = df.groupby('numerai_ticker')['Close'].pct_change(5)
    df['return_20d'] = df.groupby('numerai_ticker')['Close'].pct_change(20)
    df['return_60d'] = df.groupby('numerai_ticker')['Close'].pct_change(60)
    df['volatility_20d'] = df.groupby('numerai_ticker')['Close'].transform(lambda x: x.pct_change().rolling(20).std())
    df['dollar_volume'] = df['Close'] * df['Volume']
    df['log_dollar_volume'] = np.log1p(df['dollar_volume'].clip(lower=1e-9))
    df['volume_ma20'] = df.groupby('numerai_ticker')['Volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df['volume_strength'] = df['Volume'] / (df['volume_ma20'] + 1e-9)
    df['acceleration_5_20'] = df['return_5d'] - df['return_20d']
    df['momentum_ratio_5_20'] = df['return_5d'] / (df['return_20d'].abs() + 1e-9)
    df = add_advanced_technical_indicators(df)
    size_roll = df.groupby('numerai_ticker')['log_dollar_volume'].transform(lambda x: x.rolling(60, min_periods=20).mean())
    df['size_proxy'] = size_roll.fillna(df['log_dollar_volume'])

    pf = [
        'return_5d','return_20d','return_60d','volatility_20d',
        'rsi_7','rsi_14','rsi_21','bb_position','macd_diff','volume_ratio',
        'momentum_3','momentum_5','momentum_10','momentum_20','momentum_60',
        'volume_momentum','pv_corr','deviation_from_ma20','deviation_from_ma60',
        'volatility_60d','volatility_ratio','log_dollar_volume','volume_strength',
        'acceleration_5_20','momentum_ratio_5_20','size_proxy'
    ]
    if ENABLE_OPENSIGNALS_FEATURES:
        df, os_cols = _apply_opensignals_feature_templates(df)
        if os_cols:
            log.info(f"    [6a-os] OpenSignals templates added ({len(os_cols)} cols).")
            pf.extend(os_cols)
    if 'has_social' in df.columns:
        df['has_social'] = df['has_social'].astype(float)
        pf.append('has_social')
    log.info("    [6b] Incorporating recent media signals...")
    sf=[]
    if isinstance(recent_media, pd.DataFrame) and not recent_media.empty:
        m = recent_media.copy()
        df = df.merge(m, on=['date','numerai_ticker'], how='left')
        for c in m.columns:
            if c.endswith('_sentiment'):
                ma = f"{c}_ma7"
                df[ma] = df.groupby('numerai_ticker')[c].transform(lambda x: x.fillna(0).rolling(7, min_periods=1).mean())
                sf.append(ma)
    log.info("    [6c] Weekly alignment & rank transforms...")
    df['friday_date'] = weekly_friday(df['date']).dt.tz_localize(None).dt.normalize()
    df = df[df['friday_date'] >= TRAIN_START_TS]
    df['volume_rank'] = df.groupby('friday_date')['Volume'].transform(_cs_rank)
    df['dollar_volume_rank'] = df.groupby('friday_date')['dollar_volume'].transform(_cs_rank)
    df['volume_strength_rank'] = df.groupby('friday_date')['volume_strength'].transform(_cs_rank)
    df['size_rank'] = df.groupby('friday_date')['size_proxy'].transform(_cs_rank)
    df['size_bucket'] = (df['size_rank']*5).fillna(0).astype(int).clip(0,4)
    pf += ['volume_rank','dollar_volume_rank','volume_strength_rank','size_rank']
    price_rank_cols = []
    for col in ['return_20d','return_60d','volatility_20d']:
        if col in df.columns:
            rank_col = f"{col}_rank"
            df[rank_col] = df.groupby('friday_date')[col].transform(_cs_rank)
            price_rank_cols.append(rank_col)
    social_feature_cols: List[str] = []
    if ENABLE_SOCIAL_FEATURES and not hist_social.empty:
        log.info("    [6d] Adding historical social composites...")
        hs = hist_social.copy()
        df = df.merge(hs, on=['friday_date','numerai_ticker'], how='left')
        if 'google_trend' in df.columns:
            df['google_trend_ma4']  = df.groupby('numerai_ticker')['google_trend'].transform(lambda x: x.rolling(4, min_periods=1).mean())
            df['google_trend_ema8'] = df.groupby('numerai_ticker')['google_trend'].transform(lambda x: x.ewm(span=EMA_SPAN, adjust=False).mean())
            df['google_trend_roc1'] = df.groupby('numerai_ticker')['google_trend'].diff(1)
            df['google_trend_rank'] = df.groupby('friday_date')['google_trend'].transform(_cs_rank)
            df['google_trend_cs_z'] = df.groupby('friday_date')['google_trend'].transform(_cs_zscore)
            sf += ['google_trend_ma4','google_trend_ema8','google_trend_roc1','google_trend_rank','google_trend_cs_z']
        if 'social_index' in df.columns:
            df['social_index_ma4']  = df.groupby('numerai_ticker')['social_index'].transform(lambda x: x.rolling(4, min_periods=1).mean())
            df['social_index_ema8'] = df.groupby('numerai_ticker')['social_index'].transform(lambda x: x.ewm(span=EMA_SPAN, adjust=False).mean())
            df['social_index_roc1'] = df.groupby('numerai_ticker')['social_index'].diff(1)
            df['social_index_rank'] = df.groupby('friday_date')['social_index'].transform(_cs_rank)
            df['social_index_cs_z'] = df.groupby('friday_date')['social_index'].transform(_cs_zscore)
            sf += ['social_index_ma4','social_index_ema8','social_index_roc1','social_index_rank','social_index_cs_z']
        if 'analyst_buzz' in df.columns:
            df['analyst_buzz_rank'] = df.groupby('friday_date')['analyst_buzz'].transform(_cs_rank)
            sf.append('analyst_buzz_rank')
        if 'analyst_sentiment' in df.columns:
            df['analyst_sentiment_rank'] = df.groupby('friday_date')['analyst_sentiment'].transform(_cs_rank)
            sf.append('analyst_sentiment_rank')
        if {'return_20d_rank','social_index_rank'}.issubset(df.columns):
            df['mom20_x_social'] = df['return_20d_rank'] * df['social_index_rank']
            sf.append('mom20_x_social')
        if {'return_60d_rank','google_trend_rank'}.issubset(df.columns):
            df['mom60_x_trend'] = df['return_60d_rank'] * df['google_trend_rank']
            sf.append('mom60_x_trend')
        if 'analyst_buzz_rank' in df.columns and 'rsi_14' in df.columns:
            rsi_rank = df.groupby('friday_date')['rsi_14'].transform(_cs_rank)
            df['oversold_attention'] = (1 - rsi_rank) * df['analyst_buzz_rank']
            sf.append('oversold_attention')
        if {'return_20d_rank','analyst_sentiment_rank'}.issubset(df.columns):
            df['bullish_rating_momentum'] = df['return_20d_rank'] * df['analyst_sentiment_rank']
            sf.append('bullish_rating_momentum')
        social_feature_cols = [c for c in sf if c in df.columns]
        if social_feature_cols:
            if SOCIAL_FEATURE_WHITELIST:
                keep = [c for c in social_feature_cols if c in SOCIAL_FEATURE_WHITELIST]
                drop = [c for c in social_feature_cols if c not in SOCIAL_FEATURE_WHITELIST]
            else:
                keep = social_feature_cols
                drop = []
            if drop:
                df.drop(columns=drop, inplace=True, errors='ignore')
                social_feature_cols = keep
            if social_feature_cols and abs(SOCIAL_FEATURE_WEIGHT - 1.0) > 1e-9:
                for col in social_feature_cols:
                    df[col] = df[col] * SOCIAL_FEATURE_WEIGHT
            sf = list(social_feature_cols)
    else:
        log.info("    [6d] Skipping historical social composites (disabled or empty).")

    # [Re-Adding Initialization for Standard Logic]
    rel_cols = []
    base_candidates = ['return_20d','return_60d','momentum_20','volatility_20d']
    targets = [c for c in base_candidates if c in df.columns]

    if targets:
        # Standard Global Interaction Features
        log.info("    [6e] Building interaction & sector-relative features (Global)...")
        # Ensure sector is filled
        df['sector'] = df['sector'].fillna('UNK')
        
        # Global GroupBy by Date (Daily) - Correct Logic
        sector_group = df.groupby(['date','sector'])
        
        for base in targets:
            rel_name = f"{base}_sector_rel"
            rel_cols.append(rel_name)
            try:
                mean_vals = sector_group[base].transform('mean')
                df[rel_name] = df[base] - mean_vals
            except Exception:
                df[rel_name] = 0.0
                
    # Fallback/fill for safety
    for col in rel_cols:
        df[col] = df[col].fillna(0.0)
    pf.extend(rel_cols + price_rank_cols)
    feats = pf + sf

    if finalize:
        # Standard Cleanup
        log.info("    [6f] Finalizing features (Standard Cleanup & Neutralization)...")
        # 1. FillNA / Inf
        df[feats] = df[feats].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        # 2. Neutralize (Global)
        if NEUTRALIZE_FEATURES:
            log.info("        Neutralizing features (Standard Global)...")
            df = neutralize_features_cross_sectional(df, feats)
            
    return df, feats


def create_training_frame(price_df, tmap, hist_social, meta, train_df, weights_df, recent_media=None):
    log.info("[6/10] Creating features for training (v7.5)")
    # Restore standard flow (finalize=True) to ensure clean features before row reduction
    feat_df, feats = merge_features(price_df, tmap, hist_social, meta, recent_media, finalize=True)
    
    log.info("    [6f+] Collapsing daily rows to Friday snapshots...")
    before_rows = len(feat_df)
    feat_df = feat_df.sort_values(['numerai_ticker','date']).drop_duplicates(
        ['numerai_ticker','friday_date'], keep='last'
    )
    log.info(f"        Rows reduced: {before_rows:,} -> {len(feat_df):,}")
    
    # Cleanup was already done in merge_features (Iteratively)
    
    log.info("    [6g] Dropping sparse tickers...")
    keep = []
    for t, g in feat_df.groupby('numerai_ticker'):
        miss = g[feats].isna().mean().mean()
        if miss <= MISSING_RATIO_DROP: 
            keep.append(t)
    feat_df = feat_df[feat_df['numerai_ticker'].isin(keep)].copy()
    log.info("    [6h] Attaching targets & filtering by TRAIN_START_TS...")
    tgt = train_df[['numerai_ticker','date','target']].rename(columns={'date':'friday_date'})
    tgt['friday_date'] = pd.to_datetime(tgt['friday_date']).dt.tz_localize(None).dt.normalize()
    w_df = pd.DataFrame()
    if weights_df is not None and not weights_df.empty:
        w_df = weights_df.rename(columns={'date':'friday_date'})
        if 'sample_weights' in w_df.columns and 'sample_weight' not in w_df.columns:
            w_df = w_df.rename(columns={'sample_weights':'sample_weight'})
        if 'friday_date' in w_df.columns:
            w_df['friday_date'] = pd.to_datetime(w_df['friday_date']).dt.tz_localize(None).dt.normalize()
    feat_df = feat_df.merge(tgt, on=['numerai_ticker','friday_date'], how='left')
    if not w_df.empty:
        feat_df = feat_df.merge(w_df, on=['numerai_ticker','friday_date'], how='left')
    if 'sample_weight' in feat_df.columns:
        feat_df['sample_weight'] = feat_df['sample_weight'].fillna(0.0)
    log.info(f"[features] Final count: {len(feats)} | Train samples: {feat_df['target'].notna().sum():,}")
    return feat_df, feats


# ----- models ----------------------------------------------------------------

def _spearman(y_true, y_pred):
    try:
        return stats.spearmanr(y_true, y_pred, nan_policy='omit')[0]
    except Exception:
        return np.nan


def _weighted_mae(y_true, y_pred, weights=None):
    err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    if weights is not None:
        w = np.asarray(weights)
        return float(np.sum(err * w) / (np.sum(w) + 1e-12))
    return float(np.mean(err))


def _spearman_weight(mae: float, spearman: float) -> float:
    sp = max(spearman, 1e-6)
    return (sp * sp) / (mae + 1e-6)


def _lightgbm_eval_spearman(y_true, y_pred):
    sp = _spearman(y_true, y_pred)
    if not np.isfinite(sp):
        sp = -1.0
    return 'spearman', float(sp), True


def train_ensemble_with_sweep(X_tr, y_tr, X_va, y_va, feats,
                              sample_weight=None, eval_weight=None,
                              fold_tag: str="core"):
    if not lgb:
        raise RuntimeError("lightgbm is required.")
    models = []

    if FAST_MODE:
        lightgbm_grid = [
            dict(
                name="fast64_lr05",
                seeds=[37],
                params=dict(
                    n_estimators=450, learning_rate=0.05, num_leaves=64, max_depth=-1,
                    min_child_samples=60, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.85, feature_fraction=0.8,
                    reg_lambda=0.3, reg_alpha=0.3, min_split_gain=0.01
                )
            ),
        ]
    else:
        lightgbm_grid = [
            dict(
                name="stable32_lr02",
                seeds=[37],
                params=dict(
                    n_estimators=1400, learning_rate=0.02, num_leaves=32, max_depth=-1,
                    min_child_samples=50, subsample=0.85, subsample_freq=2,
                    colsample_bytree=0.9, feature_fraction=0.8,
                    reg_lambda=0.3, reg_alpha=0.3, min_split_gain=0.005
                )
            ),
            dict(
                name="balanced64_lr035",
                seeds=[41],
                params=dict(
                    n_estimators=1100, learning_rate=0.035, num_leaves=64, max_depth=-1,
                    min_child_samples=50, subsample=0.9, subsample_freq=2,
                    colsample_bytree=0.9, feature_fraction=0.82,
                    reg_lambda=0.3, reg_alpha=0.3, min_split_gain=0.005
                )
            ),
            dict(
                name="fast128_lr05",
                seeds=[55],
                params=dict(
                    n_estimators=850, learning_rate=0.05, num_leaves=128, max_depth=-1,
                    min_child_samples=50, subsample=0.85, subsample_freq=1,
                    colsample_bytree=0.85, feature_fraction=0.78,
                    reg_lambda=0.3, reg_alpha=0.3, min_split_gain=0.01
                )
            ),
        ]

    lgb_results = []
    for i, spec in enumerate(lightgbm_grid, start=1):
        base_params = spec['params'].copy()
        reg_lambda = base_params.pop('reg_lambda', 0.05)
        reg_alpha = base_params.pop('reg_alpha', 0.1)
        seeds = spec.get('seeds', [42 + i * 11])
        name = spec.get('name', f"cfg#{i}")
        for seed in seeds:
            cfg = base_params.copy()
            m = lgb.LGBMRegressor(
                objective='mae',
                n_jobs=-1,
                random_state=seed,
                verbose=-1,
                use_missing=False,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                **cfg
            )
            fit_kwargs = dict(
                eval_set=[(X_va, y_va)],
                eval_metric=_lightgbm_eval_spearman,
                callbacks=[lgb.early_stopping(200, verbose=False)]
            )
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight
            if eval_weight is not None:
                fit_kwargs['eval_sample_weight'] = [eval_weight]
            m.fit(X_tr, y_tr, **fit_kwargs)
            pred = m.predict(X_va)
            mae = _weighted_mae(y_va, pred, eval_weight)
            sp = float(_spearman(y_va, pred))
            if not np.isfinite(sp) or sp <= 0:
                log.info(f"  [{fold_tag}] LGBM {name} (seed={seed})  MAE={mae:.6f}  SPEARMAN={sp:.4f} (skipped)")
                continue
            w = _spearman_weight(mae, sp)
            models.append(('lgbm', m, w, mae, sp))
            lgb_results.append((mae, sp, name, seed, cfg))
            best_eval = m.best_score_.get('valid_0', {}).get('spearman')
            if best_eval is not None:
                log.info(f"  [{fold_tag}] LGBM {name} (seed={seed})  MAE={mae:.6f}  SPEARMAN={sp:.4f}  eval@best={best_eval:.4f}  leaves={cfg.get('num_leaves')}")
            else:
                log.info(f"  [{fold_tag}] LGBM {name} (seed={seed})  MAE={mae:.6f}  SPEARMAN={sp:.4f}  leaves={cfg.get('num_leaves')}")

    if lgb_results:
        best_mae, best_sp, best_name, best_seed, best_cfg = max(
            lgb_results,
            key=lambda x: (x[1], -x[0])
        )
        log.info(f"  [{fold_tag}] Best LGBM -> {best_name} (seed={best_seed}) SPEARMAN={best_sp:.4f}, MAE={best_mae:.6f}, params={best_cfg}")

    if xgb:
        xgb_grid = [
            dict(n_estimators=750, learning_rate=0.04, max_depth=6, subsample=0.85, colsample_bytree=0.85),
        ]
        for j, cfg in enumerate(xgb_grid):
            xg = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_jobs=-1,
                random_state=70 + j * 7,
                **cfg
            )
            xg.fit(X_tr, y_tr, sample_weight=sample_weight, eval_set=[(X_va, y_va)], verbose=False)
            pred = xg.predict(X_va)
            mae = _weighted_mae(y_va, pred, eval_weight)
            sp = float(_spearman(y_va, pred))
            if not np.isfinite(sp) or sp <= 0:
                log.info(f"  XGB cfg#{j+1}   MAE={mae:.6f}  SPEARMAN={sp:.4f} (skipped)")
                continue
            w = max(1e-6, sp) / (mae + 1e-6)
            models.append(('xgb', xg, w, mae, sp))
            log.info(f"  XGB cfg#{j+1}   MAE={mae:.6f}  SPEARMAN={sp:.4f} (added)")
    else:
        log.info("  XGBoost not installed; skipping.")

    return models


def predict_ensemble(models, X):
    stack_entry = None
    base_entries = []
    for entry in models:
        if entry[0] == 'stack':
            stack_entry = entry
        else:
            base_entries.append(entry)
    if not base_entries:
        return np.zeros(len(X))
    base_preds = [entry[1].predict(X) for entry in base_entries]
    if stack_entry is not None:
        preds_matrix = np.column_stack(base_preds)
        return stack_entry[1].predict(preds_matrix)
    weights = np.array([entry[2] if entry[2] is not None else 0.0 for entry in base_entries], dtype=float)
    if not np.isfinite(weights).all() or weights.sum() == 0:
        weights = np.ones(len(base_entries))
    ws = weights / weights.sum()
    out = np.zeros(len(X))
    for pred, w in zip(base_preds, ws):
        out += w * pred
    return out


# ----- train/validate diagnostics -------------------------------------------

def train_and_validate_to_diagnostics(
    feat_df,
    feats,
    valid_df,
    build_diagnostics: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple], Optional[Dict[str, Any]]]:
    tr_base = feat_df.dropna(subset=['target']).copy()
    tr_base = tr_base.sort_values(['numerai_ticker','date'])
    tr = tr_base.drop_duplicates(['numerai_ticker','friday_date'], keep='last')
    if len(tr) < 100:
        raise RuntimeError("Insufficient training rows.")
    tr = tr.sort_values(['friday_date','numerai_ticker']).reset_index(drop=True)
    log.info("  [7a] Preparing time-series folds & sample weights...")
    weight_series = None
    if 'sample_weight' in tr.columns:
        log.info("Using provided v2.1 sample_weights for training.")
        weight_series = tr['sample_weight'].fillna(0.0)
        # To combine with era-decay instead of replacing, apply decay here:
        # if ERA_DECAY_HALFLIFE > 0:
        #     eras = pd.to_datetime(tr['friday_date']).reset_index(drop=True)
        #     codes = pd.factorize(eras)[0]
        #     decay = np.power(0.5, (codes.max() - codes) / max(1.0, ERA_DECAY_HALFLIFE))
        #     weight_series = weight_series * decay
    elif ERA_DECAY_HALFLIFE > 0 and 'friday_date' in tr.columns:
        eras = pd.to_datetime(tr['friday_date']).reset_index(drop=True)
        codes = pd.factorize(eras)[0]
        decay = np.power(0.5, (codes.max() - codes) / max(1.0, ERA_DECAY_HALFLIFE))
        weight_series = pd.Series(decay, index=tr.index)
        log.info(f"Applying era-decay weighting (half-life={ERA_DECAY_HALFLIFE} weeks).")
    X_all, y_all = tr[feats], tr['target']
    # purged era-based CV with embargo
    era_list = sorted(tr['friday_date'].unique())
    val_len = max(1, len(era_list) // MODEL_CV_FOLDS)
    fold_pairs = []
    for i in range(MODEL_CV_FOLDS):
        val_start = i * val_len
        val_end = min(len(era_list), val_start + val_len)
        val_eras = era_list[val_start:val_end]
        if not val_eras:
            continue
        train_cut = max(0, val_start - EMBARGO_WEEKS)
        train_eras = era_list[:train_cut]
        tr_idx = tr.index[tr['friday_date'].isin(train_eras)]
        va_idx = tr.index[tr['friday_date'].isin(val_eras)]
        if len(va_idx) == 0 or len(tr_idx) == 0:
            continue
        fold_pairs.append((tr_idx, va_idx))
    if not fold_pairs:
        raise RuntimeError("No valid CV folds constructed; check era coverage.")
    log.info(f"  [7b] Training ensemble models across {len(fold_pairs)} folds (limit={MODEL_CV_FOLDS})...")
    models = []
    for fold_num, (tr_idx, va_idx) in enumerate(fold_pairs, start=1):
        X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
        y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]
        w_tr = weight_series.iloc[tr_idx].values if weight_series is not None else None
        w_va = weight_series.iloc[va_idx].values if weight_series is not None else None
        fold_tag = f"fold{fold_num}"
        log.info(f"    -> Fold {fold_num}/{len(fold_pairs)}: train={len(tr_idx):,} rows, val={len(va_idx):,} rows")
        fold_models = train_ensemble_with_sweep(
            X_tr, y_tr, X_va, y_va, feats,
            sample_weight=w_tr, eval_weight=w_va,
            fold_tag=fold_tag
        )
        if not fold_models:
            continue
        fold_best_sp = max(m[4] for m in fold_models)
        sp_cut = max(MODEL_MIN_SPEARMAN, fold_best_sp * MODEL_FOLD_KEEP_FRAC)
        kept = [m for m in fold_models if m[4] >= sp_cut]
        dropped = len(fold_models) - len(kept)
        if dropped > 0:
            log.info(f"    -> Fold {fold_num}: dropping {dropped} weak models below Spearman {sp_cut:.4f}")
        models.extend(kept)
    if not models:
        raise RuntimeError("No models were trained; check feature/target coverage.")
    if STACK_META_ALPHA > 0 and len(models) >= STACK_META_MIN_MODELS:
        base_models_for_stack = [m for m in models if m[0] != 'stack']
        try:
            preds_matrix = np.column_stack([mdl[1].predict(X_va) for mdl in base_models_for_stack])
            if preds_matrix.shape[1] >= STACK_META_MIN_MODELS:
                ridge = Ridge(alpha=STACK_META_ALPHA, fit_intercept=True)
                ridge.fit(preds_matrix, y_va, sample_weight=w_va)
                meta_pred = ridge.predict(preds_matrix)
                meta_sp = float(_spearman(y_va, meta_pred))
                if np.isfinite(meta_sp) and meta_sp >= STACK_META_MIN_SPEARMAN:
                    meta_mae = _weighted_mae(y_va, meta_pred, w_va)
                    models.append(('stack', ridge, None, meta_mae, meta_sp))
                    log.info(f"  [meta] Ridge blend SPEARMAN={meta_sp:.4f}, MAE={meta_mae:.6f}")
        except Exception as err:
            log.warning(f"  [meta] Ridge blend failed: {err}")

    if not build_diagnostics:
        return pd.DataFrame(), pd.DataFrame(), models, None

    v = valid_df.copy()
    if 'date' in v.columns:
        v.rename(columns={'date':'friday_date'}, inplace=True)
    v['friday_date'] = pd.to_datetime(v['friday_date']).dt.tz_localize(None).dt.normalize()
    v = v[v['friday_date'] >= PIPE_START_TS]
    log.info("  [7c] Merging validation features & metadata...")
    aux_cols = ['numerai_ticker','friday_date','sector','size_bucket']
    if 'volatility_20d' not in feats:
        aux_cols.append('volatility_20d')
    base = feat_df.sort_values(['numerai_ticker','date'])
    base = base[aux_cols + feats].drop_duplicates(['numerai_ticker','friday_date'], keep='last')
    vv = v[['numerai_ticker','friday_date','target']].astype({'numerai_ticker':str})
    valf = vv.merge(base, on=['numerai_ticker','friday_date'], how='inner')
    # Clean NaN/inf per-column to dodge pandas multi-column assignment quirks
    for col in feats:
        if col in valf.columns:
            valf[col] = valf[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    valf = valf.sort_values(['numerai_ticker','friday_date'])
    log.info("  [7d] Predicting validation eras...")
    valf['raw_pred'] = predict_ensemble(models, valf[feats])
    # smooth per ticker
    valf['raw_pred_smooth'] = valf.groupby('numerai_ticker')['raw_pred'].transform(lambda x: x.ewm(span=EMA_SPAN, adjust=False).mean())

    # baseline sector/size-bucket neutralization
    log.info("  [7e] Neutralizing predictions...")
    if NEUTRALIZE_PREDICTIONS:
        valf['raw_pred_base_neu'] = neutralize_features_cross_sectional(valf.rename(columns={'raw_pred_smooth':'tmp'}), ['tmp'])['tmp']
    else:
        valf['raw_pred_base_neu'] = valf['raw_pred_smooth']

    # risk neutralization to size & volatility (partial)
    if RISK_NEUTRALIZE:
        valf['raw_pred_neu'] = neutralize_to_factors(valf, 'raw_pred_base_neu', ['size_proxy','volatility_20d'], keys=['friday_date'], strength=0.0)  # estimate betas
        valf['raw_pred_neu'] = neutralize_to_factors(valf.assign(raw_pred_base_neu=valf['raw_pred_base_neu']), 'raw_pred_base_neu', ['size_proxy'], keys=['friday_date'], strength=SIZE_NEUTRALIZE_STRENGTH)
        valf['raw_pred_neu'] = neutralize_to_factors(valf.assign(raw_pred_base_neu=valf['raw_pred_neu']), 'raw_pred_base_neu', ['volatility_20d'], keys=['friday_date'], strength=VOL_NEUTRALIZE_STRENGTH)
    else:
        valf['raw_pred_neu'] = valf['raw_pred_base_neu']

    def _rank(g):
        return pd.Series(pct_rank_stable(g['raw_pred_neu']), index=g.index)
    valf['signal'] = valf.groupby('friday_date', group_keys=False).apply(_rank)
    valf['signal'] = valf['signal'].clip(1e-6, 1-1e-6)
    def _rank_raw(g):
        return pd.Series(pct_rank_stable(g['raw_pred_smooth']), index=g.index)
    valf['signal_raw'] = valf.groupby('friday_date', group_keys=False).apply(_rank_raw).clip(1e-6, 1-1e-6)

    log.info("  [7f] Finalizing diagnostics frame...")
    try:
        recent_n = min(RECENT_VALIDATION_ERAS, valf['friday_date'].nunique())
        eval_df = valf[['friday_date','numerai_ticker','signal','target']].dropna()
        era_corr_val = _era_spearman(eval_df, 'signal', 'target')
        if not era_corr_val.empty:
            overall = era_corr_val.mean()
            recent = era_corr_val.tail(recent_n).mean()
            log.info(f"  [recent-val] eras={len(era_corr_val)} overall_corr={overall:.4f} | last{recent_n}_corr={recent:.4f}")
            if np.isfinite(overall) and np.isfinite(recent) and recent < overall * 0.5:
                log.warning(f"[recent-val] Recent correlation dropped below 50% of overall ({recent:.4f} vs {overall:.4f}).")
    except Exception as err:
        log.warning(f"[recent-val] Validation health check failed: {err}")
    # uniqueness proxy (log)
    try:
        base_fac = ['momentum_20','volatility_20d','size_proxy']
        tmp = feat_df[['numerai_ticker','friday_date']+base_fac].drop_duplicates(['numerai_ticker','friday_date'])
        cc = valf[['numerai_ticker','friday_date','signal']].merge(tmp, on=['numerai_ticker','friday_date'], how='left')
        uc = float(np.nanmean([abs(stats.spearmanr(cc['signal'], cc[c], nan_policy='omit')[0]) for c in base_fac]))
        log.info(f"  Uniqueness proxy vs (momentum, vol, size): {1-uc:.3f}")
    except Exception:
        pass

    diag = valf[['numerai_ticker','friday_date','signal']].copy()
    diag_raw = valf[['numerai_ticker','friday_date','signal_raw']].copy()
    for frame in (diag, diag_raw):
        frame['friday_date'] = pd.to_datetime(frame['friday_date'])
    eras = sorted(diag['friday_date'].unique())
    if len(eras) > DIAG_LAST_ERAS:
        keep = set(eras[-DIAG_LAST_ERAS:])
        diag = diag[diag['friday_date'].isin(keep)]
        diag_raw = diag_raw[diag_raw['friday_date'].isin(keep)]
    diag = diag.sort_values(['friday_date','numerai_ticker'])
    diag['friday_date'] = diag['friday_date'].dt.strftime('%Y-%m-%d')
    diag_raw = diag_raw.sort_values(['friday_date','numerai_ticker']).rename(columns={'signal_raw':'signal'})
    diag_raw['friday_date'] = diag_raw['friday_date'].dt.strftime('%Y-%m-%d')
    risk_diag = None
    try:
        risk_diag = _risk_tilt_snapshot(valf, label="diag")
    except Exception:
        pass
    return diag, diag_raw, models, risk_diag


def save_and_optional_upload_diagnostics(napi, diag_df, model_id, prefix="diagnostics_submission", full_mode=False):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "full" if full_mode else "k"
    path = f"{prefix}_{ts}_{tag}{DIAG_LAST_ERAS}.csv"
    diag_df.to_csv(path, index=False)
    log.info(f"[diag] Saved: {path}")
    try:
        if napi: 
            napi.upload_diagnostics(path, model_id=model_id)
        log.info("[diag] Upload complete")
    except Exception as e:
        log.warning(f"Diagnostics upload failed: {e}")

    eras = diag_df['friday_date'].unique()
    if len(eras) > DIAG_TARGET_ERAS_COUNT:
        keep = set(eras[-DIAG_TARGET_ERAS_COUNT:])
        trimmed = diag_df[diag_df['friday_date'].isin(keep)].copy()
    else:
        trimmed = diag_df.copy()
    tpath = f"{'diagnostics_submission_last' + ('_full' if full_mode else '')}{DIAG_TARGET_ERAS_COUNT}_{ts}.csv"
    trimmed.to_csv(tpath, index=False)
    log.info(f"  [diag] Trimmed diagnostics saved: {tpath} (eras={trimmed['friday_date'].nunique()}, rows={len(trimmed):,})")
    try:
        if napi and DIAG_UPLOAD_TRIMMED: 
            napi.upload_diagnostics(tpath, model_id=model_id)
        if DIAG_UPLOAD_TRIMMED: 
            log.info("[diag] Trimmed upload complete")
    except Exception as e:
        log.warning(f"Trimmed upload failed: {e}")
    return path, tpath


# ----- payout-focused backtest report -----------------------------------------


def _era_spearman(series_df: pd.DataFrame, signal_col: str, target_col: str) -> pd.Series:
    """Compute per-era Spearman correlation between signal and target."""
    def _corr(group):
        if len(group) < 5:
            return np.nan
        try:
            return stats.spearmanr(group[signal_col], group[target_col], nan_policy='omit')[0]
        except Exception:
            return np.nan
    corr = series_df.groupby('friday_date').apply(_corr).dropna()
    if corr.empty:
        return pd.Series(dtype=float)
    return corr.sort_index()


def _simulate_stake_equity(payout_returns: pd.Series, stake_frac: float, start_cash: float, settlement_lag: int):
    """Simulate Numerai-style staking with delayed settlement."""
    lag = max(0, int(settlement_lag))
    payout_clean = payout_returns.fillna(0.0)
    if payout_clean.empty:
        return pd.Series(dtype=float), float(start_cash)

    equity_vals = []
    bankroll = float(start_cash)
    if lag > 0:
        pending = deque([0.0] * lag)
    else:
        pending = deque()

    for val in payout_clean.values:
        if lag > 0 and pending:
            bankroll += pending.popleft()
        stake_amt = bankroll * stake_frac
        profit = stake_amt * val
        if lag > 0:
            pending.append(profit)
        else:
            bankroll += profit
        equity_vals.append(bankroll)

    if lag > 0:
        while pending:
            bankroll += pending.popleft()

    equity_series = pd.Series(equity_vals, index=payout_clean.index)
    return equity_series, float(bankroll)


def _build_payout_metrics(diag_df: pd.DataFrame, valid_df: pd.DataFrame, weights_df: Optional[pd.DataFrame]=None):
    """Reusable computation of payout-era correlations and derived metrics."""
    pred = diag_df.copy()
    pred['numerai_ticker'] = pred['numerai_ticker'].astype(str)
    pred['friday_date'] = pd.to_datetime(pred['friday_date']).dt.tz_localize(None).dt.normalize()
    v = valid_df.copy()
    if 'date' in v.columns:
        v = v.rename(columns={'date':'friday_date'})
    v['friday_date'] = pd.to_datetime(v['friday_date']).dt.tz_localize(None).dt.normalize()
    v['numerai_ticker'] = v['numerai_ticker'].astype(str)
    keep_cols = ['friday_date','numerai_ticker','target']
    merged = pred.merge(v[keep_cols], on=['friday_date','numerai_ticker'], how='inner').dropna(subset=['signal','target'])
    if weights_df is not None and not weights_df.empty:
        w = weights_df.copy()
        if 'date' in w.columns:
            w['friday_date'] = pd.to_datetime(w['date']).dt.tz_localize(None).dt.normalize()
        else:
            w['friday_date'] = pd.to_datetime(w['friday_date']).dt.tz_localize(None).dt.normalize()
        w['numerai_ticker'] = w['numerai_ticker'].astype(str)
        # weights file uses sample_weights column name
        weight_col = 'sample_weight' if 'sample_weight' in w.columns else 'sample_weights' if 'sample_weights' in w.columns else None
        if weight_col:
            merged = merged.merge(w[['friday_date','numerai_ticker',weight_col]], on=['friday_date','numerai_ticker'], how='left')
            merged.rename(columns={weight_col:'sample_weight'}, inplace=True)
    merged['sample_weight'] = merged.get('sample_weight', 1.0)
    merged['sample_weight'] = merged['sample_weight'].fillna(1.0)
    if merged.empty:
        return None, None, None
    era_corr = _era_spearman(merged, 'signal', 'target')
    if era_corr.empty:
        return None, None, None
    era_weight = merged.groupby('friday_date')['sample_weight'].sum()
    era_weight = era_weight.reindex(era_corr.index).fillna(1.0).clip(lower=1e-9)
    payout_returns = (era_corr / PAYOUT_CORR_DIVISOR).clip(lower=PAYOUT_CLIP_LOW, upper=PAYOUT_CLIP_HIGH)
    stake_frac = min(max(PAYOUT_STAKE_FRACTION, 0.0), 1.0)
    stake_returns = payout_returns * stake_frac
    equity, equity_final_val = _simulate_stake_equity(
        stake_returns,
        stake_frac=stake_frac,
        start_cash=PAYOUT_BACKTEST_START_CASH,
        settlement_lag=PAYOUT_SETTLEMENT_LAG
    )
    w = era_weight.values
    corr_vals = era_corr.values
    payout_vals = payout_returns.values
    stake_vals = stake_returns.values
    w_mean_corr = float((corr_vals * w).sum() / w.sum())
    w_mean_payout = float((payout_vals * w).sum() / w.sum())
    w_mean_stake = float((stake_vals * w).sum() / w.sum())
    w_var_stake = float(((stake_vals - w_mean_stake) ** 2 * w).sum() / w.sum())
    w_std_stake = w_var_stake ** 0.5
    recent_n = min(20, len(era_corr))
    w_recent = era_weight.tail(recent_n).values
    corr_recent_vals = era_corr.tail(recent_n).values
    payout_recent_vals = payout_returns.tail(recent_n).values
    stake_recent_vals = payout_recent_vals * stake_frac
    corr_recent_mean = float((corr_recent_vals * w_recent).sum() / (w_recent.sum() + 1e-9))
    payout_recent_mean = float((payout_recent_vals * w_recent).sum() / (w_recent.sum() + 1e-9))
    stake_recent_mean = float((stake_recent_vals * w_recent).sum() / (w_recent.sum() + 1e-9))
    metrics = {
        "era_count": len(era_corr),
        "corr_overall": w_mean_corr,
        "corr_recent20": corr_recent_mean,
        "corr_roll60_mean": float(era_corr.rolling(60).mean().dropna().iloc[-1]) if len(era_corr) >= 60 else float("nan"),
        "corr_roll60_std": float(era_corr.rolling(60).std().dropna().iloc[-1]) if len(era_corr) >= 60 else float("nan"),
        "corr_best": float(era_corr.max()),
        "corr_worst": float(era_corr.min()),
        "payout_overall": w_mean_payout,
        "payout_recent20": payout_recent_mean,
        "payout_best": float(payout_returns.max()),
        "payout_worst": float(payout_returns.min()),
        "stake_fraction": stake_frac,
        "stake_overall": w_mean_stake,
        "stake_recent20": stake_recent_mean,
        "equity_final": float(equity_final_val),
        "growth_multiple": float(equity_final_val / PAYOUT_BACKTEST_START_CASH) if PAYOUT_BACKTEST_START_CASH != 0 else float("nan"),
        "max_dd": float((equity / equity.cummax() - 1.0).min()) if not equity.empty else float("nan"),
        "sharpe_weekly": float(w_mean_stake / (w_std_stake + 1e-12)) if np.isfinite(w_std_stake) else float("nan"),
    }
    return metrics, era_corr, equity


def _risk_tilt_snapshot(df: pd.DataFrame, label: str):
    """Log sector/size/vol tilt diagnostics for a predictions frame."""
    if df is None or df.empty:
        return {}
    res: Dict[str, Any] = {}
    try:
        beta_size = float(stats.spearmanr(df['signal'], df.get('size_proxy'), nan_policy='omit')[0]) if 'size_proxy' in df.columns else float("nan")
        beta_vol = float(stats.spearmanr(df['signal'], df.get('volatility_20d'), nan_policy='omit')[0]) if 'volatility_20d' in df.columns else float("nan")
        res['beta_size'] = beta_size
        res['beta_vol'] = beta_vol
        sector_means = pd.Series(dtype=float)
        if 'sector' in df.columns:
            sector_means = df.groupby('sector')['signal'].mean().dropna()
            if not sector_means.empty:
                sector_means = sector_means[sector_means.index != 'UNK'] if 'UNK' in sector_means.index else sector_means
                sector_means = sector_means[sector_means.index.notna()]
        res['sector_means'] = sector_means
        log.info(f"[risk-tilt:{label}] beta_size={beta_size:.4f} beta_vol={beta_vol:.4f}")
        if not sector_means.empty:
            dense = sector_means.dropna()
            top_sec = dense.sort_values(ascending=False).head(3)
            bot_sec = dense.sort_values().head(3)
            log.info(f"[risk-tilt:{label}] top sectors {top_sec.to_dict()} | bottom {bot_sec.to_dict()}")
    except Exception as err:
        log.warning(f"[risk-tilt:{label}] tilt calc failed: {err}")
    return res


def generate_payout_backtest_report(diag_df: pd.DataFrame, valid_df: pd.DataFrame, weights_df: Optional[pd.DataFrame]=None, label: str="k") -> Optional[str]:
    """Approximate payout-style backtest (60d) with a fast 20-era health check."""
    if diag_df is None or diag_df.empty or valid_df is None or valid_df.empty:
        log.warning("[backtest] Missing diagnostics or validation data; skipping payout report.")
        return None
    metrics, era_corr, equity = _build_payout_metrics(diag_df, valid_df, weights_df=weights_df)
    if metrics is None or era_corr is None or equity is None:
        log.warning("[backtest] Unable to build payout metrics; skipping text report.")
        return None

    report_lines = [
        "Numerai Signals Payout-Focused Backtest",
        f"Label: {label}",
        f"Eras evaluated: {metrics['era_count']}",
        f"Payout-style corr (full): {metrics['corr_overall']:.5f}",
        f"Health-check corr (last 20 eras): {metrics['corr_recent20']:.5f}",
        f"Rolling60 corr mean: {metrics['corr_roll60_mean']:.5f}",
        f"Rolling60 corr std: {metrics['corr_roll60_std']:.5f}",
        f"Payout mean (clipped): {metrics['payout_overall']:.5f}",
        f"Payout mean (last 20 eras): {metrics['payout_recent20']:.5f}",
        f"Assumed stake fraction (of bankroll): {metrics['stake_fraction']:.0%}",
        f"Settlement lag (eras) applied: {PAYOUT_SETTLEMENT_LAG}",
        f"Final equity (start {PAYOUT_BACKTEST_START_CASH:,.0f}): {metrics['equity_final']:,.2f} (x{metrics['growth_multiple']:.2f})",
        f"Max drawdown: {metrics['max_dd']:.4f}",
        f"Sharpe (weekly proxy, stake-scaled): {metrics['sharpe_weekly']:.3f}",
        f"Best era: {metrics['corr_best']:.4f}",
        f"Worst era: {metrics['corr_worst']:.4f}",
    ]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"payout_backtest_report_{label}_{ts}.txt"
    Path(path).write_text("\n".join(report_lines), encoding="utf-8")
    log.info(f"[backtest] Payout report saved: {path}")
    return path


def generate_payout_backtest_pdf(diag_df: pd.DataFrame, valid_df: pd.DataFrame, weights_df: Optional[pd.DataFrame]=None, label: str="k") -> Optional[str]:
    """PDF version of the payout-style backtest with quick (20-era) health check visuals."""
    if plt is None:
        log.warning("[backtest] matplotlib not available; PDF report skipped.")
        return None
    try:
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as err:
        log.warning(f"[backtest] PdfPages import failed: {err}")
        return None
    if diag_df is None or diag_df.empty or valid_df is None or valid_df.empty:
        log.warning("[backtest] Missing diagnostics or validation data; skipping payout PDF.")
        return None
    metrics, era_corr, equity = _build_payout_metrics(diag_df, valid_df, weights_df=weights_df)
    if metrics is None or era_corr is None or equity is None:
        log.warning("[backtest] Unable to build payout metrics; skipping PDF.")
        return None
    ret_weekly = pd.Series(equity).pct_change().dropna()
    best = metrics["corr_best"]
    worst = metrics["corr_worst"]
    corr_recent = metrics["corr_recent20"]
    roll_mean_60 = era_corr.rolling(60).mean().dropna()
    roll_std_60 = era_corr.rolling(60).std().dropna()
    roll_latest_mean = metrics["corr_roll60_mean"]
    roll_latest_std = metrics["corr_roll60_std"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"payout_backtest_report_{label}_{ts}.pdf"
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(2, 2, figsize=(10, 7))
        ax = ax.flatten()
        # Equity curve
        ax[0].plot(equity.index, equity.values, color="#1565c0", lw=1.6)
        ax[0].set_title("Payout-Style Equity Curve", loc="left", fontweight="bold")
        ax[0].grid(True, ls="--", alpha=0.3)
        # Drawdown
        dd = equity / equity.cummax() - 1.0
        ax[1].fill_between(dd.index, dd.values, color="#ef6c00", alpha=0.5)
        ax[1].set_title("Drawdown", loc="left", fontweight="bold")
        ax[1].grid(True, ls="--", alpha=0.3)
        # Era correlation curve
        ax[2].plot(era_corr.index, era_corr.values, color="#2e7d32", lw=1.4)
        ax[2].axhline(0, color="black", lw=0.8, ls="--")
        ax[2].set_title("Era Spearman (target vs signal)", loc="left", fontweight="bold")
        ax[2].grid(True, ls="--", alpha=0.3)
        # Histogram of era correlations
        ax[3].hist(era_corr.values, bins=30, color="#6a1b9a", alpha=0.8)
        ax[3].set_title("Era Correlation Distribution", loc="left", fontweight="bold")
        ax[3].set_xlabel("Spearman")
        ax[3].set_ylabel("Count")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Metrics page
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.axis("off")
        lines = [
            "Numerai Signals Payout-Focused Backtest",
            f"Label: {label}",
            f"Eras evaluated: {metrics['era_count']}",
            f"Payout-style corr (full): {metrics['corr_overall']:.5f}",
            f"Health-check corr (last {min(20, len(era_corr))} eras): {corr_recent:.5f}",
            f"Rolling60 corr mean: {roll_latest_mean:.5f}",
            f"Rolling60 corr std: {roll_latest_std:.5f}",
            f"Payout mean (clipped): {metrics['payout_overall']:.5f}",
            f"Payout mean (last {min(20, len(era_corr))} eras): {metrics['payout_recent20']:.5f}",
            f"Assumed stake fraction (of bankroll): {metrics['stake_fraction']:.0%}",
            f"Settlement lag (eras) applied: {PAYOUT_SETTLEMENT_LAG}",
            f"Final equity (start {PAYOUT_BACKTEST_START_CASH:,.0f}): {metrics['equity_final']:,.2f} (x{metrics['growth_multiple']:.2f})",
            f"Max drawdown: {metrics['max_dd']:.4f}",
            f"Sharpe (weekly proxy, stake-scaled): {metrics['sharpe_weekly']:.3f}",
            f"Best era: {best:.4f}",
            f"Worst era: {worst:.4f}",
        ]
        ax2.text(0.01, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
        fig2.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)

        if not roll_mean_60.empty:
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(roll_mean_60.index, roll_mean_60.values, color="#1565c0", lw=1.4, label="Rolling60 mean")
            if not roll_std_60.empty:
                upper = roll_mean_60 + roll_std_60.reindex(roll_mean_60.index, fill_value=np.nan)
                lower = roll_mean_60 - roll_std_60.reindex(roll_mean_60.index, fill_value=np.nan)
                ax3.fill_between(roll_mean_60.index, lower, upper, color="#1565c0", alpha=0.2, label="+/- std")
            ax3.axhline(0, color="black", lw=0.8, ls="--")
            ax3.set_title("Rolling 60-Era Correlation (payout-style)", loc="left", fontweight="bold")
            ax3.grid(True, ls="--", alpha=0.3)
            ax3.legend(loc="upper left")
            fig3.tight_layout()
            pdf.savefig(fig3)
            plt.close(fig3)

    log.info(f"[backtest] Payout PDF report saved: {path}")
    return path


# ----- prev predictions cache ------------------------------------------------

def load_prev_preds() -> pd.DataFrame:
    if PREV_PRED_CACHE.exists():
        try: 
            return pd.read_parquet(PREV_PRED_CACHE)
        except Exception: 
            pass
    return pd.DataFrame(columns=['numerai_ticker','signal'])


def save_prev_preds(df: pd.DataFrame):
    if not df.empty:
        df[['numerai_ticker','signal']].to_parquet(PREV_PRED_CACHE, index=False)


def blend_with_prev(current: pd.DataFrame, w_new: float=PREDICTION_BLEND_WEIGHT) -> pd.DataFrame:
    prev = load_prev_preds()
    if prev.empty: 
        return current
    m = current.merge(prev, on='numerai_ticker', how='left', suffixes=('_new','_old'))
    m['signal_old'] = m['signal_old'].fillna(m['signal_new'])
    m['signal'] = w_new*m['signal_new'] + (1-w_new)*m['signal_old']
    m['signal'] = pd.Series(pct_rank_stable(m['signal'])).clip(1e-6,1-1e-6)
    out = current.copy()
    out['signal'] = m['signal'].values
    return out


# ----- live submission -------------------------------------------------------

def create_live_submission(models, feats, price_df, tmap, live_df, hist_social, meta, recent_media=None):
    base, _ = merge_features(price_df, tmap, hist_social, meta, recent_media)
    snap = base.groupby('numerai_ticker').last().reset_index()
    live_universe = set(live_df['numerai_ticker'].astype(str))
    mapped_universe = set(tmap['numerai_ticker'].astype(str))
    universe = mapped_universe & live_universe if live_universe else mapped_universe
    if not universe:
        universe = live_universe
    snap = snap[snap['numerai_ticker'].astype(str).isin(universe)]
    X = snap[feats].fillna(0).replace([np.inf,-np.inf],0)
    snap['raw_pred'] = predict_ensemble(models, X)

    # base neutralize (sector/size bucket)
    if NEUTRALIZE_PREDICTIONS:
        snap['raw_pred_base_neu'] = neutralize_features_cross_sectional(snap.rename(columns={'raw_pred':'tmp'}), ['tmp'])['tmp']
    else:
        snap['raw_pred_base_neu'] = snap['raw_pred']

    # risk neutralize (size + volatility)
    if RISK_NEUTRALIZE:
        snap['raw_pred_neu'] = neutralize_to_factors(snap.assign(friday_date=snap['friday_date']), 'raw_pred_base_neu', ['size_proxy'], keys=['friday_date'], strength=SIZE_NEUTRALIZE_STRENGTH)
        snap['raw_pred_neu'] = neutralize_to_factors(snap.assign(raw_pred_base_neu=snap['raw_pred_neu'], friday_date=snap['friday_date']), 'raw_pred_base_neu', ['volatility_20d'], keys=['friday_date'], strength=VOL_NEUTRALIZE_STRENGTH)
    else:
        snap['raw_pred_neu'] = snap['raw_pred_base_neu']

    snap['signal'] = pd.Series(pct_rank_stable(snap['raw_pred_neu'])).clip(1e-6,1-1e-6)
    sub = snap[['numerai_ticker','signal']].copy()
    sub = blend_with_prev(sub, w_new=PREDICTION_BLEND_WEIGHT)

    # backfill missing tickers with previous signal (or neutral 0.5) to keep universe stable and lower churn
    prev = load_prev_preds()
    prev_map = {}
    prev_tickers = set()
    if not prev.empty:
        prev_map = dict(zip(prev['numerai_ticker'].astype(str), prev['signal']))
        prev_tickers = set(prev['numerai_ticker'].astype(str))
    expected = (universe or set(sub['numerai_ticker'].astype(str))) | prev_tickers
    expected_df = pd.DataFrame({'numerai_ticker': sorted(expected)})
    combined = expected_df.merge(sub, on='numerai_ticker', how='left')
    missing = [t for t in expected if t not in set(sub['numerai_ticker'].astype(str))]
    prev_hits = sum(1 for t in missing if t in prev_map)
    neutral_fills = len(missing) - prev_hits
    if missing:
        log.info(f"[live-fill] Backfilled {len(missing)} missing tickers (prev={prev_hits}, neutral={neutral_fills}).")
    def _fill(row):
        sig = row['signal']
        if pd.notna(sig):
            return sig
        return prev_map.get(str(row['numerai_ticker']), 0.5)
    combined['signal'] = combined.apply(_fill, axis=1)
    combined['signal'] = pd.Series(pct_rank_stable(combined['signal'])).clip(1e-6,1-1e-6)
    save_prev_preds(combined)
    return combined


# ----- orchestrator ----------------------------------------------------------

def run_all(args):
    # stage progress headline
    print("\n" + "="*80)
    print("NUMERAI SIGNALS v7.5 - ALL-IN-ONE")
    print("="*80)

    napi, reddit, newsapi, pytrends, analyzer = setup_apis()
    models_map = napi.get_models()
    model_name = DEFAULTS["NUMERAI_MODEL_NAME"]
    if model_name not in models_map:
        raise RuntimeError(f"Model '{model_name}' not found in your Numerai account.")
    model_id = models_map[model_name]

    latest_era = None
    auto_mode = bool(getattr(args, "auto", False))
    force_submit = bool(getattr(args, "force_submit", False))
    auto_live = bool(getattr(args, "auto_live", False))
    auto_mode = auto_mode or auto_live
    live_only = bool(getattr(args, "live_only", False)) or auto_live
    if auto_mode and not force_submit:
        try:
            live_probe = download_live_data(napi)
            latest_era = get_latest_live_era(live_probe)
            if latest_era and already_submitted_for_era(model_name, latest_era):
                log.info(f"[auto] Already submitted for era {latest_era} (model={model_name}). Skipping.")
                return
            if latest_era:
                log.info(f"[auto] Latest live era detected: {latest_era}")
        except Exception as err:
            log.warning(f"[auto] Live-era check failed: {err}. Proceeding with full run.")

    train, valid, live, t_weights, v_weights = download_numerai_data(napi)
    if latest_era is None:
        latest_era = get_latest_live_era(live)
    tmap = create_ticker_map(live)
    meta = add_sector_mcap_meta(tmap)
    hist_social = pd.DataFrame()
    if ENABLE_SOCIAL_FEATURES:
        hist_social = get_historical_social_signals_cached(
            tmap,
            pytrends,
            meta,
            refresh_social=getattr(args, "refresh_social_cache", False)
        )
    else:
        log.info("[5/10] Skipping historical social signals (disabled).")

    tmap = tmap.copy()
    if ENABLE_SOCIAL_FEATURES and not hist_social.empty:
        cover = hist_social.groupby('numerai_ticker').size().sort_values(ascending=False)
        top = set(cover.head(SOCIAL_MEDIA_TICKER_COUNT).index)
        tmap['has_social'] = tmap['numerai_ticker'].isin(top)
    else:
        tmap['has_social'] = False
    if SOCIAL_ONLY_UNIVERSE and ENABLE_SOCIAL_FEATURES and tmap['has_social'].any():
        aligned = tmap[tmap['has_social']].drop_duplicates('yahoo_ticker')
        log.info(f"  Universe aligned to social coverage: {len(aligned)} tickers (max_k={SOCIAL_MEDIA_TICKER_COUNT}).")
    else:
        aligned = tmap.drop_duplicates('yahoo_ticker')
        log.info(f"  Universe set intact: {len(aligned)} tickers, {int(tmap['has_social'].sum())} with dense social coverage (top_k={SOCIAL_MEDIA_TICKER_COUNT}).")

    price = get_price_data_cached(aligned)

    recent_media = pd.DataFrame()  # kept minimal

    feat_df, feats = create_training_frame(price, aligned, hist_social, meta, train, t_weights, recent_media)
    log.info("[7/10] Training ensemble & building diagnostics (this can take several minutes)...")
    diag, diag_raw, models, diag_risk = train_and_validate_to_diagnostics(
        feat_df,
        feats,
        valid,
        build_diagnostics=not live_only
    )

    if live_only:
        log.info("[mode] Live-only enabled: skipping diagnostics/backtests/metrics uploads.")
    else:
        diag_path, diag_trimmed = save_and_optional_upload_diagnostics(
            napi,
            diag,
            model_id,
            prefix="diagnostics_submission",
            full_mode=False
        )
        raw_diag_path = None
        if diag_raw is not None and not diag_raw.empty:
            raw_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_diag_path = f"diagnostics_raw_{raw_ts}.csv"
            diag_raw.to_csv(raw_diag_path, index=False)
            log.info(f"[diag] Raw diagnostics saved: {raw_diag_path}")
        try:
            eras_cov = diag.groupby('friday_date')['numerai_ticker'].nunique()
            log.info(f"[diag-cover] eras={len(eras_cov)} | min_tickers={eras_cov.min() if not eras_cov.empty else 0} | max_tickers={eras_cov.max() if not eras_cov.empty else 0}")
            if not eras_cov.empty and eras_cov.min() < DIAG_MIN_TICKERS_PER_ERA:
                log.warning(f"[diag-cover] Some eras below DIAG_MIN_TICKERS_PER_ERA ({DIAG_MIN_TICKERS_PER_ERA}).")
        except Exception as err:
            log.warning(f"[diag-cover] Coverage check failed: {err}")

        try:
            generate_payout_backtest_report(diag, valid, weights_df=v_weights, label="k")
        except Exception as err:
            log.warning(f"[backtest] Payout report generation failed: {err}")
        try:
            generate_payout_backtest_pdf(diag, valid, weights_df=v_weights, label="k")
        except Exception as err:
            log.warning(f"[backtest] Payout PDF generation failed: {err}")
        try:
            metrics, _, _ = _build_payout_metrics(diag, valid, weights_df=v_weights)
            if metrics:
                pct_live = None
                try:
                    pct_live = live['numerai_ticker'].nunique()
                except Exception:
                    pct_live = None
                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **metrics,
                    "diag_eras": diag['friday_date'].nunique(),
                    "diag_min_tickers": diag.groupby('friday_date')['numerai_ticker'].nunique().min(),
                    "diag_max_tickers": diag.groupby('friday_date')['numerai_ticker'].nunique().max(),
                    "live_universe": pct_live,
                }
                header = list(row.keys())
                write_header = not PAYOUT_METRICS_LOG.exists()
                PAYOUT_METRICS_LOG.parent.mkdir(parents=True, exist_ok=True)
                with PAYOUT_METRICS_LOG.open("a", encoding="utf-8") as f:
                    if write_header:
                        f.write(",".join(header) + "\n")
                    f.write(",".join(str(row[k]) for k in header) + "\n")
                log.info(f"[metrics-log] Saved payout/coverage metrics -> {PAYOUT_METRICS_LOG}")
        except Exception as err:
            log.warning(f"[metrics-log] Failed to write metrics: {err}")
        try:
            if diag_risk:
                log.info(f"[risk-tilt:diag] beta_size={diag_risk.get('beta_size')} beta_vol={diag_risk.get('beta_vol')}")
        except Exception:
            pass

        if DIAG_FULL_UNIVERSE and not args.full_diag_off:
            log.info("[7b/10] Building FULL-universe diagnostics...")
            price_full = get_price_data_cached(tmap)
            base_all, _ = merge_features(price_full, tmap, hist_social, meta, recent_media)
            base_all = base_all.sort_values(['numerai_ticker','date']).drop_duplicates(['numerai_ticker','friday_date'], keep='last')
            v = valid[['numerai_ticker','date']].copy()
            v['friday_date'] = pd.to_datetime(v['date']).dt.tz_localize(None).dt.normalize()
            vv = v[['numerai_ticker','friday_date']].astype({'numerai_ticker':str}).drop_duplicates()
            meta_map = tmap[['numerai_ticker','yahoo_ticker']].merge(
                meta[['yahoo_ticker','sector']], on='yahoo_ticker', how='left'
            )[['numerai_ticker','sector']].drop_duplicates()
            vv = vv.merge(meta_map, on='numerai_ticker', how='left')
            vv['sector'] = vv['sector'].fillna('UNK')

            aux_cols = ['numerai_ticker','friday_date','sector','size_bucket']
            if 'volatility_20d' not in feats:
                aux_cols.append('volatility_20d')
            feat_all = base_all[aux_cols + feats]
            valf = vv.merge(feat_all, on=['numerai_ticker','friday_date'], how='left')
            mask = valf[feats].notna().any(axis=1)
            preds = pd.Series(index=valf.index, dtype=float)
            if mask.any():
                X = valf.loc[mask, feats].fillna(0).replace([np.inf,-np.inf],0)
                preds.loc[mask] = predict_ensemble(models, X)
            valf['raw_pred'] = preds.values

            def _fill_era(g):
                if g['raw_pred'].notna().sum()==0:
                    g['raw_pred'] = 0.0; return g
                med = g['raw_pred'].median()
                jitter = g['numerai_ticker'].apply(lambda z: (hash(z)%997)/1e6 - 5e-7)
                g['raw_pred'] = g['raw_pred'].fillna(med) + jitter
                return g
            valf = valf.groupby('friday_date', group_keys=False).apply(_fill_era)

            # neutralize: sector/size-bucket then risk (size+vol)
            valf['raw_pred_base_neu'] = neutralize_features_cross_sectional(valf.rename(columns={'raw_pred':'tmp'}), ['tmp'])['tmp']
            if RISK_NEUTRALIZE:
                valf['raw_pred_neu'] = neutralize_to_factors(valf.assign(tmp=valf['raw_pred_base_neu']), 'tmp', ['size_proxy'], keys=['friday_date'], strength=SIZE_NEUTRALIZE_STRENGTH)
                valf['raw_pred_neu'] = neutralize_to_factors(valf.assign(tmp=valf['raw_pred_neu']), 'tmp', ['volatility_20d'], keys=['friday_date'], strength=VOL_NEUTRALIZE_STRENGTH)
            else:
                valf['raw_pred_neu'] = valf['raw_pred_base_neu']

            def _rank(g): return pd.Series(pct_rank_stable(g['raw_pred_neu']), index=g.index)
            valf['signal'] = valf.groupby('friday_date', group_keys=False).apply(_rank).clip(1e-6,1-1e-6)
            diag_full = valf[['numerai_ticker','friday_date','signal']].copy()
            diag_full['friday_date'] = pd.to_datetime(diag_full['friday_date']).dt.strftime('%Y-%m-%d')
            cover = diag_full.groupby('friday_date')['numerai_ticker'].nunique()
            ok_eras = cover[cover >= DIAG_MIN_TICKERS_PER_ERA].index
            diag_full = diag_full[diag_full['friday_date'].isin(ok_eras)]
            save_and_optional_upload_diagnostics(
                napi,
                diag_full,
                model_id,
                prefix="diagnostics_submission",
                full_mode=True
            )

    if args.no_upload:
        log.info("[9/10] Skipping live submission (--no-upload).")
        return

    log.info("[9/10] Creating live submission...")
    sub = create_live_submission(models, feats, price, aligned, live, hist_social, meta, recent_media)
    try:
        live_cov = sub['numerai_ticker'].nunique()
        expected = aligned['numerai_ticker'].nunique()
        log.info(f"[live-cover] prediction tickers={live_cov:,} / mapped universe={expected:,} ({(live_cov/max(expected,1)):.1%})")
        if live_cov < expected * 0.9:
            log.warning("[live-cover] Coverage below 90% of mapped universe.")
        std_val = sub['signal'].std()
        uniq = sub['signal'].nunique()
        log.info(f"[live-health] std={std_val:.6f} | unique={uniq:,}")
        if std_val < 1e-4 or std_val > 0.35:
            log.warning("[live-health] Signal std outside expected range (low dispersion or too wide).")
        try:
            live_base, _ = merge_features(price, aligned, hist_social, meta, recent_media)
            snap_live = live_base.groupby('numerai_ticker').last().reset_index()
            live_merge = sub.merge(snap_live[['numerai_ticker','sector','size_proxy','volatility_20d']], on='numerai_ticker', how='left')
            _risk_tilt_snapshot(live_merge, label="live")
        except Exception as tilt_err:
            log.warning(f"[risk-tilt:live] tilt calc failed: {tilt_err}")
    except Exception as err:
        log.warning(f"[live-cover] Coverage check failed: {err}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    live_path = f"live_submission_{ts}.csv"
    sub.to_csv(live_path, index=False)
    log.info(f"[live] Saved: {live_path} | Std={sub['signal'].std():.6f} | Unique={sub['signal'].nunique():,}")
    upload_ok = False
    try:
        napi.upload_predictions(live_path, model_id=model_id)
        upload_ok = True
        log.info("[live] Upload successful")
    except Exception as e:
        log.warning(f"Live upload failed: {e}")
    if upload_ok and latest_era:
        try:
            update_submission_state(model_name, model_id, latest_era, live_path)
            log.info(f"[auto] Submission state updated for era {latest_era}.")
        except Exception as err:
            log.warning(f"[auto] Failed to update submission state: {err}")


def prep_cache(args):
    """Download datasets + price/meta/social caches, then exit."""
    print("\n" + "="*80)
    print("NUMERAI SIGNALS v7.5 - CACHE PREP")
    print("="*80)
    napi, reddit, newsapi, pytrends, analyzer = setup_apis()
    train, valid, live, t_weights, v_weights = download_numerai_data(napi)
    tmap = create_ticker_map(live)
    meta = add_sector_mcap_meta(tmap)
    if ENABLE_SOCIAL_FEATURES:
        _ = get_historical_social_signals_cached(
            tmap,
            pytrends,
            meta,
            refresh_social=getattr(args, "refresh_social_cache", False)
        )
    else:
        log.info("[prep] Skipping historical social signals (disabled).")
    _ = get_price_data_cached(tmap)
    log.info("[prep] Cache warm-up complete.")


# ----- CLI -------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Numerai Signals All-in-One v7.5")
    ap.add_argument("--run-all", action="store_true", help="Run diagnostics (k & optional FULL) + live upload.")
    ap.add_argument("--auto", action="store_true", help="Auto-run only when new live era is available.")
    ap.add_argument("--force-submit", action="store_true", help="Force run/upload even if already submitted.")
    ap.add_argument("--auto-live", action="store_true", help="Auto-run on new era and only submit live (skip diagnostics).")
    ap.add_argument("--live-only", action="store_true", help="Skip diagnostics/backtests and only submit live.")
    ap.add_argument("--prep-cache", action="store_true", help="Download datasets + caches, then exit.")
    ap.add_argument("--refresh-social-cache", action="store_true", help="Force social-data refresh (slow).")
    ap.add_argument("--no-upload", action="store_true", help="Do not upload to Numerai; only save files.")
    ap.add_argument("--full-diag-off", action="store_true", help="Disable FULL diagnostics build.")
    ap.add_argument("--model-name", type=str, help="Explicitly specify the Numerai model name (overrides env var).")
    return ap.parse_args()


def main():
    args = parse_args()
    
    # [Override] Model Name from flag
    if args.model_name:
        DEFAULTS["NUMERAI_MODEL_NAME"] = args.model_name.strip()
        log.info(f"[main] Model name overridden via CLI: {DEFAULTS['NUMERAI_MODEL_NAME']}")

    heartbeat_stop = None
    enable_heartbeat = bool(int(os.getenv("LOG_HEARTBEAT", "0"))) or os.getenv("GITHUB_ACTIONS", "").lower() == "true"
    if enable_heartbeat:
        heartbeat_stop = _start_heartbeat(interval_sec=120)
    try:
        if args.prep_cache:
            prep_cache(args)
        elif args.run_all or args.auto or args.auto_live or args.live_only:
            run_all(args)
        else:
            print("Select a mode: --run-all, --auto, --auto-live, --live-only, or --prep-cache")
    except Exception as e:
        log.exception("Unhandled exception")
        print(f"\n[Error] {e}")
        traceback.print_exc()
    finally:
        if heartbeat_stop is not None:
            heartbeat_stop.set()


if __name__ == "__main__":
    main()




