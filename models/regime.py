"""
SENTINEL — Hidden Markov Model Regime Detection.

Uses a Gaussian HMM trained on multiple market observables to mathematically
detect which regime the market is in RIGHT NOW, with probabilities.

Observables fed to the HMM:
  1. SPY daily returns (5-day smoothed)
  2. SPY realized volatility (20-day)
  3. VIX level (scaled)
  4. Bond trend (TLT 20-day return — yield curve proxy)
  5. Credit spread proxy (HYG vs LQD relative performance)

The HMM discovers 3 hidden states from the data. We label them by their
characteristics: highest-return state = BULL, highest-vol state = BEAR,
the remaining one = TRANSITION.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from data.stocks import fetch_ohlcv
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path(__file__).parent / "saved"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "regime_hmm.pkl"


class RegimeDetector:
    """HMM regime detector with rich multi-observable input."""

    def __init__(self, n_states=3, n_iter=200):
        self.n_states = n_states
        self.n_iter = n_iter
        self.model = None
        self._labels = {}
        self._scaler = StandardScaler()
        self._trained_at = None

    # ── Feature engineering ───────────────────────────────────

    def _build_obs(self, spy, vix=None, tlt=None, hyg=None, lqd=None):
        """Build observation matrix from raw OHLCV DataFrames."""
        obs = pd.DataFrame(index=spy.index)

        # 1. Smoothed returns
        obs["ret"] = spy["Close"].pct_change().rolling(5).mean()

        # 2. Realized vol (20d, annualized)
        obs["vol"] = spy["Close"].pct_change().rolling(20).std() * np.sqrt(252)

        # 3. VIX (scaled)
        if vix is not None and not vix.empty:
            obs["vix"] = vix["Close"].reindex(spy.index, method="ffill") / 100
        else:
            obs["vix"] = obs["vol"]

        # 4. Bond trend
        if tlt is not None and not tlt.empty:
            obs["bond"] = tlt["Close"].reindex(spy.index, method="ffill").pct_change(20)
        else:
            obs["bond"] = 0.0

        # 5. Credit spread proxy
        if hyg is not None and lqd is not None and not hyg.empty and not lqd.empty:
            h = hyg["Close"].reindex(spy.index, method="ffill").pct_change(20)
            l = lqd["Close"].reindex(spy.index, method="ffill").pct_change(20)
            obs["credit"] = h - l
        else:
            obs["credit"] = 0.0

        obs = obs.dropna()
        return obs

    def _label_states(self):
        """Label discovered states by mean return and volatility."""
        means = self.model.means_
        ret_col, vol_col = 0, 1

        ret_means = means[:, ret_col]
        vol_means = means[:, vol_col]

        bull_state = int(np.argmax(ret_means))
        bear_state = int(np.argmax(vol_means))
        if bull_state == bear_state:
            # Break tie: bear is highest vol among non-bull
            candidates = [i for i in range(self.n_states) if i != bull_state]
            bear_state = int(max(candidates, key=lambda i: vol_means[i]))

        remaining = [i for i in range(self.n_states) if i not in (bull_state, bear_state)]
        trans_state = remaining[0] if remaining else bear_state

        self._labels = {
            bull_state: {
                "name": "BULL", "color": "#00C853",
                "description": "Low volatility, positive returns. Trend is up with confidence.",
                "avg_ret": float(ret_means[bull_state]),
                "avg_vol": float(vol_means[bull_state]),
            },
            bear_state: {
                "name": "BEAR", "color": "#FF1744",
                "description": "High volatility, negative or flat returns. Fear dominates.",
                "avg_ret": float(ret_means[bear_state]),
                "avg_vol": float(vol_means[bear_state]),
            },
            trans_state: {
                "name": "TRANSITION", "color": "#FFD600",
                "description": "Mixed signals, elevated uncertainty. Market deciding direction.",
                "avg_ret": float(ret_means[trans_state]),
                "avg_vol": float(vol_means[trans_state]),
            },
        }

    # ── Training ──────────────────────────────────────────────

    def fit(self, spy, vix=None, tlt=None, hyg=None, lqd=None, seeds=5):
        """Train HMM with multiple random seeds, keep best.
        Features are StandardScaler-normalized so no single input dominates."""
        obs_df = self._build_obs(spy, vix, tlt, hyg, lqd)
        if len(obs_df) < 100:
            raise ValueError(f"Only {len(obs_df)} observations — need at least 100")

        # Scale features so all inputs have equal influence on the HMM
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(obs_df.values)
        best_model, best_score = None, -np.inf

        for seed in range(seeds):
            try:
                m = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="full",
                    n_iter=self.n_iter,
                    random_state=seed * 42,
                    verbose=False,
                )
                m.fit(X)
                sc = m.score(X)
                if sc > best_score:
                    best_score, best_model = sc, m
            except Exception as e:
                logger.warning(f"Seed {seed} failed: {e}")

        if best_model is None:
            raise RuntimeError("All HMM training attempts failed")

        self.model = best_model
        self._label_states()
        self._obs_df = obs_df
        self._score = best_score
        self._trained_at = datetime.now()
        logger.info(f"Trained HMM: {len(obs_df)} obs, score={best_score:.1f}")
        return self

    def predict(self, obs_df=None):
        """Return full state sequence and probabilities."""
        if self.model is None:
            raise RuntimeError("Call fit() first")
        if obs_df is None:
            obs_df = self._obs_df
        X = self._scaler.transform(obs_df.values)
        states = self.model.predict(X)
        probs = self.model.predict_proba(X)
        return states, probs, obs_df.index

    # ── Current regime query ─────────────────────────────────

    def current_regime(self, close_or_df, vix=None):
        """Return current regime dict from already-trained model.

        Accepts either a Close price Series or a full OHLCV DataFrame.
        If the model is already trained, uses stored observations for
        prediction (avoids re-building features from scratch).
        """
        if self.model is None:
            return {
                "regime": "UNKNOWN", "confidence": 0,
                "probabilities": {}, "color": "#888",
                "description": "Model not trained",
            }

        states, probs, idx = self.predict()
        probs = _temper_probabilities(probs, temperature=2.0, floor=0.03)

        current_state = states[-1]
        label = self._labels.get(current_state, {
            "name": "UNKNOWN", "color": "#888",
            "description": "Unknown state",
        })
        current_probs = probs[-1]

        prob_dict = {}
        for snum, info in self._labels.items():
            prob_dict[info["name"]] = {
                "probability": float(current_probs[snum]) * 100,
                "color": info["color"],
            }

        # Streak
        streak = 0
        for s in reversed(states):
            if s == current_state:
                streak += 1
            else:
                break

        conf = float(current_probs[current_state]) * 100

        return {
            "regime": label["name"],
            "confidence": conf,
            "description": label["description"],
            "color": label["color"],
            "probabilities": prob_dict,
            "streak_days": streak,
        }

    # ── Persistence ───────────────────────────────────────────

    def save(self, path=MODEL_PATH):
        joblib.dump({
            "model": self.model, "labels": self._labels,
            "scaler": self._scaler, "obs_df": self._obs_df,
            "score": self._score, "trained_at": self._trained_at,
        }, path)
        logger.info(f"Saved to {path}")

    def load(self, path=MODEL_PATH):
        data = joblib.load(path)
        self.model = data["model"]
        self._labels = data["labels"]
        self._scaler = data.get("scaler", StandardScaler())
        self._obs_df = data["obs_df"]
        self._score = data["score"]
        self._trained_at = data.get("trained_at")
        # If loaded model was saved without scaler, fit scaler on stored obs
        if not hasattr(self._scaler, 'mean_') or self._scaler.mean_ is None:
            self._scaler = StandardScaler()
            self._scaler.fit(self._obs_df.values)
        return self


# ── High-level API ────────────────────────────────────────────

def _model_is_stale(det, max_age_days=7):
    """Check if saved model is too old and needs retraining."""
    if det._trained_at is None:
        return True  # No timestamp = old format, retrain
    age = datetime.now() - det._trained_at
    if age > timedelta(days=max_age_days):
        logger.info(f"Regime model is {age.days} days old (limit: {max_age_days}) — retraining")
        return True
    return False


def train_regime_model(force=False):
    """Train or load cached HMM. Auto-retrains if model is > 7 days old."""
    det = RegimeDetector()

    if not force and MODEL_PATH.exists():
        try:
            det.load()
            if not _model_is_stale(det):
                logger.info("Loaded cached regime model")
                return det
            logger.info("Cached model is stale, retraining")
        except Exception:
            logger.warning("Cache corrupt, retraining")

    spy = fetch_ohlcv("SPY")
    vix = fetch_ohlcv("^VIX")
    tlt = fetch_ohlcv("TLT")
    hyg = fetch_ohlcv("HYG")
    lqd = fetch_ohlcv("LQD")

    if spy.empty:
        raise RuntimeError("No SPY data")

    det.fit(spy, vix=vix, tlt=tlt, hyg=hyg, lqd=lqd)
    det.save()
    return det


def _temper_probabilities(probs, temperature=2.0, floor=0.03):
    """
    Soften HMM probabilities so they never snap to 100%/0%.

    Raw HMM posteriors are often 99.9%+ for the dominant regime which makes
    the probability charts show a solid wall of one color.  This applies:
      1. Temperature scaling (raises entropy, spreads mass)
      2. A minimum floor (every regime always shows at least floor %)
      3. Renormalization so rows still sum to 1.0

    Result: dominant regime might show 75-85% instead of 99.9%, and the
    alternative regimes stay visible on charts.
    """
    # Temperature: take log, divide by T, re-exponentiate
    eps = 1e-12
    log_p = np.log(probs + eps)
    scaled = np.exp(log_p / temperature)
    row_sums = scaled.sum(axis=1, keepdims=True)
    tempered = scaled / row_sums

    # Floor: ensure every state gets at least floor probability
    n = probs.shape[1]
    tempered = np.maximum(tempered, floor)
    row_sums = tempered.sum(axis=1, keepdims=True)
    tempered = tempered / row_sums

    return tempered


def detect_current_regime():
    """
    Full regime detection result for dashboard consumption.
    Returns dict with regime, confidence, probabilities, history, explanation.
    """
    try:
        det = train_regime_model()
    except Exception as e:
        return {
            "regime": "UNKNOWN", "confidence": 0,
            "probabilities": {}, "history": pd.DataFrame(),
            "explanation": f"Model error: {e}",
            "color": "#888",
        }

    states, probs, idx = det.predict()
    labels = det._labels

    # Temper raw HMM probabilities — they snap to 99.9% too easily which
    # makes the charts useless.  Apply temperature scaling + floor so the
    # non-dominant regimes always show a realistic minimum probability.
    probs = _temper_probabilities(probs, temperature=2.0, floor=0.03)

    current_state = states[-1]
    current_label = labels[current_state]
    current_probs = probs[-1]

    # Probability dict
    prob_dict = {}
    for snum, info in labels.items():
        prob_dict[info["name"]] = {
            "probability": float(current_probs[snum]) * 100,
            "color": info["color"],
            "avg_ret": info["avg_ret"],
            "avg_vol": info["avg_vol"],
        }

    # History DataFrame
    history = pd.DataFrame(index=idx)
    history["state"] = states
    history["regime"] = [labels[s]["name"] for s in states]
    for snum, info in labels.items():
        history[f"prob_{info['name']}"] = probs[:, snum]

    # Current streak
    streak = 0
    for s in reversed(states):
        if s == current_state:
            streak += 1
        else:
            break

    # Regime durations
    durations = {}
    for snum, info in labels.items():
        runs, count = [], 0
        for s in states:
            if s == snum:
                count += 1
            elif count > 0:
                runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)
        if runs:
            durations[info["name"]] = {
                "avg_days": float(np.mean(runs)),
                "max_days": int(max(runs)),
                "occurrences": len(runs),
            }

    # Transition matrix
    trans = det.model.transmat_
    trans_dict = {}
    for i, fi in labels.items():
        trans_dict[fi["name"]] = {}
        for j, fj in labels.items():
            trans_dict[fi["name"]][fj["name"]] = float(trans[i, j]) * 100

    conf = float(current_probs[current_state]) * 100

    return {
        "regime": current_label["name"],
        "confidence": conf,
        "description": current_label["description"],
        "color": current_label["color"],
        "probabilities": prob_dict,
        "history": history,
        "transition_matrix": trans_dict,
        "streak_days": streak,
        "durations": durations,
        "n_observations": len(states),
        "explanation": _explain(current_label, conf, streak, durations, trans_dict),
    }


def _explain(label, conf, streak, durations, trans):
    """Plain-English regime explanation."""
    name = label["name"]
    parts = [
        f"The HMM detects a {name} regime with {conf:.0f}% confidence.",
        label["description"],
        f"We have been in this regime for {streak} trading days.",
    ]
    if name in durations:
        avg = durations[name]["avg_days"]
        parts.append(f"{name} regimes last an average of {avg:.0f} days historically.")
    if name in trans:
        for target, prob in trans[name].items():
            if target != name and prob > 8:
                parts.append(f"{prob:.0f}% daily chance of shifting to {target}.")
    return " ".join(parts)


# ── Backward-compat aliases ──────────────────────────────────

def build_equity_regime_model():
    return train_regime_model(force=True)

def build_crypto_regime_model():
    from data.crypto import fetch_crypto_ohlcv
    btc = fetch_crypto_ohlcv("bitcoin")
    if btc.empty:
        raise RuntimeError("No BTC data")
    det = RegimeDetector()
    det.fit(btc)
    det.save(MODEL_DIR / "crypto_regime_hmm.pkl")
    return det
