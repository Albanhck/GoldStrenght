from __future__ import annotations
import time
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
import numpy as np
import requests

# ----------------------------
# CONFIGURATION
# ----------------------------
API_KEY = "vlno9xSDPNRVEARxO1WEZptEW2lKYU5I"  # ta clé Massive
XAU_TICKER = "C:XAUUSD"
UUP_VARIANTS: List[str] = ["UUP", "C:UUP", "U:UUP"]  # variantes possibles
TIMEFRAME_MINUTES = 15
DAYS_LOOKBACK = 14  # nombre de jours pour récupérer l'historique
MIN_DATA_POINTS = 100  # seuil minimum pour corrélation
PRINT_JSON_RESPONSES = False  # True pour debug


# ----------------------------
# FONCTIONS UTILITAIRES
# ----------------------------
def build_massive_url(ticker: str, start_date: date, end_date: date, agg_minutes: int = 5) -> str:
    """Construit l'URL Massive API pour récupérer les bougies d'agrégat M5."""
    base = "https://api.massive.com/v2/aggs/ticker"
    return (
        f"{base}/{ticker}/range/{agg_minutes}/minute/{start_date}/{end_date}"
        f"?adjusted=true&sort=asc&limit=1000000&apiKey={API_KEY}"
    )


def fetch_massive_agg_minute(ticker: str, start_date: date, end_date: date, agg_minutes: int = 5) -> Optional[pd.DataFrame]:
    """Récupère des bougies d'agrégat M5 depuis Massive API."""
    url = build_massive_url(ticker, start_date, end_date, agg_minutes)
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if PRINT_JSON_RESPONSES:
            import json
            print(json.dumps(data, indent=2))

        candles = data.get("results") or []
        if not candles:
            print(f"⚠️ Aucune 'results' pour {ticker}.")
            return None

        df = pd.DataFrame(candles)
        df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.set_index("t")
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
        return df
    except requests.exceptions.RequestException as exc:
        print(f"❌ Erreur requête Massive pour {ticker} : {exc}")
        return None
    except ValueError as exc:
        print(f"❌ Erreur parsing JSON pour {ticker} : {exc}")
        return None


def try_fetch_variants(variants: List[str], start_date: date, end_date: date) -> Optional[tuple[str, pd.DataFrame]]:
    """Essaie chaque variante de ticker et retourne le premier DataFrame valide."""
    for tick in variants:
        print(f"-> Tentative récupération pour '{tick}' ...")
        df = fetch_massive_agg_minute(tick, start_date, end_date, agg_minutes=TIMEFRAME_MINUTES)
        if df is not None and not df.empty:
            print(f"✅ Succès avec ticker '{tick}' ({len(df)} points).")
            return tick, df
        time.sleep(0.5)  # pause courte entre tentatives
    return None


def calculate_logreturn_correlation(df1: pd.DataFrame, df2: pd.DataFrame, min_points: int = MIN_DATA_POINTS) -> Optional[float]:
    """Aligne les 'Close', calcule log returns et corrélation Pearson."""
    combined = pd.concat([df1["Close"], df2["Close"]], axis=1, keys=["XAU_Close", "UUP_Close"])
    combined = combined.dropna()
    if len(combined) < min_points:
        print(f"⚠️ Pas assez de points synchronisés : {len(combined)} (< {min_points})")
        return None
    combined["XAU_r"] = np.log(combined["XAU_Close"] / combined["XAU_Close"].shift(1))
    combined["UUP_r"] = np.log(combined["UUP_Close"] / combined["UUP_Close"].shift(1))
    combined = combined.dropna()
    return combined["XAU_r"].corr(combined["UUP_r"])


# ----------------------------
# INTERPRÉTATION AUTOMATIQUE
# ----------------------------
def interpret_correlation(rho: float) -> str:
    """Retourne interprétation automatique pour trading GOLD/USD."""
    if rho <= -0.7:
        strength = "TRÈS FORTE corrélation négative"
        behavior = (
            "→ Gold et UUP évoluent fortement en sens inverse.\n"
            "→ UUP ↑ = GOLD ↓\n"
            "→ UUP ↓ = GOLD ↑\n"
        )
        trading = "Contexte parfait : UUP comme indicateur leader du GOLD."
    elif -0.7 < rho <= -0.3:
        strength = "corrélation négative MODÉRÉE"
        behavior = (
            "→ Relation inverse présente mais moins précise.\n"
            "→ UUP ↑ = GOLD ↓ généralement.\n"
        )
        trading = "UUP utile mais nécessite confirmation via volumes ou price action."
    elif -0.3 < rho < 0.3:
        strength = "corrélation FAIBLE / NEUTRE"
        behavior = "→ Gold et UUP ne réagissent pas l’un à l’autre.\n"
        trading = "Éviter de baser un trade sur le dollar. Possible changement de régime."
    elif 0.3 <= rho < 0.7:
        strength = "corrélation POSITIVE anormale"
        behavior = "→ Gold et USD montent ou baissent ensemble.\n"
        trading = "Risque élevé : relation USD ↔ Gold cassée."
    else:
        strength = "corrélation POSITIVE TRÈS FORTE 🚨"
        behavior = "→ Régime totalement inversé.\n"
        trading = "Ne pas baser de stratégie sur UUP. Contexte instable."

    explanation = (
        f"\n🔎 Interprétation du régime actuel\n"
        f"Corrélation = {rho:.4f} → {strength}\n\n"
        f"📉 Comportement attendu :\n{behavior}\n"
        f"🎯 Conséquence trading :\n{trading}\n"
    )
    return explanation


# ----------------------------
# POINT D'ENTRÉE
# ----------------------------
def main():
    print("--- Analyse corrélation XAUUSD vs UUP (M5) via Massive API ---")
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=DAYS_LOOKBACK)

    # Récupération XAUUSD
    df_xau = fetch_massive_agg_minute(XAU_TICKER, start_date, end_date, agg_minutes=TIMEFRAME_MINUTES)
    if df_xau is None:
        print("❌ Impossible de récupérer XAUUSD. Arrêt.")
        return

    # Récupération UUP (essaie variantes)
    res = try_fetch_variants(UUP_VARIANTS, start_date, end_date)
    if res is None:
        print("❌ Aucune variante UUP n'a retourné de données valides.")
        return
    used_ticker, df_uup = res

    print(f"-> Points XAU: {len(df_xau)}, Points UUP({used_ticker}): {len(df_uup)}")

    # Calcul corrélation
    rho = calculate_logreturn_correlation(df_xau, df_uup)
    if rho is None:
        print("❌ Corrélation non calculée (données insuffisantes).")
        return

    # Affichage final et interprétation
    print("\n" + "=" * 70)
    print(f"CORRÉLATION (log returns) XAUUSD vs UUP({used_ticker}) : {rho:.4f}")
    print(interpret_correlation(rho))
    print("=" * 70)


if __name__ == "__main__":
    main()