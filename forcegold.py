import requests
import pandas as pd
import numpy as np
import time
from typing import Optional, Dict
from datetime import date, timedelta

# --- Configuration Alpha Vantage ---

API_KEY = "IOS5XMNLR4HFTUIZ"
BASE_URL = "https://www.alphavantage.co/query"

# --- Configuration des Tickers et Période ---
TICKER_XAU = "GLD"  # ETF pour l'Or (proxy de XAUUSD)
TICKER_DXY = "UUP"  # Ticker standard pour l'Indice du Dollar sur AV (Peut nécessiter d'être vérifié)
INTERVAL = "5min"  # Intervalle M5
OUTPUT_SIZE = "full"  # Pour obtenir les 30 derniers jours (sinon, compact=100 points)
DAYS_LOOKBACK = 30  # Période utilisée par 'full' (environ 30 jours)
MIN_DATA_POINTS = 100  # Seuil minimum pour considérer la corrélation significative


# --- 1. Fonction d'Acquisition de Données M5 (Alpha Vantage) ---

def fetch_data_av(symbol: str, interval: str, outputsize: str, api_key: str) -> Optional[pd.DataFrame]:
    """
    Récupère les données intraday d'un symbole via Alpha Vantage.
    """
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': interval,
        'outputsize': outputsize,
        'apikey': api_key,
        'datatype': 'json',
        'extended_hours': 'false'  # Limiter aux heures de trading régulières pour l'alignement
    }

    print(f"-> Requête API pour {symbol} ({interval})...")

    try:
        response = requests.get(BASE_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        # Le nom de la clé de la série temporelle dépend de l'intervalle (ex: 'Time Series (5min)')
        time_series_key = f"Time Series ({interval})"

        if 'Error Message' in data:
            print(f"❌ Erreur AV pour {symbol}: {data['Error Message']}")
            return None
        if time_series_key not in data:
            # Souvent un message de limite de taux ou de données manquantes
            print(f"⚠️ Données non trouvées pour {symbol}. Statut: {data.get('Note', 'Inconnu')}")
            return None

        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')

        # Nettoyage et conversion
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={'4. close': 'Close'})  # Alpha Vantage utilise des préfixes numériques
        df['Close'] = pd.to_numeric(df['Close'])

        return df[['Close']].sort_index()

    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur réseau lors de la requête pour {symbol}: {e}")
        return None


# --- 2. Fonction d'Analyse de Corrélation ---

def calculate_correlation(df_xau: pd.DataFrame, df_dxy: pd.DataFrame) -> Optional[float]:
    """
    Calcule le coefficient de corrélation de Pearson entre les variations (Log Returns).
    """

    # 1. Alignement des Données
    combined_df = pd.concat([df_xau['Close'], df_dxy['Close']], axis=1, keys=['XAU_Close', 'DXY_Close'])
    combined_df = combined_df.dropna()

    if len(combined_df) < MIN_DATA_POINTS:
        print(f"⚠️ Pas assez de points synchronisés ({len(combined_df)}). Nécessite au moins {MIN_DATA_POINTS}.")
        return None

    # 2. Calcul des Taux de Variation (Log Returns)
    combined_df['XAU_Return'] = np.log(combined_df['XAU_Close'] / combined_df['XAU_Close'].shift(1))
    combined_df['DXY_Return'] = np.log(combined_df['DXY_Close'] / combined_df['DXY_Close'].shift(1))

    combined_df = combined_df.dropna()

    # 3. Calcul du Coefficient de Corrélation de Pearson
    correlation = combined_df['XAU_Return'].corr(combined_df['DXY_Return'])

    return correlation


# --- 3. Point d'Entrée Principal ---

def main():
    print(f"--- 📊 Analyse de Corrélation XAU (GLD) / Dollar Index (DXY) ({INTERVAL} - {DAYS_LOOKBACK} jours) ---")

    # NOTE: Limite de 5 appels par minute pour le plan gratuit d'Alpha Vantage.
    # Nous devons introduire un délai pour éviter l'erreur 429.

    # 1. Acquisition des données XAU
    df_xau = fetch_data_av(TICKER_XAU, INTERVAL, OUTPUT_SIZE, API_KEY)

    if df_xau is None:
        print("\n❌ Arrêt du script.")
        return

    # Attente pour respecter la limite de taux (5 requêtes/min max)
    print("... Attente de 15 secondes pour la limite de taux Alpha Vantage...")
    time.sleep(15)

    # 2. Acquisition des données DXY
    df_dxy = fetch_data_av(TICKER_DXY, INTERVAL, OUTPUT_SIZE, API_KEY)

    if df_dxy is None:
        print("\n❌ Arrêt du script.")
        return

    print(f"-> {TICKER_XAU} points : {len(df_xau)}")
    print(f"-> {TICKER_DXY} points : {len(df_dxy)}")

    # 3. Calcul de la corrélation
    rho = calculate_correlation(df_xau, df_dxy)

    if rho is None:
        print("\n❌ Corrélation non calculée car les séries ne sont pas assez synchronisées.")
        return

    # 4. Affichage et interprétation

    if rho < -0.7:
        strength, expected_relation = "TRES FORTE", "✅ NÉGATIVE confirmée. Potentiel de trading élevé."
    elif rho < -0.3:
        strength, expected_relation = "MODÉRÉE", "NÉGATIVE modérée."
    elif rho > 0.3:
        strength, expected_relation = "MODÉRÉE", "⚠️ POSITIVE modérée. Rupture de la relation historique."
    elif rho > 0.7:
        strength, expected_relation = "TRES FORTE", "🚨 POSITIVE TRES FORTE. Anomalie."
    else:
        strength, expected_relation = "FAIBLE", "Relation FAIBLE/NEUTRE."

    print("\n" + "=" * 80)
    print(f"| CORRÉLATION {TICKER_XAU} vs {TICKER_DXY} (M5 Log Returns) : {rho:.4f}")
    print("|" + "-" * 78 + "|")
    print(f"| Force Actuelle : **{strength}** (Négative attendue)")
    print(f"| Interprétation : {expected_relation}")
    print("=" * 80)


if __name__ == "__main__":
    main()