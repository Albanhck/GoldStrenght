import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List

# --- Configuration MT5 et Calcul ---
# Note: Ces identifiants DOIVENT être ceux de votre compte MT5.
MT5_LOGIN = 5042417769  # VOTRE NUMÉRO DE COMPTE MT5
MT5_PASSWORD = "3kKeRzL*"  # VOTRE MOT DE PASSE
MT5_SERVER = "-f3mQyMq"  # Ex: "ICMarkets-Demo" ou le nom du serveur de votre courtier

# Tickers requis (XAUUSD + 3 devises pour la dérivation)
TICKER_XAU = "XAUUSD"
TICKERS_FX = ["EURUSD", "USDJPY", "GBPUSD"]

M5_BARS_COUNT = 300  # Nombre de bougies M5 à analyser (environ 25 heures de données)
EMA_PERIOD = 20  # Période de lissage
SCALING_FACTOR = 1000  # Mise à l'échelle pour l'affichage (plage cible: ~±0.0x)


def get_mt5_data() -> Optional[pd.DataFrame]:
    """
    Se connecte à MT5 et récupère les données de clôture M5 pour tous les actifs requis.
    """
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"❌ Échec de l'initialisation/connexion MT5. Code d'erreur: {mt5.last_error()}")
        return None

    data_frames = {}

    # 1. Récupération des données pour tous les tickers
    all_tickers = [TICKER_XAU] + TICKERS_FX
    rates_end_time = datetime.now()  # Récupérer les données jusqu'à maintenant

    for symbol in all_tickers:
        # Tenter la récupération
        rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M5, rates_end_time, M5_BARS_COUNT)

        if rates is None or len(rates) < M5_BARS_COUNT:
            print(f"⚠️ Échec de récupération pour {symbol}. Données manquantes.")
            mt5.shutdown()
            return None

        # Conversion en DataFrame Pandas
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')
        data_frames[symbol] = df['close']  # Utiliser le prix de clôture

    mt5.shutdown()

    # 2. Synchronisation et fusion
    combined_df = pd.concat(data_frames, axis=1).dropna()
    combined_df.columns.name = 'Ticker'

    if len(combined_df) < M5_BARS_COUNT / 2:  # Seuil minimum de synchronisation
        print("❌ Les données ne sont pas suffisamment synchronisées entre les paires.")
        return None

    return combined_df


# --- Fonction de Calcul de l'Indice de Force Composé ---

def calculate_gold_strength_index(combined_df: pd.DataFrame) -> Optional[float]:
    """
    Calcule l'Indice de Force du Gold (GSI) en moyenne arithmétique des variations dérivées.
    """
    df_returns = pd.DataFrame(index=combined_df.index)

    # 1. Dérivation des Paires XAU/Devise et Calcul des Variations

    for currency_ticker in TICKERS_FX:
        # 1a. Détermination de la Devise de la Paire (JPY, USD, EUR, etc.)
        if currency_ticker == 'EURUSD':
            devise_code = 'EUR'
        elif currency_ticker == 'USDJPY':
            devise_code = 'JPY'
        elif currency_ticker == 'GBPUSD':
            devise_code = 'GBP'
        else:
            continue  # Ignorer les paires non traitées

        xau_devise_name = f'XAU/{devise_code}'

        # 1b. Calcul de la Paire Dérivée (XAU/Devise)
        if currency_ticker.startswith('EUR') or currency_ticker.startswith('GBP'):
            # Paire Devise/USD (ex: EURUSD) -> XAU/EUR = XAUUSD / EURUSD
            price_series = combined_df[TICKER_XAU] / combined_df[currency_ticker]
        elif currency_ticker.startswith('USD'):
            # Paire USD/Devise (ex: USDJPY) -> XAU/JPY = XAUUSD * USDJPY
            price_series = combined_df[TICKER_XAU] * combined_df[currency_ticker]
        else:
            continue

        # 1c. Calcul de la Variation en Pourcentage (Return)
        df_returns[f'Var_{xau_devise_name}'] = price_series.pct_change()

    # 2. Ajout de la variation XAUUSD elle-même (XAU/USD)
    df_returns['Var_XAU/USD'] = combined_df[TICKER_XAU].pct_change()

    df_returns = df_returns.dropna()

    # 3. Calcul de la Moyenne Arithmétique des Rendements (GSI Brut)
    return_columns = [col for col in df_returns.columns if col.startswith('Var_')]

    if len(return_columns) < 2:
        return None

    df_returns['GSI_Raw'] = df_returns[return_columns].mean(axis=1)

    # 4. Lissage et Mise à l'Échelle (Simulation Mataf)
    df_returns['Smoothed_Index'] = df_returns['GSI_Raw'].ewm(span=EMA_PERIOD, adjust=False).mean()

    final_scaled_index = df_returns['Smoothed_Index'].iloc[-1] * SCALING_FACTOR

    return final_scaled_index


# --- Point d'Entrée Principal ---

def main():
    # 1. Acquisition des données
    combined_df = get_mt5_data()

    if combined_df is None:
        return

    # 2. Calcul de l'Indice de Force Composé (GSI)
    gold_index = calculate_gold_strength_index(combined_df)

    if gold_index is None:
        print("❌ Le calcul de l'Indice de Force a échoué (données insuffisantes après dérivation).")
        return

    # 3. Affichage et Interprétation

    pressure = "HAUSSIÈRE" if gold_index > 0 else "BAISSIÈRE"

    print("\n" + "=" * 80)
    print(f"| 🥇 Indice de Force du Gold (GSI) - MT5 M{mt5.TIMEFRAME_M5} ({M5_BARS_COUNT} barres) |")
    print("|" + "-" * 78 + "|")
    print(f"| Valeur Actuelle (Indice Simulé) : {gold_index:,.4f}")
    print(f"| Pression : {pressure}")
    print(f"| Interprétation : Mesure la moyenne des forces de XAU/USD, XAU/EUR, XAU/JPY, etc.")
    print("=" * 80)


if __name__ == "__main__":
    main()