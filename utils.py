import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy import stats

#Fufill the utils requirement
def download_tickers(tickers, period_days=365, save_dir="."):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=period_days)

    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )
        # 5 attributes
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        # compute daily percent change, this is also my class
        df['Pct Change'] = df['Close'].pct_change() * 100
        df.dropna(inplace=True)

        # save to CSV
        filename = f"{save_dir}/{ticker}-past-year-data.csv"
        df.to_csv(filename)
        print(f"Saved {filename}")