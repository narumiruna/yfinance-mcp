from typing import Literal

SearchType = Literal[
    "all",
    "quotes",
    "news",
]


TopType = Literal[
    "top_etfs",
    "top_mutual_funds",
    "top_companies",
    "top_growth_companies",
    "top_performing_companies",
]


Sector = Literal[
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
]


Period = Literal[
    "1d",
    "5d",
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "10y",
    "ytd",
    "max",
]


Interval = Literal[
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
]


ChartType = Literal[
    "price_volume",
    "vwap",
    "volume_profile",
]
