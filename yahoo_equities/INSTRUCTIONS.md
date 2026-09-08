# Folder structure

This folder contains ticker data from yfinance from January 1st 2025 up to September 7 2026. There is a folder named "prices", which contains price data for each ticker. Ticker metadata is present in "universe.csv". Valid tickers are in "valid_tickers.csv", while substantial (those that have the data for each trading day in the given period) are in "substantial_tickers.csv". Whether a ticker is valid, invalid, sufficient or not, is stored in "status.csv".

# How to handle the files

All files in this folder are read-only, except maybe for this file, "INSTRUCTIONS.md". You should not change any of them whatsoever. If you need to create a new file, do it in a separate folder. Only "INSTRUCTIONS.md" may be changed, and that only in one case: when you ask me a question on how to do something and I give you the direction. This file should reflect what was done with the data.

# Task

1. cluster tickers according to their region (read the region from "universe.csv");
2. for each region, do the following:
    1. make a hash table (dictionary in Python) of type String x Set{String}
    2. for each ticker in that region:
        1. find its set of trading days (those that had prices);
        2. hash that set (or a list), using polynomial hashing function. To do this, assign to each date a number, starting from 1. Choose as the prime number p = 1e9+7;
        3. store the ticker in the hash table, where the key is the hash of the trading day set, and the value is the ticker string.
    3. for each (region, key) group, make exactly one CSV file containing the shared `Date` column and the `Close` price of every ticker associated with that group. Use one column per ticker, named `<ticker>__Close`. Store the group CSV files in a separate output folder.

The group identity is the combination of the region and the trading-day hash. Only tickers listed in `substantial_tickers.csv` are included.

# Human-AI interaction

Always ask what to do whenever there's an ambiguity or vagueness.

# Code

Store the code in the same output folder as those csv files. Follow SOLID design patterns, make the code minimal, and separate into clear modules. Do it in Python.
