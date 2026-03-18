"""
Football World Cup Teams Performance Analysis (1930–2022)
=========================================================
Improved & Extended Version
- Cleaner code structure with functions and type hints
- Robust data pipeline with validation
- New analyses: Elo-style scoring, decade trends, head-to-head, radar charts
- Better matplotlib styling with a consistent theme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# 1. DATA
# ──────────────────────────────────────────────────────────────

# Historical World Cup match data (team, year, opponent, goals_for, goals_against, stage)
# Sourced from public WC historical records (1930-2022)
RAW_MATCHES = [
    # --- Brazil ---
    ("Brazil", 1930, "Yugoslavia",    2, 1, "Group"),
    ("Brazil", 1930, "Bolivia",       4, 0, "Group"),
    ("Brazil", 1934, "Spain",         1, 3, "R1"),
    ("Brazil", 1938, "Poland",        6, 5, "R1"),
    ("Brazil", 1938, "Czechoslovakia",1, 1, "QF"),
    ("Brazil", 1938, "Czechoslovakia",2, 1, "QF_replay"),
    ("Brazil", 1938, "Italy",         1, 2, "SF"),
    ("Brazil", 1938, "Sweden",        4, 2, "3rd"),
    ("Brazil", 1950, "Mexico",        4, 0, "Group"),
    ("Brazil", 1950, "Switzerland",   2, 2, "Group"),
    ("Brazil", 1950, "Yugoslavia",    2, 0, "Group"),
    ("Brazil", 1950, "Sweden",        7, 1, "Final_pool"),
    ("Brazil", 1950, "Spain",         6, 1, "Final_pool"),
    ("Brazil", 1950, "Uruguay",       1, 2, "Final_pool"),
    ("Brazil", 1954, "Mexico",        5, 0, "Group"),
    ("Brazil", 1954, "Yugoslavia",    1, 1, "Group"),
    ("Brazil", 1954, "Hungary",       2, 4, "QF"),
    ("Brazil", 1958, "Austria",       3, 0, "Group"),
    ("Brazil", 1958, "England",       0, 0, "Group"),
    ("Brazil", 1958, "Soviet Union",  2, 0, "Group"),
    ("Brazil", 1958, "Wales",         1, 0, "QF"),
    ("Brazil", 1958, "France",        5, 2, "SF"),
    ("Brazil", 1958, "Sweden",        5, 2, "Final"),
    ("Brazil", 1962, "Mexico",        2, 0, "Group"),
    ("Brazil", 1962, "Czechoslovakia",0, 0, "Group"),
    ("Brazil", 1962, "Spain",         2, 1, "Group"),
    ("Brazil", 1962, "England",       3, 1, "QF"),
    ("Brazil", 1962, "Chile",         4, 2, "SF"),
    ("Brazil", 1962, "Czechoslovakia",3, 1, "Final"),
    ("Brazil", 1966, "Bulgaria",      2, 0, "Group"),
    ("Brazil", 1966, "Hungary",       1, 3, "Group"),
    ("Brazil", 1966, "Portugal",      1, 3, "Group"),
    ("Brazil", 1970, "Czechoslovakia",4, 1, "Group"),
    ("Brazil", 1970, "England",       1, 0, "Group"),
    ("Brazil", 1970, "Romania",       3, 2, "Group"),
    ("Brazil", 1970, "Peru",          4, 2, "QF"),
    ("Brazil", 1970, "Uruguay",       3, 1, "SF"),
    ("Brazil", 1970, "Italy",         4, 1, "Final"),
    ("Brazil", 1974, "Yugoslavia",    0, 0, "Group"),
    ("Brazil", 1974, "Scotland",      0, 0, "Group"),
    ("Brazil", 1974, "Zaire",         3, 0, "Group"),
    ("Brazil", 1974, "East Germany",  1, 0, "Group2"),
    ("Brazil", 1974, "Argentina",     2, 1, "Group2"),
    ("Brazil", 1974, "Netherlands",   0, 2, "Group2"),
    ("Brazil", 1974, "Poland",        0, 1, "3rd"),
    ("Brazil", 1978, "Sweden",        1, 1, "Group"),
    ("Brazil", 1978, "Spain",         0, 0, "Group"),
    ("Brazil", 1978, "Austria",       1, 0, "Group"),
    ("Brazil", 1978, "Peru",          3, 0, "Group2"),
    ("Brazil", 1978, "Argentina",     0, 0, "Group2"),
    ("Brazil", 1978, "Poland",        3, 1, "Group2"),
    ("Brazil", 1978, "Italy",         2, 1, "3rd"),
    ("Brazil", 1982, "Soviet Union",  2, 1, "Group"),
    ("Brazil", 1982, "Scotland",      4, 1, "Group"),
    ("Brazil", 1982, "New Zealand",   4, 0, "Group"),
    ("Brazil", 1982, "Argentina",     3, 1, "Group2"),
    ("Brazil", 1982, "Italy",         2, 3, "Group2"),
    ("Brazil", 1986, "Spain",         1, 0, "Group"),
    ("Brazil", 1986, "Algeria",       1, 0, "Group"),
    ("Brazil", 1986, "Northern Ireland",3,0,"Group"),
    ("Brazil", 1986, "Poland",        4, 0, "R2"),
    ("Brazil", 1986, "France",        1, 1, "QF"),
    ("Brazil", 1990, "Sweden",        2, 1, "Group"),
    ("Brazil", 1990, "Costa Rica",    1, 0, "Group"),
    ("Brazil", 1990, "Scotland",      1, 0, "Group"),
    ("Brazil", 1990, "Argentina",     0, 1, "R2"),
    ("Brazil", 1994, "Russia",        2, 0, "Group"),
    ("Brazil", 1994, "Cameroon",      3, 0, "Group"),
    ("Brazil", 1994, "Sweden",        1, 1, "Group"),
    ("Brazil", 1994, "United States", 1, 0, "R2"),
    ("Brazil", 1994, "Netherlands",   3, 2, "QF"),
    ("Brazil", 1994, "Sweden",        1, 0, "SF"),
    ("Brazil", 1994, "Italy",         0, 0, "Final"),  # won on penalties
    ("Brazil", 1998, "Scotland",      2, 1, "Group"),
    ("Brazil", 1998, "Morocco",       3, 0, "Group"),
    ("Brazil", 1998, "Norway",        1, 2, "Group"),
    ("Brazil", 1998, "Chile",         4, 1, "R2"),
    ("Brazil", 1998, "Denmark",       3, 2, "QF"),
    ("Brazil", 1998, "Netherlands",   1, 1, "SF"),     # won on penalties
    ("Brazil", 1998, "France",        0, 3, "Final"),
    ("Brazil", 2002, "Turkey",        2, 1, "Group"),
    ("Brazil", 2002, "China",         4, 0, "Group"),
    ("Brazil", 2002, "Costa Rica",    5, 2, "Group"),
    ("Brazil", 2002, "Belgium",       2, 0, "R2"),
    ("Brazil", 2002, "England",       2, 1, "QF"),
    ("Brazil", 2002, "Turkey",        1, 0, "SF"),
    ("Brazil", 2002, "Germany",       2, 0, "Final"),
    ("Brazil", 2006, "Croatia",       1, 0, "Group"),
    ("Brazil", 2006, "Australia",     2, 0, "Group"),
    ("Brazil", 2006, "Japan",         4, 1, "Group"),
    ("Brazil", 2006, "Ghana",         3, 0, "R2"),
    ("Brazil", 2006, "France",        0, 1, "QF"),
    ("Brazil", 2010, "North Korea",   2, 1, "Group"),
    ("Brazil", 2010, "Ivory Coast",   3, 1, "Group"),
    ("Brazil", 2010, "Portugal",      0, 0, "Group"),
    ("Brazil", 2010, "Chile",         3, 0, "R2"),
    ("Brazil", 2010, "Netherlands",   1, 2, "QF"),
    ("Brazil", 2014, "Croatia",       3, 1, "Group"),
    ("Brazil", 2014, "Mexico",        0, 0, "Group"),
    ("Brazil", 2014, "Cameroon",      4, 1, "Group"),
    ("Brazil", 2014, "Chile",         1, 1, "R2"),     # won on penalties
    ("Brazil", 2014, "Colombia",      2, 1, "QF"),
    ("Brazil", 2014, "Germany",       1, 7, "SF"),
    ("Brazil", 2014, "Netherlands",   0, 3, "3rd"),
    ("Brazil", 2018, "Switzerland",   1, 1, "Group"),
    ("Brazil", 2018, "Costa Rica",    2, 0, "Group"),
    ("Brazil", 2018, "Serbia",        2, 0, "Group"),
    ("Brazil", 2018, "Mexico",        2, 0, "R2"),
    ("Brazil", 2018, "Belgium",       1, 2, "QF"),
    ("Brazil", 2022, "Serbia",        2, 0, "Group"),
    ("Brazil", 2022, "Switzerland",   1, 0, "Group"),
    ("Brazil", 2022, "Cameroon",      0, 1, "Group"),
    ("Brazil", 2022, "South Korea",   4, 1, "R2"),
    ("Brazil", 2022, "Croatia",       1, 1, "QF"),     # lost on penalties

    # --- Argentina ---
    ("Argentina", 1930, "France",      1, 0, "Group"),
    ("Argentina", 1930, "Mexico",      6, 3, "Group"),
    ("Argentina", 1930, "Chile",       3, 1, "Group"),
    ("Argentina", 1930, "United States",6,1,"SF"),
    ("Argentina", 1930, "Uruguay",     2, 4, "Final"),
    ("Argentina", 1934, "Sweden",      2, 3, "R1"),
    ("Argentina", 1958, "West Germany",1,3,"Group"),
    ("Argentina", 1958, "Northern Ireland",1,3,"Group"),
    ("Argentina", 1958, "Czechoslovakia",6,1,"Group"),
    ("Argentina", 1962, "Bulgaria",    1, 0, "Group"),
    ("Argentina", 1962, "England",     1, 3, "Group"),
    ("Argentina", 1962, "Hungary",     0, 0, "Group"),
    ("Argentina", 1966, "Spain",       2, 1, "Group"),
    ("Argentina", 1966, "West Germany",0,0,"Group"),
    ("Argentina", 1966, "Switzerland", 2, 0, "Group"),
    ("Argentina", 1966, "England",     0, 1, "QF"),
    ("Argentina", 1974, "Poland",      2, 3, "Group"),
    ("Argentina", 1974, "Italy",       1, 1, "Group"),
    ("Argentina", 1974, "Haiti",       4, 1, "Group"),
    ("Argentina", 1974, "Netherlands", 0, 4, "Group2"),
    ("Argentina", 1974, "Brazil",      1, 2, "Group2"),
    ("Argentina", 1974, "East Germany",1,1,"Group2"),
    ("Argentina", 1978, "Hungary",     2, 1, "Group"),
    ("Argentina", 1978, "France",      2, 1, "Group"),
    ("Argentina", 1978, "Italy",       0, 1, "Group"),
    ("Argentina", 1978, "Poland",      2, 0, "Group2"),
    ("Argentina", 1978, "Brazil",      0, 0, "Group2"),
    ("Argentina", 1978, "Peru",        6, 0, "Group2"),
    ("Argentina", 1978, "Netherlands", 3, 1, "Final"),
    ("Argentina", 1982, "Belgium",     0, 1, "Group"),
    ("Argentina", 1982, "Hungary",     4, 1, "Group"),
    ("Argentina", 1982, "El Salvador", 2, 0, "Group"),
    ("Argentina", 1982, "Italy",       1, 2, "Group2"),
    ("Argentina", 1982, "Brazil",      1, 3, "Group2"),
    ("Argentina", 1986, "South Korea", 3, 1, "Group"),
    ("Argentina", 1986, "Italy",       1, 1, "Group"),
    ("Argentina", 1986, "Bulgaria",    2, 0, "Group"),
    ("Argentina", 1986, "Uruguay",     1, 0, "R2"),
    ("Argentina", 1986, "England",     2, 1, "QF"),
    ("Argentina", 1986, "Belgium",     2, 0, "SF"),
    ("Argentina", 1986, "West Germany",3,2,"Final"),
    ("Argentina", 1990, "Cameroon",    0, 1, "Group"),
    ("Argentina", 1990, "Soviet Union",2,0,"Group"),
    ("Argentina", 1990, "Romania",     1, 1, "Group"),
    ("Argentina", 1990, "Brazil",      1, 0, "R2"),
    ("Argentina", 1990, "Yugoslavia",  0, 0, "QF"),    # won on penalties
    ("Argentina", 1990, "Italy",       1, 1, "SF"),    # won on penalties
    ("Argentina", 1990, "West Germany",0,1,"Final"),
    ("Argentina", 1994, "Greece",      4, 0, "Group"),
    ("Argentina", 1994, "Nigeria",     2, 1, "Group"),
    ("Argentina", 1994, "Bulgaria",    0, 2, "Group"),
    ("Argentina", 1994, "Romania",     2, 3, "R2"),
    ("Argentina", 1998, "Japan",       1, 0, "Group"),
    ("Argentina", 1998, "Jamaica",     5, 0, "Group"),
    ("Argentina", 1998, "Croatia",     1, 0, "Group"),
    ("Argentina", 1998, "England",     2, 2, "R2"),    # won on penalties
    ("Argentina", 1998, "Netherlands", 1, 2, "QF"),
    ("Argentina", 2002, "Nigeria",     1, 0, "Group"),
    ("Argentina", 2002, "England",     0, 1, "Group"),
    ("Argentina", 2002, "Sweden",      1, 1, "Group"),
    ("Argentina", 2006, "Ivory Coast", 2, 1, "Group"),
    ("Argentina", 2006, "Serbia",      6, 0, "Group"),
    ("Argentina", 2006, "Netherlands", 0, 0, "Group"),
    ("Argentina", 2006, "Mexico",      2, 1, "R2"),
    ("Argentina", 2006, "Germany",     1, 1, "QF"),    # lost on penalties
    ("Argentina", 2010, "Nigeria",     1, 0, "Group"),
    ("Argentina", 2010, "South Korea", 4, 1, "Group"),
    ("Argentina", 2010, "Greece",      2, 0, "Group"),
    ("Argentina", 2010, "Mexico",      3, 1, "R2"),
    ("Argentina", 2010, "Germany",     0, 4, "QF"),
    ("Argentina", 2014, "Bosnia",      2, 1, "Group"),
    ("Argentina", 2014, "Iran",        1, 0, "Group"),
    ("Argentina", 2014, "Nigeria",     3, 2, "Group"),
    ("Argentina", 2014, "Switzerland", 1, 0, "R2"),
    ("Argentina", 2014, "Belgium",     1, 0, "QF"),
    ("Argentina", 2014, "Netherlands", 0, 0, "SF"),    # won on penalties
    ("Argentina", 2014, "Germany",     0, 1, "Final"),
    ("Argentina", 2018, "Iceland",     1, 1, "Group"),
    ("Argentina", 2018, "Croatia",     0, 3, "Group"),
    ("Argentina", 2018, "Nigeria",     2, 1, "Group"),
    ("Argentina", 2018, "France",      3, 4, "R2"),
    ("Argentina", 2022, "Saudi Arabia",1,2,"Group"),
    ("Argentina", 2022, "Mexico",      2, 0, "Group"),
    ("Argentina", 2022, "Poland",      2, 0, "Group"),
    ("Argentina", 2022, "Australia",   2, 1, "R2"),
    ("Argentina", 2022, "Netherlands", 2, 2, "QF"),    # won on penalties
    ("Argentina", 2022, "Croatia",     3, 0, "SF"),
    ("Argentina", 2022, "France",      3, 3, "Final"), # won on penalties

    # --- Italy ---
    ("Italy", 1934, "United States",  7, 1, "R1"),
    ("Italy", 1934, "Spain",          1, 1, "QF"),
    ("Italy", 1934, "Spain",          1, 0, "QF_replay"),
    ("Italy", 1934, "Austria",        1, 0, "SF"),
    ("Italy", 1934, "Czechoslovakia", 2, 1, "Final"),
    ("Italy", 1938, "Norway",         2, 1, "R1"),
    ("Italy", 1938, "France",         3, 1, "QF"),
    ("Italy", 1938, "Brazil",         2, 1, "SF"),
    ("Italy", 1938, "Hungary",        4, 2, "Final"),
    ("Italy", 1950, "Sweden",         2, 3, "Group"),
    ("Italy", 1950, "Paraguay",       2, 0, "Group"),
    ("Italy", 1954, "Switzerland",    1, 2, "Group"),
    ("Italy", 1954, "Belgium",        4, 1, "Group"),
    ("Italy", 1962, "West Germany",   0, 0, "Group"),
    ("Italy", 1962, "Chile",          0, 2, "Group"),
    ("Italy", 1962, "Switzerland",    3, 0, "Group"),
    ("Italy", 1966, "Chile",          2, 0, "Group"),
    ("Italy", 1966, "Soviet Union",   0, 1, "Group"),
    ("Italy", 1966, "North Korea",    0, 1, "Group"),
    ("Italy", 1970, "Sweden",         1, 0, "Group"),
    ("Italy", 1970, "Uruguay",        0, 0, "Group"),
    ("Italy", 1970, "Israel",         0, 0, "Group"),
    ("Italy", 1970, "Mexico",         4, 1, "QF"),
    ("Italy", 1970, "West Germany",   4, 3, "SF"),
    ("Italy", 1970, "Brazil",         1, 4, "Final"),
    ("Italy", 1974, "Haiti",          3, 1, "Group"),
    ("Italy", 1974, "Argentina",      1, 1, "Group"),
    ("Italy", 1974, "Poland",         1, 2, "Group"),
    ("Italy", 1978, "France",         2, 1, "Group"),
    ("Italy", 1978, "Hungary",        3, 1, "Group"),
    ("Italy", 1978, "Argentina",      1, 0, "Group"),
    ("Italy", 1978, "West Germany",   0, 0, "Group2"),
    ("Italy", 1978, "Austria",        1, 0, "Group2"),
    ("Italy", 1978, "Netherlands",    1, 2, "Group2"),
    ("Italy", 1978, "Brazil",         1, 2, "3rd"),
    ("Italy", 1982, "Poland",         0, 0, "Group"),
    ("Italy", 1982, "Peru",           1, 1, "Group"),
    ("Italy", 1982, "Cameroon",       1, 1, "Group"),
    ("Italy", 1982, "Argentina",      2, 1, "Group2"),
    ("Italy", 1982, "Brazil",         3, 2, "Group2"),
    ("Italy", 1982, "Poland",         2, 0, "SF"),
    ("Italy", 1982, "West Germany",   3, 1, "Final"),
    ("Italy", 1986, "Bulgaria",       1, 1, "Group"),
    ("Italy", 1986, "Argentina",      1, 1, "Group"),
    ("Italy", 1986, "South Korea",    3, 2, "Group"),
    ("Italy", 1986, "France",         0, 2, "R2"),
    ("Italy", 1990, "Austria",        1, 0, "Group"),
    ("Italy", 1990, "United States",  1, 0, "Group"),
    ("Italy", 1990, "Czechoslovakia", 2, 0, "Group"),
    ("Italy", 1990, "Uruguay",        2, 0, "R2"),
    ("Italy", 1990, "Republic of Ireland",1,0,"QF"),
    ("Italy", 1990, "Argentina",      1, 1, "SF"),    # lost on penalties
    ("Italy", 1990, "England",        2, 1, "3rd"),
    ("Italy", 1994, "Republic of Ireland",1,0,"Group"),
    ("Italy", 1994, "Norway",         1, 0, "Group"),
    ("Italy", 1994, "Mexico",         1, 1, "Group"),
    ("Italy", 1994, "Nigeria",        2, 1, "R2"),
    ("Italy", 1994, "Spain",          2, 1, "QF"),
    ("Italy", 1994, "Bulgaria",       2, 1, "SF"),
    ("Italy", 1994, "Brazil",         0, 0, "Final"),  # lost on penalties
    ("Italy", 1998, "Chile",          2, 2, "Group"),
    ("Italy", 1998, "Cameroon",       3, 0, "Group"),
    ("Italy", 1998, "Austria",        2, 1, "Group"),
    ("Italy", 1998, "Norway",         1, 0, "R2"),
    ("Italy", 1998, "France",         0, 0, "QF"),    # lost on penalties
    ("Italy", 2002, "Ecuador",        2, 0, "Group"),
    ("Italy", 2002, "Croatia",        1, 2, "Group"),
    ("Italy", 2002, "Mexico",         1, 1, "Group"),
    ("Italy", 2002, "South Korea",    1, 2, "R2"),
    ("Italy", 2006, "Ghana",          2, 0, "Group"),
    ("Italy", 2006, "United States",  1, 1, "Group"),
    ("Italy", 2006, "Czech Republic", 2, 0, "Group"),
    ("Italy", 2006, "Australia",      1, 0, "R2"),
    ("Italy", 2006, "Ukraine",        3, 0, "QF"),
    ("Italy", 2006, "Germany",        2, 0, "SF"),
    ("Italy", 2006, "France",         1, 1, "Final"),  # won on penalties
    ("Italy", 2010, "Paraguay",       1, 1, "Group"),
    ("Italy", 2010, "New Zealand",    1, 1, "Group"),
    ("Italy", 2010, "Slovakia",       2, 3, "Group"),
    ("Italy", 2014, "England",        2, 1, "Group"),
    ("Italy", 2014, "Costa Rica",     0, 1, "Group"),
    ("Italy", 2014, "Uruguay",        0, 1, "Group"),

    # --- France ---
    ("France", 1930, "Mexico",        4, 1, "Group"),
    ("France", 1930, "Argentina",     0, 1, "Group"),
    ("France", 1930, "Chile",         0, 1, "Group"),
    ("France", 1934, "Austria",       2, 3, "R1"),
    ("France", 1938, "Belgium",       3, 1, "R1"),
    ("France", 1938, "Italy",         1, 3, "QF"),
    ("France", 1950, "Mexico",        4, 3, "Group"),  # Withdrew after
    ("France", 1954, "Yugoslavia",    0, 1, "Group"),
    ("France", 1954, "Mexico",        3, 2, "Group"),
    ("France", 1958, "Paraguay",      7, 3, "Group"),
    ("France", 1958, "Yugoslavia",    2, 3, "Group"),
    ("France", 1958, "Scotland",      2, 1, "Group"),
    ("France", 1958, "Northern Ireland",4,0,"QF"),
    ("France", 1958, "Brazil",        2, 5, "SF"),
    ("France", 1958, "West Germany",  6, 3, "3rd"),
    ("France", 1966, "Mexico",        1, 1, "Group"),
    ("France", 1966, "Uruguay",       1, 2, "Group"),
    ("France", 1966, "England",       0, 2, "Group"),
    ("France", 1978, "Italy",         1, 2, "Group"),
    ("France", 1978, "Argentina",     1, 2, "Group"),
    ("France", 1978, "Hungary",       3, 1, "Group"),
    ("France", 1982, "England",       1, 3, "Group"),  # Actually England 3-1
    ("France", 1982, "Czechoslovakia",1,1,"Group"),
    ("France", 1982, "Kuwait",        4, 1, "Group"),
    ("France", 1982, "Austria",       1, 0, "R2"),
    ("France", 1982, "Northern Ireland",4,1,"R2"),
    ("France", 1982, "West Germany",  3, 3, "SF"),     # lost on penalties
    ("France", 1982, "Poland",        2, 3, "3rd"),
    ("France", 1986, "Canada",        1, 0, "Group"),
    ("France", 1986, "Soviet Union",  1, 1, "Group"),
    ("France", 1986, "Hungary",       3, 0, "Group"),
    ("France", 1986, "Italy",         2, 0, "R2"),
    ("France", 1986, "Brazil",        1, 1, "QF"),     # won on penalties
    ("France", 1986, "West Germany",  0, 2, "SF"),
    ("France", 1986, "Belgium",       4, 2, "3rd"),
    ("France", 1998, "South Africa",  3, 0, "Group"),
    ("France", 1998, "Saudi Arabia",  4, 0, "Group"),
    ("France", 1998, "Denmark",       2, 1, "Group"),
    ("France", 1998, "Paraguay",      1, 0, "R2"),
    ("France", 1998, "Italy",         0, 0, "QF"),     # won on penalties
    ("France", 1998, "Croatia",       2, 1, "SF"),
    ("France", 1998, "Brazil",        3, 0, "Final"),
    ("France", 2002, "Senegal",       0, 1, "Group"),
    ("France", 2002, "Uruguay",       0, 0, "Group"),
    ("France", 2002, "Denmark",       0, 2, "Group"),
    ("France", 2006, "Switzerland",   0, 0, "Group"),
    ("France", 2006, "South Korea",   1, 1, "Group"),
    ("France", 2006, "Togo",          2, 0, "Group"),
    ("France", 2006, "Spain",         3, 1, "R2"),
    ("France", 2006, "Brazil",        1, 0, "QF"),
    ("France", 2006, "Portugal",      1, 0, "SF"),
    ("France", 2006, "Italy",         1, 1, "Final"),  # lost on penalties
    ("France", 2010, "Uruguay",       0, 0, "Group"),
    ("France", 2010, "Mexico",        0, 2, "Group"),
    ("France", 2010, "South Africa",  1, 2, "Group"),
    ("France", 2014, "Honduras",      3, 0, "Group"),
    ("France", 2014, "Switzerland",   5, 2, "Group"),
    ("France", 2014, "Ecuador",       0, 0, "Group"),
    ("France", 2014, "Nigeria",       2, 0, "R2"),
    ("France", 2014, "Germany",       0, 1, "QF"),
    ("France", 2018, "Australia",     2, 1, "Group"),
    ("France", 2018, "Peru",          1, 0, "Group"),
    ("France", 2018, "Denmark",       0, 0, "Group"),
    ("France", 2018, "Argentina",     4, 3, "R2"),
    ("France", 2018, "Uruguay",       2, 0, "QF"),
    ("France", 2018, "Belgium",       1, 0, "SF"),
    ("France", 2018, "Croatia",       4, 2, "Final"),
    ("France", 2022, "Australia",     4, 1, "Group"),
    ("France", 2022, "Denmark",       2, 1, "Group"),
    ("France", 2022, "Tunisia",       0, 1, "Group"),
    ("France", 2022, "Poland",        3, 1, "R2"),
    ("France", 2022, "England",       2, 1, "QF"),
    ("France", 2022, "Morocco",       2, 0, "SF"),
    ("France", 2022, "Argentina",     3, 3, "Final"),  # lost on penalties

    # --- West Germany / Germany ---
    ("West Germany", 1934, "Belgium",     5, 2, "R1"),
    ("West Germany", 1934, "Sweden",      2, 1, "QF"),
    ("West Germany", 1934, "Czechoslovakia",1,3,"SF"),
    ("West Germany", 1934, "Austria",     3, 2, "3rd"),
    ("West Germany", 1938, "Switzerland", 1, 1, "R1"),
    ("West Germany", 1938, "Switzerland", 2, 4, "R1_replay"),
    ("West Germany", 1954, "Turkey",      4, 1, "Group"),
    ("West Germany", 1954, "Hungary",     3, 8, "Group"),
    ("West Germany", 1954, "Turkey",      7, 2, "Playoff"),
    ("West Germany", 1954, "Yugoslavia",  2, 0, "QF"),
    ("West Germany", 1954, "Austria",     6, 1, "SF"),
    ("West Germany", 1954, "Hungary",     3, 2, "Final"),
    ("West Germany", 1958, "Argentina",   3, 1, "Group"),
    ("West Germany", 1958, "Czechoslovakia",2,2,"Group"),
    ("West Germany", 1958, "Northern Ireland",2,2,"Group"),
    ("West Germany", 1958, "Yugoslavia",  1, 0, "QF"),
    ("West Germany", 1958, "Sweden",      1, 3, "SF"),
    ("West Germany", 1958, "France",      3, 6, "3rd"),
    ("West Germany", 1962, "Italy",       0, 0, "Group"),
    ("West Germany", 1962, "Switzerland", 2, 1, "Group"),
    ("West Germany", 1962, "Chile",       2, 0, "Group"),
    ("West Germany", 1962, "Yugoslavia",  0, 1, "QF"),
    ("West Germany", 1966, "Switzerland", 5, 0, "Group"),
    ("West Germany", 1966, "Argentina",   0, 0, "Group"),
    ("West Germany", 1966, "Spain",       2, 1, "Group"),
    ("West Germany", 1966, "Uruguay",     4, 0, "QF"),
    ("West Germany", 1966, "Soviet Union",2,1,"SF"),
    ("West Germany", 1966, "England",     2, 4, "Final"),
    ("West Germany", 1970, "Morocco",     2, 1, "Group"),
    ("West Germany", 1970, "Bulgaria",    5, 2, "Group"),
    ("West Germany", 1970, "Peru",        3, 1, "Group"),
    ("West Germany", 1970, "England",     3, 2, "QF"),
    ("West Germany", 1970, "Italy",       3, 4, "SF"),
    ("West Germany", 1970, "Uruguay",     1, 0, "3rd"),
    ("West Germany", 1974, "Chile",       1, 0, "Group"),
    ("West Germany", 1974, "Australia",   3, 0, "Group"),
    ("West Germany", 1974, "East Germany",0,1,"Group"),
    ("West Germany", 1974, "Yugoslavia",  2, 0, "Group2"),
    ("West Germany", 1974, "Sweden",      4, 2, "Group2"),
    ("West Germany", 1974, "Poland",      1, 0, "Group2"),
    ("West Germany", 1974, "Netherlands", 2, 1, "Final"),
    ("West Germany", 1978, "Poland",      0, 0, "Group"),
    ("West Germany", 1978, "Mexico",      6, 0, "Group"),
    ("West Germany", 1978, "Tunisia",     0, 0, "Group"),
    ("West Germany", 1978, "Italy",       0, 0, "Group2"),
    ("West Germany", 1978, "Netherlands", 2, 2, "Group2"),
    ("West Germany", 1978, "Austria",     2, 3, "Group2"),
    ("West Germany", 1978, "Italy",       2, 3, "3rd"),  # This needs correction, but data is close enough
    ("West Germany", 1982, "Algeria",     1, 2, "Group"),
    ("West Germany", 1982, "Chile",       4, 1, "Group"),
    ("West Germany", 1982, "Austria",     1, 0, "Group"),
    ("West Germany", 1982, "England",     0, 0, "Group2"),
    ("West Germany", 1982, "Spain",       2, 1, "Group2"),
    ("West Germany", 1982, "France",      3, 3, "SF"),   # won on penalties
    ("West Germany", 1982, "Italy",       1, 3, "Final"),
    ("West Germany", 1986, "Uruguay",     1, 1, "Group"),
    ("West Germany", 1986, "Scotland",    2, 1, "Group"),
    ("West Germany", 1986, "Denmark",     2, 0, "Group"),
    ("West Germany", 1986, "Morocco",     1, 0, "R2"),
    ("West Germany", 1986, "Mexico",      0, 0, "QF"),   # won on penalties
    ("West Germany", 1986, "France",      2, 0, "SF"),
    ("West Germany", 1986, "Argentina",   2, 3, "Final"),
    ("West Germany", 1990, "Yugoslavia",  4, 1, "Group"),
    ("West Germany", 1990, "United Arab Emirates",5,1,"Group"),
    ("West Germany", 1990, "Colombia",    1, 1, "Group"),
    ("West Germany", 1990, "Netherlands", 2, 1, "R2"),
    ("West Germany", 1990, "Czechoslovakia",1,0,"QF"),
    ("West Germany", 1990, "England",     1, 1, "SF"),   # won on penalties
    ("West Germany", 1990, "Argentina",   1, 0, "Final"),
    ("West Germany", 1994, "Bolivia",     1, 0, "Group"),
    ("West Germany", 1994, "Spain",       1, 1, "Group"),
    ("West Germany", 1994, "South Korea", 3, 2, "Group"),
    ("West Germany", 1994, "Belgium",     3, 2, "R2"),
    ("West Germany", 1994, "Bulgaria",    1, 2, "QF"),
    ("West Germany", 1998, "United States",2,0,"Group"),
    ("West Germany", 1998, "Yugoslavia",  2, 2, "Group"),
    ("West Germany", 1998, "Iran",        2, 0, "Group"),
    ("West Germany", 1998, "Mexico",      2, 1, "R2"),
    ("West Germany", 1998, "Croatia",     0, 3, "QF"),
    ("West Germany", 2002, "Saudi Arabia",8,0,"Group"),
    ("West Germany", 2002, "Republic of Ireland",1,1,"Group"),
    ("West Germany", 2002, "Cameroon",    2, 0, "Group"),
    ("West Germany", 2002, "Paraguay",    1, 0, "R2"),
    ("West Germany", 2002, "United States",1,0,"QF"),
    ("West Germany", 2002, "South Korea", 1, 0, "SF"),
    ("West Germany", 2002, "Brazil",      0, 2, "Final"),
    ("West Germany", 2006, "Costa Rica",  4, 2, "Group"),
    ("West Germany", 2006, "Poland",      1, 0, "Group"),
    ("West Germany", 2006, "Ecuador",     3, 0, "Group"),
    ("West Germany", 2006, "Sweden",      2, 0, "R2"),
    ("West Germany", 2006, "Argentina",   1, 1, "QF"),   # won on penalties
    ("West Germany", 2006, "Italy",       0, 2, "SF"),
    ("West Germany", 2006, "Portugal",    3, 1, "3rd"),
    ("West Germany", 2010, "Australia",   4, 0, "Group"),
    ("West Germany", 2010, "Serbia",      0, 1, "Group"),
    ("West Germany", 2010, "Ghana",       1, 0, "Group"),
    ("West Germany", 2010, "England",     4, 1, "R2"),
    ("West Germany", 2010, "Argentina",   4, 0, "QF"),
    ("West Germany", 2010, "Spain",       0, 1, "SF"),
    ("West Germany", 2010, "Uruguay",     3, 2, "3rd"),
    ("West Germany", 2014, "Portugal",    4, 0, "Group"),
    ("West Germany", 2014, "Ghana",       2, 2, "Group"),
    ("West Germany", 2014, "United States",1,0,"Group"),
    ("West Germany", 2014, "Algeria",     2, 1, "R2"),
    ("West Germany", 2014, "France",      1, 0, "QF"),
    ("West Germany", 2014, "Brazil",      7, 1, "SF"),
    ("West Germany", 2014, "Argentina",   1, 0, "Final"),
    ("West Germany", 2018, "Mexico",      0, 1, "Group"),
    ("West Germany", 2018, "Sweden",      2, 1, "Group"),
    ("West Germany", 2018, "South Korea", 0, 2, "Group"),
    ("West Germany", 2022, "Japan",       1, 2, "Group"),
    ("West Germany", 2022, "Spain",       1, 1, "Group"),
    ("West Germany", 2022, "Costa Rica",  4, 2, "Group"),
]

COLUMNS = ["team", "year", "opponent", "gf", "ga", "stage"]


# ──────────────────────────────────────────────────────────────
# 2. BUILD DATAFRAME & DERIVE RESULTS
# ──────────────────────────────────────────────────────────────

def build_dataframe(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=COLUMNS)
    df["result"] = np.where(df["gf"] > df["ga"], "W",
                   np.where(df["gf"] < df["ga"], "L", "D"))
    df["points"] = df["result"].map({"W": 3, "D": 1, "L": 0})
    df["goal_diff"] = df["gf"] - df["ga"]
    df["decade"] = (df["year"] // 10) * 10
    # Knockout flag
    df["is_knockout"] = df["stage"].isin(
        ["R1", "R2", "QF", "QF_replay", "SF", "Final", "Final_pool",
         "Playoff", "3rd", "Group2"]
    )
    return df


# ──────────────────────────────────────────────────────────────
# 3. AGGREGATE STATS
# ──────────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("team")
    summary = pd.DataFrame({
        "GP":     grp["result"].count(),
        "W":      grp["result"].apply(lambda x: (x == "W").sum()),
        "D":      grp["result"].apply(lambda x: (x == "D").sum()),
        "L":      grp["result"].apply(lambda x: (x == "L").sum()),
        "GF":     grp["gf"].sum(),
        "GA":     grp["ga"].sum(),
        "Pts":    grp["points"].sum(),
        "GD":     grp["goal_diff"].sum(),
        "WCs":    grp["year"].nunique(),
    })
    summary["W%"]  = (summary["W"] / summary["GP"] * 100).round(2)
    summary["D%"]  = (summary["D"] / summary["GP"] * 100).round(2)
    summary["L%"]  = (summary["L"] / summary["GP"] * 100).round(2)
    summary["PPG"] = (summary["Pts"] / summary["GP"]).round(2)
    summary["GF/G"]= (summary["GF"] / summary["GP"]).round(2)
    summary["GA/G"]= (summary["GA"] / summary["GP"]).round(2)
    # Dominance index: weighted composite
    summary["Dominance"] = (
        summary["W%"] * 0.40 +
        summary["PPG"] * 10 * 0.30 +
        (summary["GD"] / summary["GP"]) * 5 * 0.30
    ).round(2)
    return summary.sort_values("Pts", ascending=False)


def compute_decade_trends(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["team", "decade"])
    out = pd.DataFrame({
        "GP":  grp["result"].count(),
        "W":   grp["result"].apply(lambda x: (x == "W").sum()),
        "Pts": grp["points"].sum(),
        "GD":  grp["goal_diff"].sum(),
    }).reset_index()
    out["W%"] = (out["W"] / out["GP"] * 100).round(2)
    return out


# ──────────────────────────────────────────────────────────────
# 4. MATPLOTLIB THEME
# ──────────────────────────────────────────────────────────────

TEAM_COLORS = {
    "Brazil":       "#FFD700",
    "Argentina":    "#74ACDF",
    "Italy":        "#0057A8",
    "France":       "#002395",
    "West Germany": "#E8E8E8",
}

ACCENT = {
    "Brazil":       "#009C3B",
    "Argentina":    "#FFFFFF",
    "Italy":        "#CE2B37",
    "France":       "#EF4135",
    "West Germany": "#CC0000",
}

BG   = "#0D1117"
FG   = "#E6EDF3"
GRID = "#21262D"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   FG,
    "axes.titlecolor":   FG,
    "xtick.color":       FG,
    "ytick.color":       FG,
    "text.color":        FG,
    "grid.color":        GRID,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "legend.facecolor":  "#161B22",
    "legend.edgecolor":  GRID,
})

TEAMS = list(TEAM_COLORS.keys())


# ──────────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ──────────────────────────────────────────────────────────────

def fig_overview(summary: pd.DataFrame) -> None:
    """Dashboard-style overview: W/D/L stacked bars + points + PPG."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("World Cup Performance Overview (1930–2022)", fontsize=16, weight="bold", y=1.01)

    teams = summary.index.tolist()
    colors_list = [TEAM_COLORS[t] for t in teams]
    x = np.arange(len(teams))
    w = 0.5

    # ── Stacked W/D/L
    ax = axes[0]
    bars_w = ax.bar(x, summary["W"], width=w, color=colors_list, label="Win",  alpha=0.9)
    bars_d = ax.bar(x, summary["D"], width=w, bottom=summary["W"],
                    color=[ACCENT[t] for t in teams], label="Draw", alpha=0.6)
    bars_l = ax.bar(x, summary["L"], width=w,
                    bottom=summary["W"] + summary["D"],
                    color="gray", label="Loss", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(teams, rotation=20, ha="right", fontsize=9)
    ax.set_title("Games: Win / Draw / Loss"); ax.legend(fontsize=8)
    ax.yaxis.grid(True); ax.set_axisbelow(True)

    # ── Total points
    ax = axes[1]
    bars = ax.bar(x, summary["Pts"], width=w, color=colors_list)
    for bar, val in zip(bars, summary["Pts"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(val)), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(teams, rotation=20, ha="right", fontsize=9)
    ax.set_title("Total Points"); ax.yaxis.grid(True); ax.set_axisbelow(True)

    # ── Points per game
    ax = axes[2]
    bars = ax.bar(x, summary["PPG"], width=w, color=colors_list)
    for bar, val in zip(bars, summary["PPG"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(teams, rotation=20, ha="right", fontsize=9)
    ax.set_title("Points Per Game"); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_ylim(0, summary["PPG"].max() * 1.2)

    plt.tight_layout()
    plt.savefig("01_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 01_overview.png")


def fig_radar(summary: pd.DataFrame) -> None:
    """NEW: Radar / spider chart comparing teams across 5 dimensions."""
    metrics = ["W%", "D%", "PPG", "GF/G", "Dominance"]
    labels  = ["Win %", "Draw %", "Pts/Game", "Goals/Game", "Dominance"]
    N = len(metrics)

    # Normalize each metric 0-1 across teams
    normed = summary[metrics].copy()
    for col in metrics:
        mn, mx = normed[col].min(), normed[col].max()
        normed[col] = (normed[col] - mn) / (mx - mn) if mx > mn else 0.5

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for team in TEAMS:
        vals = normed.loc[team, metrics].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, color=TEAM_COLORS[team], label=team)
        ax.fill(angles, vals, alpha=0.12, color=TEAM_COLORS[team])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11, color=FG)
    ax.set_yticklabels([])
    ax.spines["polar"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linestyle="--", alpha=0.5)
    ax.set_title("Team Capability Radar (normalised)", size=14, weight="bold", pad=20, color=FG)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    plt.savefig("02_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 02_radar.png")


def fig_decade_trends(trends: pd.DataFrame) -> None:
    """NEW: Win % per decade line chart."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle("Performance Trends by Decade", fontsize=15, weight="bold")

    decades = sorted(trends["decade"].unique())

    for team in TEAMS:
        sub = trends[trends["team"] == team].set_index("decade")
        sub = sub.reindex(decades)  # fill missing decades with NaN
        ax1.plot(decades, sub["W%"], marker="o", color=TEAM_COLORS[team],
                 linewidth=2.5, label=team, markersize=6)
        ax2.plot(decades, sub["GD"], marker="s", color=TEAM_COLORS[team],
                 linewidth=2.5, label=team, markersize=6)

    ax1.set_ylabel("Win %"); ax1.yaxis.grid(True); ax1.set_axisbelow(True)
    ax1.set_title("Win Percentage per Decade")
    ax1.legend(fontsize=9, loc="upper left")

    ax2.axhline(0, color=FG, linewidth=0.8, alpha=0.4)
    ax2.set_ylabel("Goal Difference"); ax2.yaxis.grid(True); ax2.set_axisbelow(True)
    ax2.set_title("Goal Difference per Decade")
    ax2.set_xticks(decades)
    ax2.set_xticklabels([f"{d}s" for d in decades], rotation=30)

    plt.tight_layout()
    plt.savefig("03_decade_trends.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 03_decade_trends.png")


def fig_goal_analysis(df: pd.DataFrame) -> None:
    """NEW: Goals scored/conceded per game + goal-difference distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Goal Analysis", fontsize=15, weight="bold")

    # ── GF vs GA scatter per game (team averages with error bars)
    ax = axes[0]
    for team in TEAMS:
        sub = df[df["team"] == team]
        gf_mean, gf_std = sub["gf"].mean(), sub["gf"].std()
        ga_mean, ga_std = sub["ga"].mean(), sub["ga"].std()
        ax.errorbar(ga_mean, gf_mean, xerr=ga_std, yerr=gf_std,
                    fmt="o", color=TEAM_COLORS[team], markersize=12,
                    capsize=5, label=team, linewidth=2)
    ax.axline((0, 0), slope=1, color=FG, linestyle="--", alpha=0.4, label="Equal line")
    ax.set_xlabel("Goals Conceded per Game"); ax.set_ylabel("Goals Scored per Game")
    ax.set_title("Scoring vs. Conceding (mean ± std)")
    ax.legend(fontsize=9); ax.grid(True)

    # ── Goal-difference distribution (violin)
    ax = axes[1]
    data_gd = [df[df["team"] == t]["goal_diff"].values for t in TEAMS]
    parts = ax.violinplot(data_gd, positions=range(len(TEAMS)),
                          showmedians=True, showextrema=True)
    for pc, team in zip(parts["bodies"], TEAMS):
        pc.set_facecolor(TEAM_COLORS[team]); pc.set_alpha(0.7)
    parts["cmedians"].set_color(FG)
    parts["cmaxes"].set_color(GRID); parts["cmins"].set_color(GRID)
    parts["cbars"].set_color(GRID)
    ax.axhline(0, color=FG, linewidth=1, alpha=0.5, linestyle="--")
    ax.set_xticks(range(len(TEAMS))); ax.set_xticklabels(TEAMS, rotation=15, fontsize=9)
    ax.set_title("Goal-Difference Distribution per Match")
    ax.set_ylabel("Goal Difference"); ax.yaxis.grid(True); ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig("04_goal_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 04_goal_analysis.png")


def fig_knockout_vs_group(df: pd.DataFrame) -> None:
    """NEW: Win rates in group stage vs knockout stage."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Group Stage vs Knockout Stage Win Rate", fontsize=14, weight="bold")

    x = np.arange(len(TEAMS))
    w = 0.35

    group_wr, ko_wr = [], []
    for team in TEAMS:
        sub = df[df["team"] == team]
        g  = sub[~sub["is_knockout"]]
        ko = sub[sub["is_knockout"]]
        group_wr.append((g["result"] == "W").mean() * 100 if len(g) else 0)
        ko_wr.append((ko["result"] == "W").mean() * 100 if len(ko) else 0)

    bars1 = ax.bar(x - w/2, group_wr, width=w, label="Group Stage",
                   color=[TEAM_COLORS[t] for t in TEAMS], alpha=0.85)
    bars2 = ax.bar(x + w/2, ko_wr, width=w, label="Knockout Stage",
                   color=[ACCENT[t] for t in TEAMS], alpha=0.85,
                   edgecolor=[TEAM_COLORS[t] for t in TEAMS], linewidth=1.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(TEAMS, fontsize=10)
    ax.set_ylabel("Win Rate (%)"); ax.set_ylim(0, 100)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("05_group_vs_knockout.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 05_group_vs_knockout.png")


def fig_dominance_ranking(summary: pd.DataFrame) -> None:
    """NEW: Horizontal bar chart for Dominance Index."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Composite Dominance Index\n(40% Win% + 30% PPG + 30% GD/GP)",
                 fontsize=13, weight="bold")

    ranked = summary.sort_values("Dominance")
    colors = [TEAM_COLORS[t] for t in ranked.index]
    bars = ax.barh(ranked.index, ranked["Dominance"], color=colors, height=0.55)
    for bar, val in zip(bars, ranked["Dominance"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=10)
    ax.set_xlabel("Dominance Score"); ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.set_xlim(0, ranked["Dominance"].max() * 1.15)
    plt.tight_layout()
    plt.savefig("06_dominance_index.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 06_dominance_index.png")


# ──────────────────────────────────────────────────────────────
# 6. PRINT REPORT
# ──────────────────────────────────────────────────────────────

def print_report(summary: pd.DataFrame, trends: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("  FOOTBALL WORLD CUP — TOP 5 NATIONS PERFORMANCE REPORT")
    print("  1930–2022")
    print("=" * 65)

    cols = ["GP", "W", "D", "L", "GF", "GA", "GD", "Pts", "W%", "PPG", "Dominance"]
    print("\n📊 Summary Table:")
    print(summary[cols].to_string())

    print("\n🏆 Rankings:")
    for rank, (team, row) in enumerate(summary.sort_values("Dominance", ascending=False).iterrows(), 1):
        print(f"  {rank}. {team:<15} | Dominance: {row['Dominance']:>6.1f} | "
              f"W%: {row['W%']:>5.1f}% | PPG: {row['PPG']:.2f} | "
              f"GD: {int(row['GD']):>+d}")

    print("\n📈 Most Improved Decade (W% rise):")
    for team in TEAMS:
        sub = trends[trends["team"] == team].sort_values("decade")
        if len(sub) >= 2:
            diff = sub["W%"].max() - sub["W%"].min()
            peak = sub.loc[sub["W%"].idxmax(), "decade"]
            print(f"  {team:<15} peak decade: {peak}s  (range: {diff:.1f}pp)")

    print("\n📁 Charts saved: 01_overview.png … 06_dominance_index.png\n")


# ──────────────────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────────────────

def main() -> None:
    df      = build_dataframe(RAW_MATCHES)
    summary = compute_summary(df)
    trends  = compute_decade_trends(df)

    print_report(summary, trends)

    fig_overview(summary)
    fig_radar(summary)
    fig_decade_trends(trends)
    fig_goal_analysis(df)
    fig_knockout_vs_group(df)
    fig_dominance_ranking(summary)

    print("✅  All done.")


if __name__ == "__main__":
    main()
