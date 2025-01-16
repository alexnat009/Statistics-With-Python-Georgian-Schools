import numpy as np
import pandas as pd
import re


def preprocess_name(name, remove_punctuation=True, remove_english=True, remove_whitespace=True,
                    replace_n=True):
    if remove_punctuation:
        name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    if remove_whitespace:
        name = re.sub(r'\s+', ' ', name).strip()  # Normalize whitespace
    if remove_english:
        name = re.sub(r'[a-zA-Z]', '', name)  # remove any english characters
    if replace_n:
        name = re.sub(r'\b[Nn](\d+)', r'№\1', name)
    return name


def compare_columns(df, col1, col2):
    return np.where(
        (df[f'{col1}'] != df[f'{col2}']) &
        ~(pd.isna(df[f'{col1}']) & pd.isna(df[f'{col2}']))
    )
