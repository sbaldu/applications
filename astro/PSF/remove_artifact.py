
import pandas as pd

def clean_picture(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the picture
    """
    erase = []
    for i in range(0, len(df)):
        if df['x0'][i] < 200 and df['x1'][i] < 200:     # clean the bottom left corner
            y = df['x1'][i]
            x = df['x0'][i]
            if y < 200 - x:
                erase.append(i)
        if df['x1'][i] > 940 and df['x0'][i] < 100:     # clean the top left corner
            y = df['x1'][i]
            x = df['x0'][i]
            if y > 940 + x:
                erase.append(i)
    df = df.drop(erase)
    return df
