
import pandas as pd
from collections import Counter


# helper function to count the most frequent attributes:
def count_most_freq(df: pd.DataFrame) -> dict:
   df = pd.read_csv(df)
   cleanList = []
   for i in df['attrs']:
      crop = i.split('2')[0]
      cleanList.append(crop)

   return Counter(cleanList)

#  #(cfg[test]) helper function to subset the most frequent attributes: