import os
import json
import pandas as pd

# fname = "./raw_data/collection_issuer-directory_20250123.json"
fname = "./raw_data/clean-newsletter.json"
dic = json.load(open(fname))

df = pd.DataFrame.from_dict(dic)
print(df.columns)

df['date'] = pd.to_datetime(df['created_date'], format='mixed').dt.date

# Sort the DataFrame by date
df_sorted = df.sort_values('date')

# Group by date
df_grouped = df_sorted.groupby('date')

# Get rows for specific date
date_to_filter = pd.to_datetime('2024-09-19').date()
filtered_df = df_sorted[df_sorted['date'] == date_to_filter]
# print(filtered_df)

last_row = filtered_df.iloc[-3].to_dict()
# Convert date objects to string for JSON serialization
last_row['date'] = str(last_row['date'])
json_output = json.dumps(last_row, indent=3)
print(json_output)

fname = f"news_{last_row['id']}.txt"
text = f"title: {last_row['title']}\ncontent: {last_row['content']}"
os.makedirs("news", exist_ok=True)
o = f"news/{fname}"
with open(o, "w+") as f:
    f.write(text)
print(f"news text is saved to: {o}")
