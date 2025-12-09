import pandas as pd

posts = pd.read_csv("datasets/posts.csv")
stock_index = pd.read_csv("datasets/stock_index.csv")

merged = pd.merge(
    posts,
    stock_index[['id', 'stock_symbol']],
    on='id',
    how='left'
)

final_df = merged[['id', 'created_utc', 'stock_symbol', 'title', 'selftext']]
final_df.to_csv("posts_with_symbols.csv", index=False)
