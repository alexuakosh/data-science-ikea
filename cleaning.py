import pandas as pd
import re

"""1. DOWNLOAD THE DATASET IKEA"""
df = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


"""2. MAKE EXPLORATORY DATA ANALYSIS, INCLUDING DESCRIPTIVE STATISTICS AND VISUALIZATIONS.
DESCRIBE RESULTS."""

"""2.1 Cleaning data"""
"""2.1.1 Dropping duplicates (orientation to 'item_id), deleting irrelevant columns,
normalizing categories to most common, and grouping products by categories"""
print(len(df) - len(df['item_id'].unique()))  # There is some duplicates of item_ids
counted_ids = df['item_id'].value_counts()
duplicates = counted_ids[counted_ids.values != 1]
duplicated_ids = duplicates.index.tolist()
df_duplicates = pd.DataFrame()
df_cleaned = df
df_variables_difference = pd.DataFrame()

for item_id in duplicated_ids:
    df_cleaned = df_cleaned[df_cleaned['item_id'] != item_id]
    df_items = df[df['item_id'] == item_id]
    var_difference = dict()
    most_common_category = ''

    for column in df_items.columns.tolist():  # create difference table between items with the same item_id
        if df_items[column].nunique() > 1:
            var_difference[column] = [1]
        else:
            var_difference[column] = [0]

    for index, row in df_items.iterrows():
        if len(df[df['category'] == row['category']]) > len(df[df['category'] == most_common_category]):
            most_common_category = row['category']
        df_items.loc[:, ['category']] = most_common_category  # assign most common category to duplicates

    df_variables_difference = pd.concat([df_variables_difference, pd.DataFrame(var_difference)])
    df_duplicates = pd.concat([df_duplicates, df_items], axis='index')

# print(df_variables_difference)
"""As we can see, product items with the same item_id have the difference only in Categories.
It means that most likely there was mistake of people responsible for accounting of products.
They classified the same product to different categories. Maybe there is some need to standardize
categories in the company."""

df_duplicates = df_duplicates.drop_duplicates(subset=['item_id'], keep='first')  # drop duplicates of products
df_cleaned = pd.concat([df_cleaned, df_duplicates], axis='index')
df_cleaned = df_cleaned.drop(columns=['link', 'short_description'])
df_cleaned = df_cleaned.sort_values(by=['category'], ascending=True)  # grouping by category is more relevant for further analysis
"""It seems there is some sense to replace the category values in duplicates with the most common values.
As a result we have dataset with unique product items and most relevant category values."""

"""2.1.2 Normalizing old price values"""
def normalize_price(price):
    norm_price = 0
    if ',' in price:
        price = price.replace(',', '')
    if re.search(r'\d+', price) is not None:
        norm_price = re.search(r'\d+', price)[0]
    return norm_price

df_cleaned['old_price'] = df_cleaned['old_price'].apply(normalize_price)