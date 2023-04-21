import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from scipy.stats import pearsonr
from cleaning import df_cleaned

"""2. MAKE EXPLORATORY DATA ANALYSIS, INCLUDING DESCRIPTIVE STATISTICS AND VISUALIZATIONS.
DESCRIBE RESULTS."""
"""2.2 Descriptive analysis"""
PLOT_COLORS = ['green', 'red', 'yellow', 'black', 'orange']

def create_barplot(df_arg, df_column, amount_of_values, colors):
    column_count = df_arg[df_column].value_counts()
    x = column_count.head(amount_of_values).index.tolist()
    y = column_count.head(amount_of_values).values
    plt.bar(x, y, color=colors)
    plt.xlabel(f'{df_column}'.upper())
    plt.ylabel(f'Amount of items in column "{df_column}"'.upper())
    plt.show()

# create_barplot(df_cleaned, 'category', 5, PLOT_COLORS[0:5])
# create_barplot(df_cleaned, 'sellable_online', 2, PLOT_COLORS[0:2])
# create_barplot(df_cleaned, 'other_colors', 2, PLOT_COLORS[-4::-1])

print(len(df_cleaned['depth'].dropna())/len(df_cleaned['depth']))  # about 62 % of availability
print(len(df_cleaned['height'].dropna())/len(df_cleaned['height']))  # about 75 % of availability
print(len(df_cleaned['width'].dropna())/len(df_cleaned['width']))  # about 85 % of availability
""" Despite that percentage of availability for depth is quite small (62%) and imputation
without losing of accuracy of overall data distribution is less likely there is some sense 
to consider product volume (height * width * depth) to estimate for instance 
if quantity of materials required to produce this product affects its price."""

# old_price_availability = len(df_cleaned[df_cleaned['old_price'] != 0])/len(df_cleaned) # about 19 percent of old price availability
# df_with_old_price = df_cleaned[df_cleaned['old_price'] != 0]
#
# df_with_old_price['discount'] = (df_with_old_price['old_price'].astype(float) -
#                                  df_with_old_price['price'].astype(float)) / df_with_old_price['old_price'].astype(float)
# avg_discount = df_with_old_price.groupby(['category'])['discount'].mean()
# avg_discount = avg_discount.sort_values(ascending=False)
# print(avg_discount)
# plt.bar(avg_discount.head(5).index, avg_discount.head(5).values, color=PLOT_COLORS)  # categories with the largest discount
# plt.xlabel('categories with the biggest discount'.upper())
# plt.ylabel('average category discount'.upper())
# plt.show()


"""3. BASED ON EDA AND COMMON SENSE CHOSE 2 HYPOTHESIS WHICH YOU'D LIKE TO TEST/ANALYZE.
FOR EACH HYPOTHESIS DECLARE NULL-HYPOTHESIS AND ALTERNATIVE HYPOTHESES, DEVELOP TESTS TO CHECK THEM.
DESCRIBE RESULTS.
"""
"""3.1 HYPOTHESIS ONE: Size(volume) of product affects its price"""
"""3.1.1 Value imputation"""
df_volume = df_cleaned[['depth', 'height', 'width']]
imputer = KNNImputer(n_neighbors=7)  # There are several categories with amount of products equals 7 and less
predicted_size_parameters = imputer.fit_transform(df_volume)

""" 3.1.2 Calculation of product sizes(volume) and finding correlation"""
product_volumes = [sublist[0] * sublist[1] * sublist[2] for sublist in predicted_size_parameters]
print(pearsonr(df_cleaned['price'].tolist(), product_volumes))
df_new = pd.DataFrame({'volume': product_volumes, 'price': df_cleaned['price'].tolist()})
sns.lmplot(x="volume", y="price", data=df_new)
plt.show()
"""The correlation coefficient is about 0.72, obviously there is some (not very strong) correlation
bet product size(volume) and its price."""





