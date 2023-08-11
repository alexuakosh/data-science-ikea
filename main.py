import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from cleaning import df_cleaned
from ml_model import Regressors

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


create_barplot(df_cleaned, 'category', 5, PLOT_COLORS[0:5])
create_barplot(df_cleaned, 'sellable_online', 2, PLOT_COLORS[0:2])
create_barplot(df_cleaned, 'other_colors', 2, PLOT_COLORS[-4::-1])

print('depth availability:')
print(len(df_cleaned['depth'].dropna()) / len(df_cleaned['depth']))  # about 62 % of depth availability
print('height availability:')
print(len(df_cleaned['height'].dropna()) / len(df_cleaned['height']))  # about 75 % of  height availability
print('width availability:')
print(len(df_cleaned['width'].dropna()) / len(df_cleaned['width']))  # about 85 % of width availability
print('====================================================================================')
""" Despite that percentage of availability for depth is quite small (62%) and imputation
without losing of accuracy of overall data distribution is less likely there is some sense 
to consider product volume (height * width * depth) to estimate for instance 
if quantity of materials required to produce this product affects its price."""

old_price_availability = len(df_cleaned[df_cleaned['old_price'] != 0]) / len(
    df_cleaned)  # about 19 percent of old price availability
df_with_old_price = df_cleaned[df_cleaned['old_price'] != 0]

df_with_old_price.loc[:, ['discount']] = (df_with_old_price['old_price'].astype(float) -
                                 df_with_old_price['price'].astype(float)) / df_with_old_price['old_price'].astype(
    float)
avg_discount = df_with_old_price.groupby(['category'])['discount'].mean()
avg_discount = avg_discount.sort_values(ascending=False)
print('Product price discounts (old_price - price) by category:')
print(avg_discount)
print('===============================================================================')
plt.bar(avg_discount.head(5).index, avg_discount.head(5).values, color=PLOT_COLORS)  # categories with the largest discount
plt.xlabel('categories with the biggest discount'.upper())
plt.ylabel('average category discount'.upper())
plt.show()


"""3. BASED ON EDA AND COMMON SENSE CHOSE 2 HYPOTHESIS WHICH YOU'D LIKE TO TEST/ANALYZE.
FOR EACH HYPOTHESIS DECLARE NULL-HYPOTHESIS AND ALTERNATIVE HYPOTHESES, DEVELOP TESTS TO CHECK THEM.
DESCRIBE RESULTS.
"""
"""3.1 NULL-HYPOTHESIS ONE: Size(volume) of product affects its price. Alternatively -
Sizes don't correlate with prices."""
"""3.1.1 Value imputation"""
df_volume = df_cleaned[['depth', 'height', 'width']]
imputer = KNNImputer(n_neighbors=7)  # There are several categories with amount of products equals 7 and less
predicted_size_parameters = imputer.fit_transform(df_volume)

""" 3.1.2 Calculation of product sizes(volume) and finding correlation"""
product_volumes = [sublist[0] * sublist[1] * sublist[2] for sublist in predicted_size_parameters]
print("Correlation coefficient between product size(volume) and its price: ")
print(pearsonr(df_cleaned['price'].tolist(), product_volumes))
print('=============================================================================')
df_volumes = pd.DataFrame({'volume': product_volumes, 'price': df_cleaned['price'].tolist()})
sns.lmplot(x="volume", y="price", data=df_volumes)
plt.show()
"""The correlation coefficient is about 0.72, obviously there is some (not very strong) correlation
bet product size(volume) and its price."""

"""3.2 NULL-HYPOTHESIS TWO: Availability of additional colors affects its price. Alternatively - 
availability of additional color doesn't affect prices."""
# and
"""3.3 NULL-HYPOTHESIS THREE: Group of designers creates more expensive products than single designer.
Alternatively - number of designers doesn't affect prices."""
categories = set(df_cleaned['category'].tolist())


def define_designer_type(row):
    if '/' in row:
        return 'Group'
    else:
        return 'Individual'


df_cleaned['designer_type'] = df_cleaned['designer'].apply(define_designer_type)

for category in categories:
    df_by_category = df_cleaned[df_cleaned['category'] == category]
    df_colors = df_by_category.groupby('other_colors').agg(mean_price=('price', 'mean'))
    df_designers = df_by_category.groupby('designer_type').agg(mean_price=('price', 'mean'))
    fig, ax = plt.subplots(1, 2)
    if len(df_colors.index) == 2:  # creating the graph of mean prices depended on other colors availability
        ax[0].bar(['No', 'Yes'], [df_colors.loc['No']['mean_price'], df_colors.loc['Yes']['mean_price']], color=['red', 'blue'])
        ax[0].set_title('Mean prices depended on other colors availability')
    if len(df_designers.index) == 2:  # creating the graph of mean prices depended on designer type (group/individual)
        ax[1].bar(['Group', 'Individual'], [df_designers.loc['Group']['mean_price'], df_designers.loc['Individual']['mean_price']], color=['red', 'blue'])
        ax[1].set_title('Mean prices depended on designer type')
    plt.xlabel(category.upper())
    plt.ylabel('Mean price'.upper())
    plt.show()

"""3.2 As we see different categories have large variety in mean price of products with and without
additional colors. We cannot say definitely that additional colors affect prices.
3.3 Influence of group or individual designers to the prices are also not so obvious but maybe group of
designers in some cases (maybe in particular categories of products) increases prices.

P.S. There is NO sense to research if prices of products that sellable online differ from products that are
not sellable online considering that there are to few products that are not sellable online and we
cannot form relevant samples."""

"""4. TEACH THE MODEL TO PREDICT PRICE.
 - DEFINE WHICH COLUMNS SHOULD NOT BE INCLUDED TO THE MODEL AND WHY.
 - CREATE THE PIPELINE FOR CLEANING AND TEACHING MODEL AND ESTIMATING MODEL PERFORMANCE,
 INCLUDING (IF NECESSARY) SUCH STEPS AS IMPUTATION OF MISSING VALUES AND NORMALIZATION.
 - SUGGEST METHODS FOR INCREASING OF MODEL PERFORMANCE. DESCRIBE RESULTS."""

regressors = Regressors(df_cleaned)
regressors.train()
# creates and presents model described in file ml_model.py
"""4.2 Overall model performance is better when using Random Forest Regression. Model score is approx. 0.8
It would be definitely better if we had more relevant data to analyze, first of all it would be great
if we had cost prices of products, or at least information that could point on it 
(materials, information about man hours of production, marketing costs regarding to the specific
products, etc.)"""

