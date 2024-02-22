# Let's attempt to answer all the questions, starting by re-loading the dataset and then proceeding to each question.
# Since the dataset was loaded earlier, I'll proceed directly with analysis and visualization for each question.

# Re-defining necessary imports and data loading in case the environment has been reset.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is already loaded and preprocessed, but re-loading for completeness
df = pd.read_csv('avocado.csv')

# Answer 1: Average avocado prices vary by year
avg_price_by_year = df.groupby('year')['AveragePrice'].mean().reset_index()

# Answer 2: Distribution of avocado sales volume by type
sales_volume_by_type = df.groupby('type')['Total Volume'].sum().reset_index()

# Answer 3: Average price of avocados vary by region
avg_price_by_region = df.groupby('region')['AveragePrice'].mean().reset_index().sort_values(by='AveragePrice', ascending=False)

# Answer 4: Trend in the size of bags sold over time
bags_sold_over_time = df.groupby('year').agg({
    'Small Bags': 'sum',
    'Large Bags': 'sum',
    'XLarge Bags': 'sum'
}).reset_index()

# Answer 5: Relationship between total volume sold and average price
# For this, we'll use the entire dataset without grouping, directly in the scatter plot.

# Now, plotting all the visualizations
plt.figure(figsize=(18, 10))

# Plot 1: Average Price by Year
plt.subplot(2, 3, 1)
plt.plot(avg_price_by_year['year'], avg_price_by_year['AveragePrice'], marker='o')
plt.title('Average Price by Year')
plt.xlabel('Year')
plt.ylabel('Average Price ($)')

# Plot 2: Sales Volume by Type
plt.subplot(2, 3, 2)
sns.barplot(x='type', y='Total Volume', data=sales_volume_by_type)
plt.title('Sales Volume by Type')
plt.xlabel('Type')
plt.ylabel('Total Volume')

# Plot 3: Average Price by Region
plt.subplot(2, 3, 3)
sns.barplot(x='AveragePrice', y='region', data=avg_price_by_region.head(10), orient='h')
plt.title('Top 10 Regions by Average Price')
plt.xlabel('Average Price ($)')
plt.ylabel('Region')

# Plot 4: Bags Sold Over Time
plt.subplot(2, 3, 4)
plt.plot(bags_sold_over_time['year'], bags_sold_over_time['Small Bags'], label='Small Bags')
plt.plot(bags_sold_over_time['year'], bags_sold_over_time['Large Bags'], label='Large Bags')
plt.plot(bags_sold_over_time['year'], bags_sold_over_time['XLarge Bags'], label='XLarge Bags')
plt.title('Bags Sold Over Time')
plt.xlabel('Year')
plt.ylabel('Total Bags Sold')
plt.legend()

# Plot 5: Relationship Between Total Volume Sold and Average Price
plt.subplot(2, 3, 5)
plt.scatter(df['Total Volume'], df['AveragePrice'])
plt.title('Total Volume Sold vs. Average Price')
plt.xlabel('Total Volume')
plt.ylabel('Average Price ($)')
plt.xscale('log')  # Using log scale due to wide distribution of total volume

plt.tight_layout()
plt.show()
