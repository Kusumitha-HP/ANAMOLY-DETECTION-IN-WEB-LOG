import pandas as pd
import plotly.express as px

# Load data
logs = pd.read_csv('data/simulated_logs.csv', parse_dates=['timestamp'])

# Count events per product per minute
product_counts = logs.groupby([pd.Grouper(key='timestamp', freq='1Min'), 'product_id']).size().unstack(fill_value=0)

# Identify products with sudden spikes
threshold = product_counts.mean() + 2*product_counts.std()
trending_products = (product_counts > threshold).sum(axis=0)
trending_products = trending_products[trending_products > 0].sort_values(ascending=False)

print("Trending / booming products detected:")
print(trending_products)

# Save recommendations
top_trending = trending_products.head(5).index.tolist()
recommendations = pd.DataFrame({'recommended_products': top_trending})
recommendations.to_csv('data/recommendations.csv', index=False)
print("\n✅ Top 5 recommended products saved to data/recommendations.csv")

# Convert trending products to DataFrame for plotting
trending_df = trending_products.reset_index()
trending_df.columns = ['product_id', 'spike_count']

# Create bar chart
fig = px.bar(trending_df.head(10), 
             x='product_id', 
             y='spike_count', 
             color='spike_count',
             text='spike_count',
             title='Top Trending / Booming Products',
             labels={'product_id':'Product ID','spike_count':'Spike Count'},
             color_continuous_scale='Viridis')

fig.update_traces(textposition='outside')
fig.update_layout(
    yaxis=dict(dtick=1),
    xaxis=dict(title='Product ID'),
    yaxis_title="Spike Count",
    title_x=0.5
)

fig.show()
