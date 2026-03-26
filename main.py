import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def wfh_by_city():
    excel_path = "./data/WFH_TimeSeries.xlsx"
    sheet_name = "WFH by city"
    
    # Mapping to align WFH shorthand names with Zillow Metro names
    city_mapping = {
        "Atlanta": "Atlanta-Sandy Springs-Alpharetta, GA", # modern zillow mapping uses "Alpharetta" instead of "Roswell".
        "BayArea": "San Francisco-Oakland-Berkeley, CA",
        "Chicagoland": "Chicago-Naperville-Elgin, IL-IN-WI",
        "DC": "Washington-Arlington-Alexandria, DC-VA-MD-WV",
        "Dallas": "Dallas-Fort Worth-Arlington, TX",
        "Houston": "Houston-The Woodlands-Sugar Land, TX",
        "LosAngeles": "Los Angeles-Long Beach-Anaheim, CA",
        "Miami": "Miami-Fort Lauderdale-Pompano Beach, FL",
        "NewYork": "New York-Newark-Jersey City, NY-NJ-PA"
    }

    df = pd.read_excel(excel_path, sheet_name=sheet_name, usecols="A, E:M")
    
    # Clean Date Column
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col].dt.year != 2020]
    df.rename(columns={date_col: 'date'}, inplace=True)

    # Clean City Column Names
    df.columns = [col.split('_')[-1] if '_' in str(col) else col for col in df.columns]

    # Melt to Long Format: [date, City_Shorthand, WFH_Score]
    df_long = df.melt(id_vars=['date'], var_name='City_Shorthand', value_name='WFH_Score')
    
    # Map to Zillow Metro Names
    df_long['Metro'] = df_long['City_Shorthand'].map(city_mapping)
    
    # Drop cities not in our mapping.
    df_long = df_long.dropna(subset=["Metro"])
    return df_long

def home_value_index_by_zip():
    csv_path = "./data/ZHVI_ZIP.csv"
    df = pd.read_csv(csv_path)

    id_vars = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName', 
               'State', 'City', 'Metro', 'CountyName']
    
    # Melt: Date columns to rows
    df_long = df.melt(id_vars=id_vars, var_name='date', value_name='ZHVI')
    
    df_long['date'] = pd.to_datetime(df_long['date'])
    df_long = df_long[df_long['date'].dt.year >= 2021]
    return df_long

def merge_and_analyze(wfh_df, zhvi_df):
    # 2. Convert date strings to datetime objects
    wfh_df['date'] = pd.to_datetime(wfh_df['date'])
    zhvi_df['date'] = pd.to_datetime(zhvi_df['date'])

    # 3. Create a temporary 'year_month' column to align the dates
    # This ensures 2021-01-01 (WFH) matches 2021-01-31 (Zillow)
    wfh_df['year_month'] = wfh_df['date'].dt.to_period('M')
    zhvi_df['year_month'] = zhvi_df['date'].dt.to_period('M')

    # 4. Merge the datasets
    # We merge on 'year_month' and 'Metro' so each ZIP code gets 
    # the WFH score for its specific Metro area in that month.
    merged_df = pd.merge(
        zhvi_df, 
        wfh_df[['year_month', 'Metro', 'WFH_Score', 'City_Shorthand']], 
        on=['year_month', 'Metro'], 
    
        how='inner'
    )
    merged_df = merged_df.dropna(subset=['WFH_Score']) # Drops rows where WFH_Score is missing.

    # 5. Apply the Urban/Suburban Classification (as discussed)
    merged_df = classify_urban_suburban(merged_df)
    return merged_df

def classify_urban_suburban(df):
    # Map the Metro name to a LIST of cities that count as 'Urban'. 
    # Rationale: "Los Angelas" and "Long Beach" are both considered urban hubs of California. We can't just say "LA" is the only urban.
    core_cities_map = {
        "Atlanta-Sandy Springs-Alpharetta, GA": ["Atlanta"], # modern zillow mapping uses "Alpharetta" instead of "Roswell".
        "San Francisco-Oakland-Berkeley, CA": ["San Francisco", "Oakland", "Berkeley"],
        "Chicago-Naperville-Elgin, IL-IN-WI": ["Chicago"],
        "Washington-Arlington-Alexandria, DC-VA-MD-WV": ["Washington", "Alexandria", "Arlington"],
        "Dallas-Fort Worth-Arlington, TX": ["Dallas", "Fort Worth", "Arlington"],
        "Houston-The Woodlands-Sugar Land, TX": ["Houston"],
        "Los Angeles-Long Beach-Anaheim, CA": ["Los Angeles", "Long Beach", "Anaheim", "Santa Ana"],
        "Miami-Fort Lauderdale-Pompano Beach, FL": ["Miami", "Fort Lauderdale"],
        "New York-Newark-Jersey City, NY-NJ-PA": ["New York", "Newark", "Jersey City"]
    }

    # Default everything to Suburban
    df['LocationType'] = 'Suburban'

    # Logic: If the 'City' is in the list for that specific 'Metro', label as Urban
    def check_urban(row):
        if pd.isna(row["City"]):
            return 'Suburban'
        
        metro_name = row['Metro']
        city_name = row['City']
        
        # Get the list of urban cities for this metro, default to empty list if not found
        urban_list = core_cities_map.get(metro_name, [])

        return "Urban" if city_name in urban_list else "Suburban"

    df['LocationType'] = df.apply(check_urban, axis=1)
    
    return df

def calculate_price_growth_index(df):
    """
    Calculates the average Home Value (ZHVI) growth index for Urban vs Suburban areas
    FOR EACH CITY, preserving the WFH_Score for analysis.
    """
    # 1. Get the WFH_Score for each city and month 
    # (Since it's the same for all ZIPs in a city for a given month, mean() works)
    wfh_scores = df.groupby(['City_Shorthand', 'date'])['WFH_Score'].mean().reset_index()
    
    # 2. Group by City, Date, AND LocationType to get the average ZHVI
    avg_zhvi = df.groupby(['City_Shorthand', 'date', 'LocationType'])['ZHVI'].mean().reset_index()
    
    # 3. Pivot so Dates & Cities are the index, and 'Urban'/'Suburban' are columns
    pivot_df = avg_zhvi.pivot(index=['City_Shorthand', 'date'], columns='LocationType', values='ZHVI')
    
    # 4. Calculate the Growth Index (Base Month = 100) PER CITY
    # This sets the first month for EACH city as the baseline
    growth_index = pivot_df.groupby(level='City_Shorthand').transform(lambda x: (x / x.iloc[0]) * 100)
    
    # 5. Reset index and merge WFH scores back in
    growth_index = growth_index.reset_index()
    final_df = pd.merge(growth_index, wfh_scores, on=['City_Shorthand', 'date'], how='left')
    
    return final_df

def generate_wfh_impact_report(growth_df):
    """
    Takes the growth_index DataFrame and creates a cross-city 
    comparison of WFH scores vs. Suburban Growth.
    """
    # 1. Calculate the overall average WFH score for each city
    wfh_stats = growth_df.groupby('City_Shorthand')['WFH_Score'].mean().reset_index()

    # 2. Extract the "Final Snapshot" (The most recent data point for each city)
    latest_date = growth_df['date'].max()
    final_snapshot = growth_df[growth_df['date'] == latest_date].copy()

    # 3. Merge latest indices with the average WFH stats
    report = pd.merge(final_snapshot, wfh_stats, on='City_Shorthand', suffixes=('_monthly', '_avg'))

    # 4. Calculate the "Donut Effect Gap" 
    # (Suburban growth % minus Urban growth %)
    report['Donut_Gap'] = report['Suburban'] - report['Urban']

    # 5. Clean up and Rank
    report = report[['City_Shorthand', 'WFH_Score_avg', 'Suburban', 'Urban', 'Donut_Gap']]
    report = report.rename(columns={
        'City_Shorthand': 'City',
        'WFH_Score_avg': 'Avg_WFH_Score',
        'Suburban': 'Total_Suburban_Growth',
        'Urban': 'Total_Urban_Growth'
    })

    # 6. Round numeric columns to 2 decimal places.
    numeric_cols = ['Avg_WFH_Score', 'Total_Suburban_Growth', 'Total_Urban_Growth', 'Donut_Gap']
    report[numeric_cols] = report[numeric_cols].round(2)

    return report.sort_values(by='Donut_Gap', ascending=False)

def visualization_wfh_vs_donut(df):
    # Create a fresh figure
    plt.figure(figsize=(10, 6))
    
    # Scatter points
    plt.scatter(df['Avg_WFH_Score'], df['Donut_Gap'], color='#2c3e50', s=100, label='City Data Points', zorder=3)

    # Calculate & plot trendline
    x = df['Avg_WFH_Score'].values
    y = df['Donut_Gap'].values
    m, b = np.polyfit(x, y, 1)
    
    # Generate points for the trendline
    plt.plot(x, m*x + b, color='#e74c3c', linestyle='--', linewidth=2, label=f'Trendline (y={m:.2f}x + {b:.2f})')

    # Annotate each city
    for i, row in df.iterrows():
        plt.annotate(row['City'], (row['Avg_WFH_Score'], row['Donut_Gap']), 
                    xytext=(8, 0), textcoords='offset points', fontsize=9)

    # Labels and Aesthetics
    plt.title('Correlation: WFH Intensity vs. The Donut Effect Gap', fontsize=14, pad=15)
    plt.xlabel('Average WFH Intensity Score', fontsize=12)
    plt.ylabel('Donut Gap (Suburban Growth - Urban Growth)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig('./outputs/visualizations/wfh_vs_donut_correlation.png')
    plt.close()

def visualization_urban_vs_suburban_home_value(df):
    plt.figure(figsize=(12, 6)) 
    df_growth = df.sort_values(by='Total_Suburban_Growth', ascending=False)
    x = np.arange(len(df_growth['City']))
    
    plt.bar(x - 0.2, df_growth['Total_Suburban_Growth'], 0.4, label='Suburban', color='skyblue')
    plt.bar(x + 0.2, df_growth['Total_Urban_Growth'], 0.4, label='Urban', color='navy')
    
    plt.xticks(x, df_growth['City'], rotation=45)
    plt.ylabel('Growth Index (Base 100)')
    plt.legend()
    plt.title('Suburban vs Urban Growth Index Comparison')
    
    plt.tight_layout()
    plt.savefig('./outputs/visualizations/growth_comparison.png')
    plt.close()

def donut_effect_by_city(df):
    plt.figure(figsize=(10, 6))
    df_sorted = df.sort_values(by='Donut_Gap', ascending=True)
    
    plt.barh(df_sorted['City'], df_sorted['Donut_Gap'], color='salmon')
    plt.title('Ranking the "Donut Effect" Magnitude (Suburban Outperformance)')
    plt.xlabel('Donut Gap (Percentage Points)')
    
    plt.tight_layout()
    plt.savefig('./outputs/visualizations/donut_gap_ranking.png')
    plt.close()

def regression(impact_report_df):
    # Extract data points.
    x = impact_report_df['Avg_WFH_Score']
    y = impact_report_df['Donut_Gap']

    # Run the regression: Ordinary Least Squares.
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Output results to txt file.
    summary = (
            f"Regression Results\n"
            f"{'-' * 20}\n"
            f"Beta (WFH Impact): {slope:.4f}\n"
            f"R-squared:         {r_value**2:.4f}\n"
            f"P-value:           {p_value:.4f}\n"
        )    
    with open("./outputs/regression.txt", "w") as f:
        f.write(summary)
    
    print("Results saved to regression_summary.txt")
    print(summary)
        

def main():
    print("Starting analysis...")

    # 1. Load and format the WFH scores
    wfh_data_df_long = wfh_by_city()
    wfh_data_df_long.to_csv("./outputs/wfh_data.csv", index=False)
    
    # 2. Load and melt the Zillow housing data
    zhvi_data_df_long = home_value_index_by_zip()
    zhvi_data_df_long.to_csv("./outputs/zhvi_data.csv", index=False)

    # 3. Merge Databases
    merged_df = merge_and_analyze(wfh_data_df_long, zhvi_data_df_long)
    merged_df.to_csv("./outputs/merged_df.csv", index=False)
    
    # 4. Calcualte the price growth index
    growth_index = calculate_price_growth_index(merged_df)
    growth_index.to_csv('./outputs/city_growth_index.csv', index=False)

    # 5. Correlation results between WFH and housing.
    impact_report_df = generate_wfh_impact_report(growth_index)
    impact_report_df.to_csv('./outputs/FINAL_REPORT.csv', index=False)
    impact_report_df.to_excel('./outputs/FINAL_REPORT.xlsx')

    # 6. Visualization: wfh vs. donut effect.
    visualization_wfh_vs_donut(impact_report_df)

    # 7. Visualization: Urban vs. Suburban Home Growth.
    visualization_urban_vs_suburban_home_value(impact_report_df)

    # 8. Visualization: Donut Effect by City.
    donut_effect_by_city(impact_report_df)

    # 9. Regression
    regression(impact_report_df)

    print("Analysis Complete. See 'outputs' folder for data & visualizations :)")

if __name__ == "__main__":
    main()