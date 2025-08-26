# disease_outbreak_prediction/src/data_collection/health_data_collector.py

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time


class HealthDataCollector:
    """
    Collects public health data from various official sources
    """

    def __init__(self):
        self.who_data = []
        self.cdc_data = []
        self.owid_data = []

    def get_who_data(self, indicator_codes=None):
        """
        Collect data from WHO Global Health Observatory
        """
        base_url = "https://ghoapi.azureedge.net/api"

        if indicator_codes is None:
            # Common disease indicators
            indicator_codes = [
                'INFLUENZA_A',  # Influenza A
                'WHS4_544',     # Acute respiratory infections
                'MALARIA_EST_DEATHS',  # Malaria deaths
                'TB_c_newinc'   # Tuberculosis incidence
            ]

        who_data = []

        for indicator in indicator_codes:
            try:
                url = f"{base_url}/{indicator}"
                response = requests.get(url)

                if response.status_code == 200:
                    data = response.json()

                    for item in data.get('value', []):
                        record = {
                            'indicator': indicator,
                            'country': item.get('SpatialDim'),
                            'year': item.get('TimeDim'),
                            'value': item.get('NumericValue'),
                            'display_value': item.get('Value'),
                            'source': 'WHO'
                        }
                        who_data.append(record)

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"Error fetching WHO data for {indicator}: {e}")

        self.who_data = who_data
        return who_data

    def get_owid_covid_data(self, countries=None):
        """
        Get COVID-19 data from Our World in Data
        """
        url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"

        try:
            df = pd.read_csv(url)

            if countries:
                df = df[df['location'].isin(countries)]

            # Convert to list of dictionaries
            covid_data = df.to_dict('records')

            # Add source information
            for record in covid_data:
                record['source'] = 'OWID'

            self.owid_data = covid_data
            return covid_data

        except Exception as e:
            print(f"Error fetching OWID data: {e}")
            return []

    def get_disease_outbreak_news(self, sources=None):
        """
        Collect disease outbreak news from reliable sources
        Note: This is a simplified version - you might want to use News API
        """
        if sources is None:
            sources = [
                'https://www.who.int/emergencies/disease-outbreak-news',
                'https://www.cdc.gov/outbreaks/',
                'https://www.ecdc.europa.eu/en/threats-and-outbreaks'
            ]

        outbreak_news = []

        # This is a placeholder - you would implement web scraping here
        # For a real implementation, consider using:
        # - News API (newsapi.org)
        # - BeautifulSoup for web scraping
        # - RSS feeds from health organizations

        sample_news = [
            {
                'title': 'Influenza outbreak reported in multiple regions',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'source': 'WHO',
                'url': 'https://who.int/example',
                'description': 'Health authorities report increased influenza cases',
                'location': 'Global',
                'disease': 'Influenza'
            }
        ]

        return sample_news

    def get_surveillance_data(self, disease_types=None):
        """
        Simulate surveillance data collection
        """
        if disease_types is None:
            disease_types = ['influenza', 'covid-19', 'measles', 'dengue']

        # Generate synthetic surveillance data for demonstration
        surveillance_data = []

        for disease in disease_types:
            for i in range(52):  # 52 weeks of data
                week_date = datetime.now() - timedelta(weeks=i)

                record = {
                    'disease': disease,
                    'week': week_date.isocalendar()[1],
                    'year': week_date.year,
                    'cases': np.random.poisson(lam=50 + np.random.normal(0, 10)),
                    'deaths': np.random.poisson(lam=2),
                    'hospitalizations': np.random.poisson(lam=10),
                    'region': 'National',
                    'data_quality': 'Good',
                    'source': 'National_Surveillance'
                }
                surveillance_data.append(record)

        return surveillance_data

    def get_environmental_data(self):
        """
        Collect environmental factors that might influence disease spread
        """
        # This would typically connect to weather APIs, air quality APIs, etc.
        # For demonstration, creating synthetic data

        environmental_data = []

        for i in range(365):  # Daily data for a year
            date = datetime.now() - timedelta(days=i)

            record = {
                'date': date.strftime('%Y-%m-%d'),
                'temperature': 20 + 10 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 2),
                'humidity': 60 + 20 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 5),
                'precipitation': max(0, np.random.exponential(2)),
                'air_quality_index': np.random.normal(50, 15),
                'uv_index': max(0, 5 + 3 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 1)),
                'source': 'Environmental_API'
            }
            environmental_data.append(record)

        return environmental_data

    def create_health_features(self, health_data):
        """
        Create features from health data for ML models
        """
        df = pd.DataFrame(health_data)

        if 'date' not in df.columns and 'year' in df.columns:
            df['date'] = pd.to_datetime(df['year'], format='%Y')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Sort by date
        df = df.sort_values('date')

        # Create time-based features
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week

        # Create seasonal indicators
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })

        return df

    def combine_all_data(self):
        """
        Combine all collected health data sources
        """
        combined_data = {
            'who_data': self.who_data,
            # Limit size
            'owid_data': self.owid_data[:1000] if len(self.owid_data) > 1000 else self.owid_data,
            'surveillance_data': self.get_surveillance_data(),
            'environmental_data': self.get_environmental_data(),
            'outbreak_news': self.get_disease_outbreak_news()
        }

        return combined_data

    def save_health_data(self, filename='health_data.json'):
        """
        Save all health data to file
        """
        all_data = self.combine_all_data()

        with open(filename, 'w') as f:
            json.dump(all_data, f, default=str, indent=2)

        print(f"Health data saved to {filename}")
        return all_data


# Example usage
if __name__ == "__main__":
    collector = HealthDataCollector()

    # Collect WHO data
    print("Collecting WHO data...")
    who_data = collector.get_who_data()
    print(f"Collected {len(who_data)} WHO records")

    # Collect COVID data (limited to first 1000 records for demo)
    print("Collecting OWID COVID data...")
    covid_data = collector.get_owid_covid_data(
        ['United States', 'United Kingdom', 'Germany'])
    print(f"Collected {len(covid_data)} COVID records")

    # Get surveillance data
    print("Generating surveillance data...")
    surveillance_data = collector.get_surveillance_data()
    print(f"Generated {len(surveillance_data)} surveillance records")

    # Save all data
    all_health_data = collector.save_health_data()

    # Create features from surveillance data
    surveillance_df = collector.create_health_features(surveillance_data)
    print(f"Created features for {len(surveillance_df)} records")
    print(surveillance_df.head())
