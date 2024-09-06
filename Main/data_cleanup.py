# Importando bibliotecas necessárias
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import datetime as dt

# Importando o dataset
filename = '../Data/listings.csv'
data = pd.read_csv(filename)

# Removendo colunas indesejadas
data = pd.DataFrame.drop(data, columns=[
    'host_name', 'notes', 'host_about', 'calendar_updated', 'host_acceptance_rate',
    'description', 'thumbnail_url', 'experiences_offered', 'listing_url', 'name', 
    'summary', 'space', 'scrape_id', 'last_scraped', 'neighborhood_overview', 'transit',
    'access', 'interaction', 'house_rules', 'medium_url', 'picture_url', 'xl_picture_url',
    'host_url', 'host_thumbnail_url', 'host_picture_url', 'host_acceptance_rate', 
    'smart_location', 'license', 'jurisdiction_names', 'street', 'neighbourhood', 
    'country', 'country_code', 'host_location', 'host_neighbourhood', 'market', 
    'is_location_exact', 'square_feet', 'weekly_price', 'monthly_price', 'availability_30', 
    'availability_60', 'availability_90', 'availability_365', 'calendar_last_scraped', 
    'first_review', 'last_review', 'requires_license', 'calculated_host_listings_count', 
    'host_listings_count', 'zipcode'
])

# Manipulação da coluna 'host_verifications'
host_verification_set = set()

def collect_host_verifications(entry):
    if isinstance(entry, str):
      entry_list = entry.replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace(" ", "").split(',')
      for verification in entry_list:
          if (verification != "" and verification != 'None'):
              host_verification_set.add(verification + "_verification")

data['host_verifications'].apply(collect_host_verifications)

def generic_verification(entry, v):
    entry_list = str(entry).replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace(" ", "").split(',')
    for verification in entry_list:
        if (verification + "_verification" == v):
            return 1
    return 0

for v in host_verification_set:
    data.insert(len(list(data)), v, 0)
    data[v] = data['host_verifications'].apply(lambda x: generic_verification(x, v))

data = pd.DataFrame.drop(data, columns=['host_verifications'])

# Funções de limpeza para taxas e booleanos
def clean_response_rate(entry):
    if (type(entry) == str):
        return entry.replace('%', '')
    else:
        return 0

data['host_response_rate'] = data['host_response_rate'].apply(clean_response_rate)

def clean_superhost(entry):
    return 1 if entry == 't' else 0

for col in ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'has_availability', 'instant_bookable', 'is_business_travel_ready', 'require_guest_profile_picture', 'require_guest_phone_verification']:
    data[col] = data[col].apply(clean_superhost)

# Limpeza de preços e contagens de quartos
def clean_price(entry):
    if (type(entry) != str and math.isnan(entry)):
        return -55
    entry1 = entry.replace('$', '').replace(',', '')
    if (float(entry1) == 0):
        return -55
    return np.log(float(entry1))

def clean_number_removal(entry):
    return -55 if math.isnan(entry) else entry

for col in ['bathrooms', 'bedrooms', 'beds']:
    data[col] = data[col].apply(clean_number_removal)
    data = data[data[col] != -55]

data['price'] = data['price'].apply(clean_price)
data = data[data['price'] != -55]

# Outras colunas financeiras
for col in ['extra_people', 'security_deposit', 'cleaning_fee']:
    data[col] = data[col].apply(clean_price)

# Limpeza de contagem de listings e estado
def clean_listings_count(entry):
    return 1 if math.isnan(entry) else entry

data['host_total_listings_count'] = data['host_total_listings_count'].apply(clean_listings_count)

def cleaned_state(entry):
    if isinstance(entry, str):
        return 'NY' if entry.upper() == 'NY' or entry.upper == 'New York' else entry
    return '' if math.isnan(entry) else entry

data['state'] = data['state'].apply(cleaned_state)
data = data[data['state'] == 'NY']

# Manipulação de amenidades
amenities_set = set()

def collect_amenities(entry):
    entry_list = entry.replace("{", "").replace("}", "").replace("'", "").replace('"', "").replace(" ", "_").split(',')
    for am in entry_list:
        if 'translation_missing' not in am and am != '':
            amenities_set.add(am)

data['amenities'].apply(collect_amenities)

def generic_amenities(entry, amenity):
    entry_list = entry.replace("{", "").replace("}", "").replace("'", "").replace('"', "").replace(" ", "_").split(',')
    for am in entry_list:
        if am == amenity:
            return 1
    return 0

for amenity in amenities_set:
    data.insert(len(list(data)), amenity, 0)
    data[amenity] = data['amenities'].apply(lambda x: generic_amenities(x, amenity))

data = pd.DataFrame.drop(data, columns=['amenities', 'state'])

# Criação de dummies
for col_name in ['property_type', 'bed_type', 'room_type', 'neighbourhood_group_cleansed', 'city', 'cancellation_policy', 'host_response_time', 'neighbourhood_cleansed']:
    parsed_cols = pd.get_dummies(data[col_name])
    data = data.drop(columns=[col_name])
    data = pd.concat([data, parsed_cols], axis=1)

# Manipulação da coluna 'host_since'
def clean_host_since(entry):
    return entry if isinstance(entry, str) else -55

data['host_since'] = data['host_since'].apply(clean_host_since)
data = data[data['host_since'] != -55]

dummy_date = dt.datetime(2018, 11, 10)
data['host_since'] = (dummy_date - pd.to_datetime(data['host_since'])).apply(lambda x: float(x.days))

# Salvando o resultado final
data.to_csv('../Data/data_cleaned.csv')
