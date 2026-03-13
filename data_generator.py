"""
data_generator.py
Synthetic Airbnb dataset generator for Dubai listings.
Run once, cached via st.cache_data.
"""

import numpy as np
import pandas as pd

def generate_airbnb_data(n=3500, seed=42):
    rng = np.random.default_rng(seed)

    cities      = ["Dubai"]
    neighborhoods = [
        "Downtown Dubai", "Dubai Marina", "Jumeirah", "Business Bay",
        "Palm Jumeirah", "DIFC", "JBR", "Al Barsha", "Deira", "Bur Dubai",
        "Dubai Hills", "Mirdif", "Silicon Oasis", "Sports City", "Discovery Gardens"
    ]
    property_types = ["Apartment", "Villa", "Studio", "Hotel Room", "Townhouse", "Penthouse"]
    room_types     = ["Entire home/apt", "Private room", "Shared room"]
    seasons        = ["Winter", "Spring", "Summer", "Autumn"]
    months         = list(range(1, 13))
    days_of_week   = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    traveler_types = ["Solo", "Couple", "Family", "Business", "Group"]

    # Neighborhood characteristics
    premium_hoods = {"Downtown Dubai", "Palm Jumeirah", "DIFC", "Dubai Marina", "JBR"}
    budget_hoods  = {"Deira", "Bur Dubai", "Silicon Oasis", "Discovery Gardens", "Mirdif"}

    neighborhood   = rng.choice(neighborhoods, n)
    is_premium     = np.array([nb in premium_hoods for nb in neighborhood])
    is_budget      = np.array([nb in budget_hoods  for nb in neighborhood])

    property_type  = rng.choice(property_types, n, p=[0.45, 0.15, 0.20, 0.05, 0.10, 0.05])
    room_type      = rng.choice(room_types,     n, p=[0.65, 0.30, 0.05])
    bedrooms       = rng.integers(0, 6, n)
    bathrooms      = np.clip(bedrooms + rng.integers(-1, 2, n), 1, 6)
    accommodates   = np.clip(bedrooms * 2 + rng.integers(0, 3, n), 1, 16)

    # Distance
    dist_center = np.where(
        is_premium,
        rng.uniform(0.5,  5.0, n),
        np.where(is_budget,
                 rng.uniform(8.0, 25.0, n),
                 rng.uniform(3.0, 15.0, n))
    )
    dist_metro  = dist_center * rng.uniform(0.3, 1.2, n)

    # Price
    base_price  = (
        200 + is_premium * 300 - is_budget * 80
        + bedrooms * 50
        + (property_type == "Villa").astype(int) * 200
        + (property_type == "Penthouse").astype(int) * 350
        - (room_type == "Private room").astype(int) * 80
        - (room_type == "Shared room").astype(int) * 120
        - dist_center * 8
    )
    price_per_night = np.clip(base_price + rng.normal(0, 60, n), 50, 2000).round(0)
    cleaning_fee    = (price_per_night * rng.uniform(0.05, 0.20, n)).round(0)
    min_nights      = rng.choice([1, 2, 3, 5, 7], n, p=[0.4, 0.25, 0.15, 0.10, 0.10])
    max_nights      = rng.choice([7, 14, 30, 90, 365], n, p=[0.1, 0.15, 0.25, 0.25, 0.25])

    # Amenities
    amenities_count   = rng.integers(3, 30, n)
    wifi              = rng.random(n) < 0.95
    kitchen           = rng.random(n) < (0.80 - is_premium * 0.05)
    air_conditioning  = rng.random(n) < 0.98
    parking           = rng.random(n) < (0.40 + is_premium * 0.30)
    self_checkin      = rng.random(n) < 0.60
    pool              = rng.random(n) < (0.10 + is_premium * 0.35)

    # Reviews & host
    review_score      = np.clip(rng.normal(4.3, 0.5, n), 1, 5).round(1)
    num_reviews       = np.clip(rng.integers(0, 300, n), 0, 500)
    host_resp_rate    = np.clip(rng.normal(90, 12, n), 0, 100).round(0)
    superhost         = (
        (review_score > 4.5) & (num_reviews > 20) & (host_resp_rate > 85)
        & (rng.random(n) < 0.55)
    )

    # Time
    month         = rng.choice(months, n)
    day_of_week   = rng.choice(days_of_week, n)
    season_map    = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                     6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"}
    season        = np.array([season_map[m] for m in month])

    traveler_type    = rng.choice(traveler_types, n, p=[0.20, 0.30, 0.25, 0.15, 0.10])
    length_of_stay   = rng.integers(1, 30, n)
    booking_lead     = rng.integers(0, 180, n)

    # Occupancy
    occ_base = (
        0.50
        + is_premium * 0.12
        + (review_score / 5) * 0.20
        + superhost.astype(float) * 0.08
        + wifi.astype(float) * 0.03
        + pool.astype(float) * 0.05
        - (price_per_night / 2000) * 0.15
        - dist_center / 100
        + (day_of_week == "Saturday").astype(float) * 0.05
        + (day_of_week == "Friday").astype(float) * 0.04
        + (season == "Winter").astype(float) * 0.10     # Dubai peak
        + rng.normal(0, 0.05, n)
    )
    occupancy_rate = np.clip(occ_base, 0.05, 0.98).round(2)

    # Booking status
    booking_prob = (
        0.40
        + (review_score / 5) * 0.25
        + superhost.astype(float) * 0.12
        + (occupancy_rate - 0.5) * 0.30
        + wifi.astype(float) * 0.05
        - (price_per_night / 2000) * 0.10
        + rng.normal(0, 0.05, n)
    )
    booking_status = np.where(rng.random(n) < np.clip(booking_prob, 0.05, 0.95),
                              "Booked", "Not Booked")

    df = pd.DataFrame({
        "Listing_ID":           np.arange(1, n+1),
        "City":                 "Dubai",
        "Neighborhood":         neighborhood,
        "Property_Type":        property_type,
        "Room_Type":            room_type,
        "Bedrooms":             bedrooms,
        "Bathrooms":            bathrooms,
        "Accommodates":         accommodates,
        "Price_Per_Night":      price_per_night,
        "Cleaning_Fee":         cleaning_fee,
        "Minimum_Nights":       min_nights,
        "Maximum_Nights":       max_nights,
        "Amenities_Count":      amenities_count,
        "Wifi":                 wifi.astype(int),
        "Kitchen":              kitchen.astype(int),
        "Air_Conditioning":     air_conditioning.astype(int),
        "Parking":              parking.astype(int),
        "Self_Checkin":         self_checkin.astype(int),
        "Pool":                 pool.astype(int),
        "Review_Score":         review_score,
        "Number_of_Reviews":    num_reviews,
        "Host_Response_Rate":   host_resp_rate,
        "Superhost_Status":     superhost.astype(int),
        "Distance_to_City_Center": dist_center.round(2),
        "Distance_to_Metro":    dist_metro.round(2),
        "Season":               season,
        "Month":                month,
        "Day_of_Week":          day_of_week,
        "Traveler_Type":        traveler_type,
        "Length_of_Stay":       length_of_stay,
        "Booking_Lead_Time":    booking_lead,
        "Occupancy_Rate":       occupancy_rate,
        "Booking_Status":       booking_status,
    })
    return df
