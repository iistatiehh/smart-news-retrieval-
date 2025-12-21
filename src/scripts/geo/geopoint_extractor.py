import json
import os
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

CACHE_PATH = "src/geo/geocache.json"

# Load cache
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        GEOCACHE = json.load(f)
else:
    GEOCACHE = {}

# Save cache helper
def save_cache():
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(GEOCACHE, f, indent=2)

# Geocoder setup
geolocator = Nominatim(user_agent="smart_doc_system", timeout=5)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

from src.geo.geonormalizer import normalize_geo_name


def get_geopoint(place_name):
    if not place_name:
        return None

    norm = normalize_geo_name(place_name)

    # 1) If normalized value has static coordinates → return directly
    STATIC_PLACES = {
        "united states": {"lat": 37.0902, "lon": -95.7129},
        "usa": {"lat": 37.0902, "lon": -95.7129},
        "u.s.": {"lat": 37.0902, "lon": -95.7129},
        "united kingdom": {"lat": 55.3781, "lon": -3.4360},
        "uk": {"lat": 55.3781, "lon": -3.4360},
    }

    if norm in STATIC_PLACES:
        return STATIC_PLACES[norm]

    # 2) Cache hit
    if norm in GEOCACHE:
        return GEOCACHE[norm]

    # 3) API geocode
    try:
        location = geocode(norm)
        if location:
            GEOCACHE[norm] = {
                "lat": location.latitude,
                "lon": location.longitude
            }
            save_cache()
            return GEOCACHE[norm]

    except Exception as e:
        print(f"⚠️ Geocode failed for '{norm}': {e}")

    GEOCACHE[norm] = None
    save_cache()
    return None
