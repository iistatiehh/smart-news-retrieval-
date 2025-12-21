import spacy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ================================
#        LOAD SPACY MODEL
# ================================
nlp = spacy.load("en_core_web_sm")


# ================================
#   GEOREFERENCE (NER) EXTRACTOR
# ================================
def extract_georeferences(text):
    """
    Extract geographical references from text using spaCy NER.
    Extracts:
      - GPE : countries, cities, states
      - LOC : natural locations
      - FAC : facilities / buildings
    """
    if not text:
        return []

    doc = nlp(text)
    places = []

    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "FAC"]:
            places.append(ent.text)

    # Remove duplicates
    return list(set(places))


# ================================
#         GEOCODER FUNCTION
# ================================
geolocator = Nominatim(user_agent="smart_doc_system")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


def geocode_place(place_name):
    """
    Convert a place name into GPS coordinates using Nominatim.
    Returns:
        { "lat": ..., "lon": ... }
    Or:
        None (if not found)
    """
    try:
        location = geocode(place_name)
        if location:
            return {
                "lat": location.latitude,
                "lon": location.longitude
            }
        return None
    except:
        return None
