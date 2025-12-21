# src/geo/geonormalizer.py

def normalize_geo_name(name):
    if not name:
        return None

    n = name.strip().lower()

    # ---------------------------------------------------------
    # 1) COMMON COUNTRY ABBREVIATIONS
    # ---------------------------------------------------------
    country_map = {
        "u.s.": "United States",
        "u.s": "United States",
        "us": "United States",
        "usa": "United States",
        "the u.s.": "United States",
        "america": "United States",
        "united states": "United States",

        "uk": "United Kingdom",
        "britain": "United Kingdom",
        "great britain": "United Kingdom",
        "england": "United Kingdom",

        "ussr": "Russia",
        "soviet union": "Russia",
        "russia": "Russia",

        "west germany": "Germany",
        "east germany": "Germany",

        "china": "China",
        "prc": "China",

        "uae": "United Arab Emirates",
        "u.a.e.": "United Arab Emirates",

        "south korea": "South Korea",
        "north korea": "North Korea",

        "saudi": "Saudi Arabia",
        "saudi arabia": "Saudi Arabia",

        "iran": "Iran",
        "iraq": "Iraq",

        "canada": "Canada",
        "mexico": "Mexico",
        "france": "France",
        "italy": "Italy",
        "spain": "Spain",
        "brazil": "Brazil",
        "argentina": "Argentina",
        "egypt": "Egypt",
        "japan": "Japan",
        "india": "India",
        "turkey": "Turkey",
        "greece": "Greece",
    }

    if n in country_map:
        return country_map[n]

    # ---------------------------------------------------------
    # 2) BIG U.S. CITIES
    # ---------------------------------------------------------
    us_city_map = {
        "new york": "New York, USA",
        "nyc": "New York, USA",
        "manhattan": "Manhattan, New York, USA",

        "washington": "Washington, DC, USA",
        "washington d.c.": "Washington, DC, USA",
        "dc": "Washington, DC, USA",

        "los angeles": "Los Angeles, USA",
        "san francisco": "San Francisco, USA",

        "boston": "Boston, USA",
        "chicago": "Chicago, USA",
        "atlanta": "Atlanta, USA",
        "houston": "Houston, USA",
        "miami": "Miami, USA",
        "seattle": "Seattle, USA",
        "detroit": "Detroit, USA",
        "philadelphia": "Philadelphia, USA",
        "pittsburgh": "Pittsburgh, USA",

        "white house": "White House, Washington DC, USA",
        "the white house": "White House, Washington DC, USA",
        "the pentagon": "Pentagon, Washington DC, USA",
    }

    if n in us_city_map:
        return us_city_map[n]

    # ---------------------------------------------------------
    # 3) INTERNATIONAL CITIES
    # ---------------------------------------------------------
    international_cities = {
        "toronto": "Toronto, Canada",
        "montreal": "Montreal, Canada",
        "vancouver": "Vancouver, Canada",

        "london": "London, UK",
        "paris": "Paris, France",
        "rome": "Rome, Italy",
        "madrid": "Madrid, Spain",
        "lisbon": "Lisbon, Portugal",

        "berlin": "Berlin, Germany",
        "frankfurt": "Frankfurt, Germany",
        "hamburg": "Hamburg, Germany",
        "bonn": "Bonn, Germany",

        "moscow": "Moscow, Russia",
        "kyiv": "Kyiv, Ukraine",

        "tokyo": "Tokyo, Japan",
        "osaka": "Osaka, Japan",

        "beijing": "Beijing, China",
        "shanghai": "Shanghai, China",
        "hong kong": "Hong Kong",

        "cairo": "Cairo, Egypt",
        "johannesburg": "Johannesburg, South Africa",
        "sydney": "Sydney, Australia",
        "melbourne": "Melbourne, Australia",
    }

    if n in international_cities:
        return international_cities[n]

    # ---------------------------------------------------------
    # 4) FINANCIAL LOCATIONS
    # ---------------------------------------------------------
    financial_map = {
        "toronto stock exchange": "Toronto, Canada",
        "new york stock exchange": "New York, USA",
        "chicago board of trade": "Chicago, USA",
        "wall street journal": "New York, USA",
        "dow jones": "New York, USA",
    }

    if n in financial_map:
        return financial_map[n]

    # ---------------------------------------------------------
    # 5) DATELINE CAPITALS
    # ---------------------------------------------------------
    dateline_cities = {
        "brussels": "Brussels, Belgium",
        "vienna": "Vienna, Austria",
        "athens": "Athens, Greece",
        "ankara": "Ankara, Turkey",
        "istanbul": "Istanbul, Turkey",
        "jakarta": "Jakarta, Indonesia",
        "seoul": "Seoul, South Korea",
        "singapore": "Singapore",
        "nairobi": "Nairobi, Kenya",
        "lagos": "Lagos, Nigeria",
        "riyadh": "Riyadh, Saudi Arabia",
        "dubai": "Dubai, UAE",
        "abu dhabi": "Abu Dhabi, UAE",
    }

    if n in dateline_cities:
        return dateline_cities[n]

    # ---------------------------------------------------------
    # 6) FALLBACK
    # ---------------------------------------------------------
    return name.title()
