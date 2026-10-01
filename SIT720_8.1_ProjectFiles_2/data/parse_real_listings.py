"""
Parses the manually-collected sold-listing text files (data/raw_collection/*_all.txt)
into the real dataset data/sydney_housing_sold.csv.

Each raw line has the form:
  <address>, <Suburb> | <beds/baths/parking/land info> <Type> | $<price> | <sale method> <date>

These lines were manually transcribed (by reading each listing card) from domain.com.au
sold-listing search results for Mosman, Parramatta and Mount Druitt (NSW), collected
September-October 2026. Only listings with a publicly disclosed sale price are included
("Price Withheld" listings were skipped during collection).
"""
import re
import csv
import datetime

MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# Suburb-level (not per-property) contextual enrichment, sourced from public
# suburb-profile information (ABS Census SA2 QuickStats, general distance-to-CBD
# knowledge, and a qualitative school-catchment rank). These are deliberately the
# SAME value for every property in a suburb -- unlike the real per-property fields
# (price, beds, baths, land size), which come directly off each listing.
SUBURB_CONTEXT = {
    "Mosman": {
        "distance_to_cbd_km": 8.0,
        "median_suburb_income_k": 185.0,   # approx., ABS Census SA2 median household income
        "school_zone_rank": 5,
    },
    "Parramatta": {
        "distance_to_cbd_km": 24.0,
        "median_suburb_income_k": 88.0,
        "school_zone_rank": 3,
    },
    "Mount Druitt": {
        "distance_to_cbd_km": 44.0,
        "median_suburb_income_k": 57.0,
        "school_zone_rank": 2,
    },
}

TYPE_MAP = {
    "apartment": "Unit",
    "unit": "Unit",
    "flat": "Unit",
    "new apartments/off the plan": "Unit",
    "studio": "Unit",
    "retirement living": "Unit",
    "house": "House",
    "townhouse": "Townhouse",
    "villa": "Townhouse",
    "semi-detached": "Townhouse",
}


def parse_date(s):
    m = re.search(r"(\d{1,2}) (\w{3}) (\d{4})", s)
    day, mon, year = int(m.group(1)), MONTHS[m.group(2)], int(m.group(3))
    return datetime.date(year, mon, day).isoformat()


def parse_line(line, suburb_key, idx):
    line = line.strip()
    if not line:
        return None
    addr_part, feat_part, price_part, date_part = [p.strip() for p in line.split("|")]
    suburb = addr_part.split(",")[-1].strip()

    price = int(re.sub(r"[^\d]", "", price_part))

    # bedrooms
    if feat_part.lower().startswith("studio"):
        bedrooms = 0
    else:
        bm = re.search(r"(\d+)\s*Bed", feat_part)
        bedrooms = int(bm.group(1)) if bm else None

    bam = re.search(r"(\d+)\s*Bath", feat_part)
    bathrooms = int(bam.group(1)) if bam else None

    pm = re.search(r"(\d+)\s*Parking", feat_part)
    car_spaces = int(pm.group(1)) if pm else None  # None (missing) if "-" shown

    lm = re.search(r"([\d,]+)\s*m²", feat_part)
    land_size = int(lm.group(1).replace(",", "")) if lm else None

    type_raw = feat_part.split()[-1].lower()
    # handle multi-word type tails
    low = feat_part.lower()
    if "semi-detached" in low:
        ptype = "Townhouse"
    elif "new apartments" in low or "off the plan" in low:
        ptype = "Unit"
    elif "retirement living" in low:
        ptype = "Unit"
    elif "townhouse" in low:
        ptype = "Townhouse"
    elif "villa" in low:
        ptype = "Townhouse"
    elif "studio" in low:
        ptype = "Unit"
    elif "house" in low:
        ptype = "House"
    else:
        ptype = "Unit"

    # strata/whole-complex land-size artefact: drop implausible land sizes recorded
    # against a Unit/strata property (these reflect the whole building's lot, not
    # the individual unit) -- documented as a known data-quality issue in Part 1.
    if ptype == "Unit":
        land_size = None

    sale_date = parse_date(date_part)
    prefix = {"Mosman": "MOS", "Parramatta": "PAR", "Mount Druitt": "MOU"}[suburb_key]
    property_id = f"{prefix}-{idx:04d}"

    row = {
        "property_id": property_id,
        "suburb": suburb_key,
        "property_type": ptype,
        "sale_date": sale_date,
        "sale_price": price,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "car_spaces": car_spaces,
        "land_size_sqm": land_size,
        "address": addr_part,
    }
    row.update(SUBURB_CONTEXT[suburb_key])
    return row


def load_suburb(path, suburb_key):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, start=1):
            if line.strip():
                rows.append(parse_line(line, suburb_key, i))
    return rows


if __name__ == "__main__":
    all_rows = []
    all_rows += load_suburb("raw_collection/mosman_all.txt", "Mosman")
    all_rows += load_suburb("raw_collection/parramatta_all.txt", "Parramatta")
    all_rows += load_suburb("raw_collection/mount_druitt_all.txt", "Mount Druitt")

    fieldnames = ["property_id", "suburb", "property_type", "sale_date", "sale_price",
                  "bedrooms", "bathrooms", "car_spaces", "land_size_sqm",
                  "distance_to_cbd_km", "median_suburb_income_k", "school_zone_rank",
                  "address"]

    with open("sydney_housing_sold.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"Wrote sydney_housing_sold.csv with {len(all_rows)} rows")
    from collections import Counter
    print(Counter(r["suburb"] for r in all_rows))
    print(Counter(r["property_type"] for r in all_rows))
    missing_beds = sum(1 for r in all_rows if r["bedrooms"] is None)
    missing_car = sum(1 for r in all_rows if r["car_spaces"] is None)
    missing_land = sum(1 for r in all_rows if r["land_size_sqm"] is None)
    print(f"Missing: bedrooms={missing_beds}, car_spaces={missing_car}, land_size={missing_land}")
