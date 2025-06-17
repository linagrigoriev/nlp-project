import re
import json
import random
import time
import html
import requests
from bs4 import BeautifulSoup
import statistics

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
EUR_TO_HUF = 400

def generate_keywords(asset_type: str, make: str, model: str, year: str = "", mileage: str = ""):
    asset_type = asset_type.lower()
    base = f"{make} {model} {year}".strip()
    kw = []
    kw += [
        f"{base} site:marktplaats.nl",
        f"{base} site:olx.pl",
        f"{base} site:leboncoin.fr",
        f"{base} site:gumtree.co.uk",
        f"{base} site:ebay.com",
    ]
    if asset_type == "car":
        kw = [
            f"{base} site:mobile.de",
            f"{base} site:autoscout24.com",
            f"{base} site:hasznaltauto.hu",
            f"{base} site:autotrader.co.uk",
            f"{base} site:car.gr",
            f"{base} site:autovit.ro",
            f"{base} site:otomoto.pl",
            f"{base} site:leboncoin.fr",
            f"{base} site:ebay-kleinanzeigen.de",
        ]
    elif asset_type == "truck":
        kw = [
            f"{base} site:trucks.autoscout24.com",
            f"{base} site:kleyntrucks.com",
            f"{base} site:truckscout24.com",
            f"{base} site:mascus.com",
            f"{base} site:usedtrucks.eu",
            f"{base} site:basworld.com",
            f"{base} site:autoline.info",
        ]
    elif asset_type == "train":
        kw = [f"{base} site:railwaygazette.com",
            f"{base} site:railswap.org",
            f"{base} site:mascus.com",
            f"{base} site:usedlocomotives.com",
        ]
    else:
        kw = [f"{base} site:mascus.com",
        f"{base} site:machineryzone.eu",
        f"{base} site:equipmenttrader.com",
        f"{base} site:marketbook.eu",
        f"{base} site:agriaffaires.co.uk",
        f"{base} site:trademachines.com",
        ]
    return kw

def fetch_html(q: str) -> str:
    # Append "&hl=en" to force English language results.
    url = "https://www.google.com/search?q=" + requests.utils.quote(q) + "&hl=en"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.text

# Use re.IGNORECASE so our currency symbols and abbreviations match regardless of case.
PRICE_RE = re.compile(r"(?:EUR|€|HUF|Ft)\s*([\d.,]+)|(\d[\d\s.,]*)\s*(?:EUR|€|HUF|Ft)", re.IGNORECASE)

def extract_prices(html_text: str):
    soup = BeautifulSoup(html_text, "html.parser")
    texts = [html.unescape(t.strip()) for t in soup.stripped_strings]
    prices, samples = [], []

    for idx, txt in enumerate(texts):
        matches = PRICE_RE.finditer(txt)
        for m in matches:
            try:
                price_str = m.group(1) or m.group(2)
                # Remove extra spaces and common punctuation
                price_str = price_str.replace(" ", "").replace(".", "").replace(",", "")
                num = int(price_str)

                if num < 1000:
                    continue

                cur = "EUR" if any(c in txt for c in ("EUR", "€")) else "HUF"
                if cur == "HUF":
                    num_eur = round(num / EUR_TO_HUF)
                else:
                    num_eur = num

                if num_eur < 5000 or num_eur > 200000:
                    continue

                prices.append(num_eur)

                start_idx = max(0, idx - 2)
                end_idx = min(len(texts), idx + 3)
                context = " ".join(texts[start_idx:end_idx])
                context = context[:200]

                samples.append({
                    "price_eur": num_eur,
                    "currency": cur,
                    "context": context
                })

            except (ValueError, AttributeError):
                continue

    return prices, samples

def estimate_value(item_type: str, make: str, model: str, year: str = "", mileage: str = ""):
    all_prices = []
    all_samples = []

    for kw in generate_keywords(item_type, make, model, year, mileage):
        try:
            print(f"Searching with keyword: {kw}")
            html_text = fetch_html(kw)
            prices, samples = extract_prices(html_text)
            all_prices.extend(prices)
            all_samples.extend(samples)
            # Delay between requests to simulate human behavior
            time.sleep(random.uniform(3, 8))
        except Exception as exc:
            print(f"Search failed for {kw}: {exc}")
            continue

    if not all_prices:
        return {"result": None, "samples": []}

    all_prices.sort()
    avg_price = int(statistics.median(all_prices))

    result = {
        "estimated_average": avg_price,
        "price_range_eur": [all_prices[0], all_prices[-1]],
        "approx_huf": int(avg_price * EUR_TO_HUF),
        "listings": len(all_prices)
    }

    return {
        "result": result,
        "samples": sorted(all_samples, key=lambda x: x["price_eur"])[:5]
    }

def extract_fields(text):
    FIELD_PATTERNS = {
        "asset_type": r"-\s+\*\*Asset Type:\*\*\s*(.+)",
        "make":       r"-\s+\*\*Manufacturer:\*\*\s*([A-Za-z\- ]+)",
        "model":      r"-\s+\*\*Model:\*\*\s*([A-Za-z0-9\- ]+)",
        "year":       r"-\s+\*\*Year of Manufacture:\*\*\s*(\d{4})",
        "mileage":    r"-\s+\*\*Odometer Reading:\*\*\s*([\d,\.]+)\s*km",
    }

    results = {}
    for field, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            print(f"{field}: {'MATCH' if match else 'NO MATCH'}")
            value = match.group(1).strip()
            if field == "mileage":
                value = value.replace(",", "").replace(".", "")  # normalize to digits only
            results[field] = value
        else:
            results[field] = None

    return results

def format_sample_block(samples):
    if not samples:
        return ""

    lines = []
    for s in samples:
        context = re.sub(r'[^\w\s,.-]', '', s["context"])
        context = re.sub(r'\s+', ' ', context).strip()
        lines.append(
            f"- {context[:100]}... ; **{s['price_eur']:,} EUR** (~{int(s['price_eur'] * EUR_TO_HUF):,} HUF)"
        )
    return "\n".join(lines)

def update_report_with_valuation_text(report_text, average_price_huf):
    valuation_heading = "### Valuation Principles"
    valuation_text = (
        f"{valuation_heading}\n"
        f"- The valuation is based on the vehicle's make, model, year, overall condition, market trends, and comparable sales.\n"
        f"- The estimated average price of {average_price_huf:,} HUF is considered in line with the market value for this type of vehicle in similar condition.\n"
    )

    if valuation_heading in report_text:
        pattern = re.compile(
            rf"{re.escape(valuation_heading)}.*?(?=\n### |\Z)",
            re.DOTALL | re.MULTILINE,
        )
        updated_text = pattern.sub(valuation_text, report_text)
    else:
        updated_text = report_text.strip() + "\n\n" + valuation_text

    return updated_text

def remove_outliers(prices, lower=0.1, upper=0.9):
    prices.sort()
    n = len(prices)
    start = int(n * lower)
    end = int(n * upper)
    return prices[start:end] if end > start else prices

if __name__ == "__main__":
    input_filepath = "output/generated_reports_raw/157515_v2.md"
    output_filepath = "generated_reports/157515_v2.md"

    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            report_md = f.read()
        print("File loaded successfully.\n")
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)

    print("\nExtracting fields...")
    fields = extract_fields(report_md)

    if not fields.get("make") or not fields.get("model"):
        print("Error: Could not extract necessary fields (make, model). Please check your input format.")
        exit(1)

    print("\nEstimating value by scraping online listings...")
    estimate = estimate_value(fields.get("asset_type"), fields.get("make"), fields.get("model"), fields.get("year"), fields.get("mileage"))

    if not estimate["result"]:
        print("No price data found from online search.")
        exit(1)

    print("\nEstimate Result:")
    print(json.dumps(estimate["result"], indent=2))

    print("\nUpdating report with valuation...")
    updated_report = update_report_with_valuation_text(report_md, estimate["result"]["approx_huf"])

    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(updated_report)
        print(f"Report updated successfully and saved to:\n{output_filepath}")
    except Exception as e:
        print(f"Error writing updated report: {e}")
