import re
import json
import random
import time
import html
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36")
}
EUR_TO_HUF = 400


def generate_keywords(asset_type: str, make: str, model: str, year: str = "", mileage: str = ""):
    asset_type = asset_type.lower()
    base = f"{make} {model} {year}".strip()
    kw = []
    if asset_type == "car":
        kw = [
            f"{base} site:mobile.de",
            f"{base} site:autoscout24.com Europe",
            f"{base} site:hasznaltauto.hu",
            f"{base} site:ebay.com Europe",
            f"{base} site:autotrader.com Europe",
        ]
    elif asset_type == "truck":
        kw = [
            f"{base} site:truckscout24.com Europe",
            f"{base} site:kleyntrucks.com Europe",
            f"{base} site:trucks.autoscout24.com Europe",
            f"{base} site:ebay.com Europe",
            f"{base} site:autotrader.com Europe",
        ]
    elif asset_type == "train":
        kw = [f"{base} locomotive for sale"]
    else:
        kw = [f"{base} used for sale Europe"]
    return kw


def fetch_html(q: str) -> str:
    url = "https://www.google.com/search?q=" + requests.utils.quote(q)
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.text


PRICE_RE = re.compile(r"(?:EUR|€|HUF|Ft)\s*([\d.,]+)|(\d[\d\s.,]*)\s*(?:EUR|€|HUF|Ft)")


def extract_prices(html_text: str):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_text, "html.parser")
    texts = [html.unescape(t.strip()) for t in soup.stripped_strings]
    prices, samples = [], []

    for idx, txt in enumerate(texts):
        matches = PRICE_RE.finditer(txt)
        for m in matches:
            try:
                price_str = m.group(1) or m.group(2)
                price_str = price_str.replace(" ", "").replace(".", "").replace(",", "")
                num = int(price_str)

                if num < 1000:
                    continue

                cur = "EUR" if any(c in txt for c in ("EUR", "€")) else "HUF"
                if cur == "HUF":
                    num_eur = round(num / EUR_TO_HUF)
                else:
                    num_eur = num

                if num_eur < 1000 or num_eur > 500000:
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

    def run_search(keywords):
        prices_accum = []
        samples_accum = []
        for kw in keywords:
            try:
                print(f"Searching with keyword: {kw}")
                html_text = fetch_html(kw)
                prices, samples = extract_prices(html_text)
                prices_accum.extend(prices)
                samples_accum.extend(samples)
                time.sleep(random.uniform(3, 8))
            except Exception as exc:
                print(f"Search failed for {kw}: {exc}")
                continue
        return prices_accum, samples_accum

    full_keywords = generate_keywords(item_type, make, model, year, mileage)
    prices, samples = run_search(full_keywords)

    if not prices:
        simple_model = model.split()[0] if model else ""
        fallback_keywords = generate_keywords(item_type, make, simple_model, year, mileage)
        prices, samples = run_search(fallback_keywords)

        if not prices:
            return {"result": None, "samples": []}

    all_prices = prices
    all_samples = samples

    all_prices.sort()
    avg_price = sum(all_prices) // len(all_prices)

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
        "asset_type": r"-\s+\*\*Asset Type:\*\*\s*(\w+)",
        "make": r"-\s+\*\*Manufacturer:\*\*\s*(.+)",
        "model": r"-\s+\*\*Model:\*\*\s*(.+)",
        "year": r"-\s+\*\*Year of Manufacture:\*\*\s*(\d{4})",
        "mileage": r"-\s+\*\*Odometer Reading:\*\*\s*([\d,\.]+)",
    }

    extracted = {}
    for key, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            extracted[key] = match.group(1).strip()
        else:
            extracted[key] = ""
    return extracted


def format_sample_block(samples):
    if not samples:
        return ""

    lines = []
    for s in samples:
        context = re.sub(r'[^\w\s,.-]', '', s["context"])
        context = re.sub(r'\s+', ' ', context).strip()

        lines.append(
            f"- {context[:100]}... ; "
            f"**{s['price_eur']:,} EUR** "
            f"(~{int(s['price_eur'] * EUR_TO_HUF):,} HUF)"
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
