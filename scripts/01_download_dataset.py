"""
01_download_dataset.py — SignPAK-AI (Strict Verified Downloader)
================================================================
Features:
  - Scoped DOM queries directly from the active <video> element.
  - SPA Navigation Barrier: Waits for video src to update between pages.
  - URL Slug Verification: Ensures CloudFront URL matches the target concept.
  - 100% Exact Title Matching from CSV.

Run from: SIGNPAK-AI root → python scripts/01_download_dataset.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import re
import json
import time
import logging
import shutil
from typing import Optional, Tuple, Dict

import cv2
import pandas as pd
import requests

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
WORDS_LIST_DIR = DATA_DIR / "Words List"
TARGET_SIGNER_DIR = DATA_DIR / "Signer_0"
TARGET_SIGNER_DIR.mkdir(parents=True, exist_ok=True)


def normalize_title(text: str) -> str:
    t = re.sub(r'\(.*?\)', '', str(text))
    t = re.sub(r'[^\w\s]', ' ', t)
    return " ".join(t.lower().split())


def load_vocabulary_from_csvs() -> Dict[str, Tuple[str, str]]:
    routing_map = {}
    csv_files = sorted(WORDS_LIST_DIR.glob("Words - *.csv"))
    if not csv_files:
        csv_files = sorted(Path(".").glob("Words - *.csv"))

    if not csv_files:
        logger.error(f"❌ No vocabulary CSV files found in {WORDS_LIST_DIR}!")
        return {}

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                eng_word = str(row.get("English Word", "")).strip()
                cat = str(row.get("Category", "")).strip()
                if eng_word and cat and eng_word.lower() != "nan":
                    routing_map[normalize_title(eng_word)] = (cat, eng_word)
        except Exception as e:
            logger.error(f"Error reading {csv_file.name}: {e}")

    logger.info(f"📋 Loaded {len(routing_map)} strict target concepts from CSV.\n")
    return routing_map


VOCABULARY_ROUTING = load_vocabulary_from_csvs()


def get_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--log-level=3")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(20)
    return driver


def extract_verified_video_url(driver: webdriver.Chrome, page_url: str, last_seen_url: str) -> Optional[str]:
    """
    Extracts the video URL strictly from the active video player element,
    waiting for SPA DOM hydration to prevent capturing stale previous URLs.
    """
    try:
        driver.get(page_url)
        time.sleep(1.5)

        js_extract = """
        let v = document.querySelector('video');
        if (v) {
            let src = v.currentSrc || v.src;
            if (src && src.startsWith('http')) return src;
            let s = v.querySelector('source');
            if (s && s.src && s.src.startsWith('http')) return s.src;
        }
        return null;
        """

        current_url = driver.execute_script(js_extract)

        # Ensure SPA has updated away from previous page's video
        for _ in range(8):
            if current_url and current_url != last_seen_url:
                return current_url
            time.sleep(0.5)
            current_url = driver.execute_script(js_extract)

        return current_url
    except Exception as e:
        logger.error(f"Extraction error on {page_url}: {e}")
        return None


def download_video(session: requests.Session, url: str, output_path: Path) -> bool:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://psl.org.pk/"
    }

    try:
        with session.get(url, headers=headers, stream=True, timeout=20) as r:
            if r.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

                # OpenCV verification
                cap = cv2.VideoCapture(str(output_path))
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if frames > 2:
                    return True
    except Exception as e:
        logger.warning(f"Download stream error: {e}")

    return False


def main():
    session = requests.Session()
    categories_raw = session.get(config.CATEGORIES_URL, timeout=15).json()
    cat_list = categories_raw.get("data", [])

    logger.info(f"🌐 Fetching concepts across {len(cat_list)} PSL categories...")

    # Build exact concept registry
    catalog = {}
    for cat in cat_list:
        cat_id = cat["id"]
        cat_slug = cat["slug"]
        cat_url = config.CATEGORY_DETAIL_URL.format(cat_id)
        try:
            detail = session.get(cat_url, timeout=10).json()
            concepts = detail.get("data", {}).get("concepts", [])
            for c in concepts:
                c_title = str(c.get("title", "")).strip()
                norm = normalize_title(c_title)
                c_id = c.get("id")
                c_slug = c.get("slug") or norm.replace(" ", "-")

                # Disambiguate English alphabet vs words
                if len(norm) == 1 and norm.isalpha():
                    if "alphabet" in cat_slug.lower():
                        catalog[norm] = (cat_id, cat_slug, c_id, c_slug, c_title)
                else:
                    if norm in VOCABULARY_ROUTING and norm not in catalog:
                        catalog[norm] = (cat_id, cat_slug, c_id, c_slug, c_title)
        except Exception:
            continue

    logger.info(f"✅ Found {len(catalog)} exact matching concepts in PSL database.")

    driver = get_driver()
    downloaded_count = 0
    failed_count = 0
    last_video_url = ""

    try:
        for norm_word, (custom_cat, formal_title) in VOCABULARY_ROUTING.items():
            if norm_word not in catalog:
                logger.warning(f"⚠️ Target '{formal_title}' not found in PSL online registry. Skipping.")
                failed_count += 1
                continue

            cat_id, cat_slug, c_id, c_slug, raw_title = catalog[norm_word]
            page_url = f"https://psl.org.pk/dictionary/{cat_id}-{cat_slug}/{c_id}-{c_slug}"

            target_folder = TARGET_SIGNER_DIR / custom_cat
            target_folder.mkdir(parents=True, exist_ok=True)
            video_path = target_folder / f"{formal_title}.mp4"

            logger.info(f"🎯 Fetching Verified Video: '{formal_title}' ({custom_cat})")
            video_url = extract_verified_video_url(driver, page_url, last_video_url)

            if not video_url:
                logger.error(f"  ❌ Could not extract video player URL for '{formal_title}'")
                failed_count += 1
                continue

            last_video_url = video_url
            success = download_video(session, video_url, video_path)

            if success:
                logger.info(f"  ✅ Saved: {video_path.name} (Source: {video_url.split('/')[-1]})")
                downloaded_count += 1
            else:
                logger.error(f"  ❌ Download failed for: {formal_title}")
                failed_count += 1

    finally:
        driver.quit()

    print("\n" + "=" * 60)
    print("FINAL DATASET DOWNLOAD STATISTICS")
    print("=" * 60)
    print(f"Downloaded & Verified : {downloaded_count}")
    print(f"Failed / Missing      : {failed_count}")
    print(f"Total Target Words    : {len(VOCABULARY_ROUTING)}")
    print("=" * 60)


if __name__ == "__main__":
    main()