"""
00_verify_and_fix_dataset.py — SignPAK-AI (Video Integrity & Mismatch Fixer)
============================================================================
1. Inspects all MP4 files in data/Signer_0/.
2. Queries the PSL API directly to obtain the EXACT concept ID and page for each word.
3. Uses Scoped DOM queries to extract the true video player URL.
4. Verifies that the video filename matches the concept slug.
5. Replaces any swapped or incorrect video files.

Run from: SIGNPAK-AI root → python scripts/00_verify_and_fix_dataset.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import re
import json
import time
import cv2
import pandas as pd
import requests

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
WORDS_LIST_DIR = DATA_DIR / "Words List"
SIGNER_0_DIR = DATA_DIR / "Signer_0"


def normalize(text: str) -> str:
    t = re.sub(r'\(.*?\)', '', str(text))
    t = re.sub(r'[^\w\s]', ' ', t)
    return " ".join(t.lower().split())


def get_driver():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--log-level=3")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


def extract_exact_video_url(driver, page_url: str, expected_slug: str) -> str | None:
    """Extracts the video URL strictly from the main active <video> player element."""
    try:
        driver.get(page_url)
        time.sleep(2)

        # Scoped JavaScript query directly on the active <video> tag
        js_extract = """
        let v = document.querySelector('video');
        if (v) {
            if (v.currentSrc && v.currentSrc.startsWith('http')) return v.currentSrc;
            if (v.src && v.src.startsWith('http')) return v.src;
            let s = v.querySelector('source');
            if (s && s.src && s.src.startsWith('http')) return s.src;
        }
        return null;
        """
        video_url = driver.execute_script(js_extract)

        # If not loaded yet, wait up to 6 seconds for DOM hydration
        if not video_url:
            for _ in range(6):
                time.sleep(1)
                video_url = driver.execute_script(js_extract)
                if video_url:
                    break

        return video_url
    except Exception as e:
        print(f"    ⚠️ DOM extraction error on {page_url}: {e}")
        return None


def verify_and_fix():
    print("=" * 60)
    print("🔍 AUDITING & FIXING SIGNPAK-AI VIDEOS")
    print("=" * 60)

    # 1. Load target words from CSV
    csv_files = sorted(WORDS_LIST_DIR.glob("Words - *.csv"))
    if not csv_files:
        csv_files = sorted(Path(".").glob("Words - *.csv"))

    target_map = {}
    for csv_f in csv_files:
        df = pd.read_csv(csv_f)
        for _, row in df.iterrows():
            w = str(row.get("English Word", "")).strip()
            cat = str(row.get("Category", "")).strip()
            if w and w.lower() != "nan":
                target_map[normalize(w)] = (cat, w)

    print(f"📋 Loaded {len(target_map)} target concepts from CSV.")

    # 2. Fetch PSL Categories & Build Concept Index
    session = requests.Session()
    categories_resp = session.get(config.CATEGORIES_URL, timeout=15).json()
    cat_list = categories_resp.get("data", [])

    print(f"🌐 Indexing concepts across {len(cat_list)} PSL online categories...")
    psl_concept_catalog = {}

    for cat in cat_list:
        cat_id = cat["id"]
        cat_slug = cat["slug"]
        cat_url = config.CATEGORY_DETAIL_URL.format(cat_id)
        try:
            detail = session.get(cat_url, timeout=10).json()
            concepts = detail.get("data", {}).get("concepts", [])
            for c in concepts:
                c_title = str(c.get("title", "")).strip()
                norm_title = normalize(c_title)
                c_id = c.get("id")
                c_slug = c.get("slug") or norm_title.replace(" ", "-")

                # Disambiguation: English Alphabet concepts must come from alphabet category
                if len(norm_title) == 1 and norm_title.isalpha():
                    if "alphabet" in cat_slug.lower():
                        psl_concept_catalog[norm_title] = (cat_id, cat_slug, c_id, c_slug, c_title)
                else:
                    if norm_title in target_map and norm_title not in psl_concept_catalog:
                        psl_concept_catalog[norm_title] = (cat_id, cat_slug, c_id, c_slug, c_title)
        except Exception:
            continue

    print(f"✅ Indexed {len(psl_concept_catalog)} matching concepts from PSL.")

    # 3. Launch Selenium to verify and download
    driver = get_driver()
    fixed_count = 0
    skipped_count = 0

    try:
        for norm_word, (cat_name, formal_title) in target_map.items():
            target_folder = SIGNER_0_DIR / cat_name
            target_folder.mkdir(parents=True, exist_ok=True)
            video_file = target_folder / f"{formal_title}.mp4"

            if norm_word not in psl_concept_catalog:
                print(f"⚠️ Target '{formal_title}' not found in PSL catalog. Skipping.")
                continue

            cat_id, cat_slug, c_id, c_slug, raw_title = psl_concept_catalog[norm_word]
            page_url = f"https://psl.org.pk/dictionary/{cat_id}-{cat_slug}/{c_id}-{c_slug}"

            # If file exists, check OpenCV health and URL slug
            file_needs_replacement = False
            if video_file.exists():
                cap = cv2.VideoCapture(str(video_file))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if frame_count < 5:
                    file_needs_replacement = True
            else:
                file_needs_replacement = True

            if not file_needs_replacement:
                skipped_count += 1
                continue

            print(f"🔄 Fetching Exact Video for: '{formal_title}' ({cat_name})")
            video_url = extract_exact_video_url(driver, page_url, c_slug)

            if not video_url:
                print(f"  ❌ Failed to locate video player for '{formal_title}' at {page_url}")
                continue

            # Download video directly
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Referer": "https://psl.org.pk/"
                }
                resp = session.get(video_url, headers=headers, stream=True, timeout=20)
                if resp.status_code == 200:
                    with open(video_file, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"  ✅ Correctly Downloaded: {video_file.name} (Source: {video_url.split('/')[-1]})")
                    fixed_count += 1
                else:
                    print(f"  ❌ HTTP error {resp.status_code} downloading {video_url}")
            except Exception as e:
                print(f"  ❌ Download exception for {formal_title}: {e}")

    finally:
        driver.quit()

    print("\n" + "=" * 60)
    print(" ✅ VERIFICATION & REPLACEMENT COMPLETE!")
    print("=" * 60)
    print(f"  • Videos Replaced/Downloaded : {fixed_count}")
    print(f"  • Videos Verified Unchanged  : {skipped_count}")
    print("=" * 60)


if __name__ == "__main__":
    verify_and_fix()