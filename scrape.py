import os
import requests
from bs4 import BeautifulSoup

__scrape_target_dir = "temp/scrape_storage"
__root_path = "https://www.mycostech.com"
__ignored_path = [
    "/blog/",
    "/category/",
    "/tag/",
    "/404",
    "/th/",
    "https://www.mycostech.com/th"
]

def get_urls_from_sitemap(sitemap_url: str) -> list[str]:
    """Download the sitemap XML and return a list of all <loc> URLs."""
    resp = requests.get(sitemap_url)
    resp.raise_for_status()

    xml_content = resp.content
    with open(f"{__scrape_target_dir}/sitemap.xml", "wb") as f:
        f.write(xml_content)

    # Parse as XML
    soup = BeautifulSoup(xml_content, "xml")
    return [loc.text for loc in soup.find_all("loc")]

def get_page_text(url: str) -> str:
    """Download a page and return its clean text (no HTML tags)."""
    resp = requests.get(url)
    resp.raise_for_status()
    page_soup = BeautifulSoup(resp.content, "html.parser")
    
    # return page_soup.get_text(separator="\n", strip=True)

    # content mostly on <main> tag
    return page_soup.find("main").get_text(separator="\n", strip=True)

def build_filename_from_url(url: str) -> str:
    return url.replace(__root_path, "").replace("/", "_") or "home"

def main():
    if not os.path.isdir(__scrape_target_dir):
        os.mkdir(__scrape_target_dir)

    sitemap_url = f"{__root_path}/sitemap-0.xml"
    urls = get_urls_from_sitemap(sitemap_url)
    print(f"Found {len(urls)} URLs in sitemap.")

    for url in urls:
        filename = f"{__scrape_target_dir}/{build_filename_from_url(url)}.txt"

        # if it's root path
        if not filename or filename == "":
            continue

        # skip, if the file exists
        if os.path.exists(filename):
            continue

        # skip, if url contains ignored_path
        if any(ignored_path in url for ignored_path in __ignored_path):
            continue

        # print(url)
        
        print(f"\n--- Fetching {url} ---")
        text = get_page_text(url)
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        
        print(f"Finish saved content to {filename}")

if __name__ == "__main__":
    main()
