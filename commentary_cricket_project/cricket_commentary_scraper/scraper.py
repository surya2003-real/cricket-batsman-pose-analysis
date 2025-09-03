# cricket_scraper/scraper.py
import time
from selenium import webdriver
from bs4 import BeautifulSoup

class CricketScraper:
    def __init__(self, url, scroll_pause_time=0.5, buffer_px=650, increment=300, headless=False):
        self.url = url
        self.scroll_pause_time = scroll_pause_time
        self.buffer_px = buffer_px        # number of pixels above the footer/ads to stop scrolling
        self.increment = increment          # scroll increment
        self.headless = headless
        self.driver = self._init_driver()
        self.soup = None

    def _init_driver(self):
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
        driver = webdriver.Chrome(options=options)
        return driver

    def load_page(self):
        self.driver.get(self.url)
        # Wait for the initial page load (adjust as needed)
        time.sleep(5)

    def scroll_page(self):
        while True:
            current_scroll = self.driver.execute_script("return window.pageYOffset;")
            total_height = self.driver.execute_script("return document.body.scrollHeight;")
            target_scroll = current_scroll + self.increment

            self.driver.execute_script("window.scrollTo(0, arguments[0]);", target_scroll)
            time.sleep(self.scroll_pause_time)
            new_total_height = self.driver.execute_script("return document.body.scrollHeight;")

            # If no new content loaded and our target is near the safe threshold, exit
            if new_total_height <= total_height and target_scroll >= total_height - self.buffer_px:
                break
            else:
                print(f"Scrolled to {target_scroll} of {total_height} (new_total_height: {new_total_height}).")

        print("Scrolling completed.")

    def get_page_soup(self):
        html = self.driver.page_source
        self.soup = BeautifulSoup(html, "html.parser")
        return self.soup

    def quit(self):
        self.driver.quit()
