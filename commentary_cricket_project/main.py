# main.py
import pandas as pd
from cricket_commentary_scraper.scraper import CricketScraper
from cricket_commentary_scraper.parser import CommentaryParser
from cricket_commentary_scraper.classifier import PositionClassifier

def process_url(url, output_file, headless=False):
    print(f"Processing URL: {url}")
    
    # Initialize the scraper
    scraper = CricketScraper(url, scroll_pause_time=0.5, buffer_px=650, increment=300, headless=headless)
    scraper.load_page()
    scraper.scroll_page()
    soup = scraper.get_page_soup()
    scraper.quit()
    
    # Parse the scraped page to extract detailed commentary
    parser = CommentaryParser(soup)
    detailed_comments = parser.extract_detailed_commentary()
    
    # Parse commentary data into a structured list of dictionaries
    parsed_data = parser.parse_commentary_data(detailed_comments)
    
    # Initialize the classifier and classify each commentary entry
    classifier = PositionClassifier()
    for entry in parsed_data:
        entry['Side'] = classifier.classify_side(entry['Commentary'])
    
    # Create a pandas DataFrame from the parsed data and export to an Excel file
    df = pd.DataFrame(parsed_data)
    print("Data preview:")
    print(df.head())
    print("Number of unknown positions:", (df['Side'] == "unknown position").sum())
    
    df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}\n")

def main():
    # List of ball-by-ball commentary URLs
    urls = [
        # "https://www.espncricinfo.com/series/icc-world-test-championship-2019-2021-1195334/india-vs-south-africa-2nd-test-1187008/ball-by-ball-commentary",
        # "https://www.espncricinfo.com/series/australia-tour-of-india-2013-14-647237/india-vs-australia-2nd-odi-647251/ball-by-ball-commentary",
        # "https://www.espncricinfo.com/series/australia-vs-india-2024-25-1426547/australia-vs-india-1st-test-1426555/ball-by-ball-commentary",
        # "https://www.espncricinfo.com/series/india-in-aus-2018-19-1145097/australia-vs-india-2nd-odi-1144998/ball-by-ball-commentary",
        # "https://www.espncricinfo.com/series/india-in-bangladesh-2022-23-1340842/bangladesh-vs-india-3rd-odi-1340847/ball-by-ball-commentary",
        # "https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/india-vs-new-zealand-1st-semi-final-1384437/ball-by-ball-commentary",
        # "https://www.espncricinfo.com/series/men-s-t20-asia-cup-2022-1327237/afghanistan-vs-india-11th-match-super-four-1327279/ball-by-ball-commentary",
        # "https://www.espncricinfo.com/series/icc-champions-trophy-2024-25-1459031/india-vs-pakistan-5th-match-group-a-1466418/ball-by-ball-commentary",
        # "https://www.espncricinfo.com/series/india-tour-of-west-indies-2019-1188617/west-indies-vs-india-2nd-odi-1188625/ball-by-ball-commentary",
        # "https://www.espncricinfo.com/series/aus-in-ind-2018-19-1168237/india-vs-australia-3rd-odi-1168244/ball-by-ball-commentary",
        # "https://www.espncricinfo.com/series/sri-lanka-in-india-2022-23-1348629/india-vs-sri-lanka-1st-odi-1348643/ball-by-ball-commentary",
        # "https://www.espncricinfo.com/series/india-tour-of-england-2018-1119528/england-vs-india-3rd-test-1119551/ball-by-ball-commentary",
        # "https://www.espncricinfo.com/series/west-indies-in-india-2018-19-1157747/india-vs-west-indies-1st-test-1157752/ball-by-ball-commentary"
        "https://www.espncricinfo.com/series/aus-in-ind-2018-19-1168237/india-vs-australia-2nd-t20i-1168248/ball-by-ball-commentary"
    ]
    
    # Process each URL and save to a separate Excel file.
    # The output file names use a simple numbering scheme (e.g., cricket_scores_1.xlsx, cricket_scores_2.xlsx, ...)
    for i, url in enumerate(urls, start=1):
        output_file = f"cricket_scores_{i+100}.xlsx"
        process_url(url, output_file, headless=False)  # Set headless=False if you need a visible browser window

if __name__ == "__main__":
    main()
