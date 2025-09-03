# cricket_scraper/parser.py
import re

class CommentaryParser:
    def __init__(self, soup):
        self.soup = soup

    def extract_commentary_blocks(self):
        """Extracts simple commentary text blocks."""
        commentary_blocks = self.soup.find_all("div", class_="ds-text-tight-s ds-font-regular ds-text-ui-typo-mid")
        return [div.text for div in commentary_blocks]

    def extract_ball_numbers(self):
        """Extracts the ball numbers."""
        ball_number_elements = self.soup.find_all(
            'span',
            {
                'class': "ds-text-tight-s ds-font-regular ds-mb-1 lg:ds-mb-0 lg:ds-mr-3 ds-block ds-text-center ds-text-typo-mid1"
            }
        )
        return [span.text for span in ball_number_elements]

    def extract_detailed_commentary(self):
        """Extracts detailed commentary (which includes over-ball info, bowler, batsman, etc.)."""
        detailed_comments = self.soup.find_all(
            'div',
            {
                'class': "ds-text-tight-m ds-font-regular ds-flex ds-px-3 ds-py-2 lg:ds-px-4 lg:ds-py-[10px] ds-items-start ds-select-none lg:ds-select-auto"
            }
        )
        return [div.text for div in detailed_comments]

    def parse_commentary_data(self, commentary_list):
        """Uses regular expressions to parse details from each commentary entry."""
        over_ball_pattern = re.compile(r'(\d+\.\d)([\d•Wnblbb]+)(\w+) to (\w+(\s\w+)?)(,\s(.+))?')
        data = []
        for ball in commentary_list:
            match = over_ball_pattern.search(ball)
            if match:
                over_ball, runs_str, bowler, batsman, _, _, commentary = match.groups()
                # Process runs_str to assign proper run values
                if runs_str.isdigit():
                    runs = int(runs_str)
                elif runs_str == '•':
                    runs = 0
                elif runs_str == 'W':
                    runs = "Out"
                elif runs_str[:-2].isdigit() and runs_str.endswith("nb"):
                    runs = runs_str  # leave as string to indicate no-ball
                elif runs_str[:-2].isdigit() and runs_str.endswith("lb"):
                    runs = runs_str  # leave as string to indicate leg-bye
                elif runs_str[:-1].isdigit() and runs_str.endswith("b"):
                    runs = runs_str
                elif runs_str[:-1].isdigit() and runs_str.endswith("W"):
                    runs = int(runs_str[:-1])
                elif runs_str[:-1]=='•' and runs_str.endswith("W"):
                    runs = 0
                else:
                    runs = None

                data.append({
                    'Over_Ball': over_ball,
                    'Bowler': bowler,
                    'Batsman': batsman,
                    'Runs': runs,
                    'run_str': runs_str,
                    'Commentary': commentary
                })
        return data
