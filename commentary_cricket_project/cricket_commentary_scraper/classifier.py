# cricket_scraper/classifier.py
class PositionClassifier:
    def __init__(self):
        self.fine_leg_keywords = ["fine-leg", "fine leg", "backward short leg", "backward-short-leg", "deep fine leg", "long leg", "leg gully", "deep fine-leg"]
        self.third_man_keywords = ["third man", "third-man", "deep third", "deep third man", "short third man", "short third-man", " slip", "slips"]
        self.point_keywords = ["point", "backward point", "backward-point", "silly point", "silly-point", "deep cover point", "deep backward point", "deep backward"]
        self.cover_keywords = ["cover", "extra cover", "extra-cover", "deep cover", "deep extra cover"]
        self.mid_off_keywords = ["mid-off", "mid off", "deep mid-off", "deep mid off", "long-off", "long off", "straight mid off", "off"]
        self.mid_on_keywords = ["mid-on", "mid on", "deep mid-on", "deep mid on", "long on", "long-on", "deep long on", "deep long-on", "on"]
        self.mid_wicket_keywords = ["mid-wicket", "mid wicket", "deep mid-wicket", "deep mid wicket", "midwicket"]
        self.square_leg_keywords = ["square leg", "square-leg", "deep square leg", "deep square-leg"]
        self.out_keywords = ["bowled"]
        self.no_ball_keywords = ["no-ball"]

    def classify_side(self, commentary):
        commentary_lower = commentary.lower()
        if any(keyword in commentary_lower for keyword in self.fine_leg_keywords):
            return "fine leg"
        elif any(keyword in commentary_lower for keyword in self.third_man_keywords):
            return "third man"
        elif any(keyword in commentary_lower for keyword in self.point_keywords):
            return "point"
        elif any(keyword in commentary_lower for keyword in self.cover_keywords):
            return "cover"
        elif any(keyword in commentary_lower for keyword in self.mid_wicket_keywords):
            return "mid wicket"
        elif any(keyword in commentary_lower for keyword in self.square_leg_keywords):
            return "square leg"
        elif any(keyword in commentary_lower for keyword in self.mid_off_keywords):
            return "mid off"
        elif any(keyword in commentary_lower for keyword in self.mid_on_keywords):
            return "mid on"
        elif any(keyword in commentary_lower for keyword in self.out_keywords):
            return "out"
        elif any(keyword in commentary_lower for keyword in self.no_ball_keywords):
            return "no_ball"
        return "unknown position"
