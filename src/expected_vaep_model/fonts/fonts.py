import os
import matplotlib.font_manager as fm

def load_fonts(font_path):
    
    for x in os.listdir(font_path):
        if x.split(".")[-1] == "ttf":
            fm.fontManager.addfont(f"{font_path}/{x}")
            try:
                fm.FontProperties(weight=x.split("-")[-1].split(".")[0].lower(), fname=x.split("-")[0])
            except Exception:
                continue