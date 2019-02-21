import json


def create_fixed_settings(file):
    fs = {}
    fs["xypixelsize"] = 0.1040
    fs["angle_fixed"] = 31.8
    fs["sytems_magnification"] = 62.5
    fs["pixel_pitch_um"] = 6.5
    with open(file, "w") as f:
        json.dump(fs, f)


def read_fixed_settings(file):
    with open(file, "r") as fp:
        return json.load(fp)


# create_fixed_settings("fixed_settings.json")
# read_fixed_settings("fixed_settings.json")
