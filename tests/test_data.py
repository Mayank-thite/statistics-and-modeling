import pandas as pd

from src.data import load_bank, load_bike_day, load_bike_hour

BANK_HEADER = (
    '"age";"job";"marital";"education";"default";"housing";"loan";"contact";"month";'
    '"day_of_week";"duration";"campaign";"pdays";"previous";"poutcome";"emp.var.rate";'
    '"cons.price.idx";"cons.conf.idx";"euribor3m";"nr.employed";"y"'
)
BANK_ROWS = [
    '56;"housemaid";"married";"basic.4y";"no";"no";"no";"telephone";"may";"mon";261;1;999;0;"nonexistent";1.1;93.994;-36.4;4.857;5191;"no"',
    '40;"admin.";"single";"university.degree";"no";"yes";"no";"cellular";"may";"tue";500;2;6;1;"success";-1.8;92.893;-46.2;1.313;5099.1;"yes"',
]

BIKE_HEADER = "instant,dteday,season,yr,mnth,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt"
BIKE_ROWS = [
    "1,2011-01-01,1,0,1,0,6,0,2,0.5,0.5,0.5,0.1,331,654,985",
    "2,2011-01-02,1,0,1,0,0,0,1,0.25,0.2,0.7,0.2,131,670,801",
]


def test_load_bank(tmp_path):
    p = tmp_path / "bank.csv"
    p.write_text("\n".join([BANK_HEADER, *BANK_ROWS]) + "\n")
    df = load_bank(p)
    assert "duration" not in df.columns and "y" not in df.columns
    assert df["subscribed"].tolist() == [0, 1]
    assert df["previously_contacted"].tolist() == [0, 1]
    assert pd.isna(df.loc[0, "pdays"]) and df.loc[1, "pdays"] == 6
    assert "emp_var_rate" in df.columns
    assert "duration" in load_bank(p, drop_duration=False).columns


def test_load_bike_day_denormalises(tmp_path):
    p = tmp_path / "day.csv"
    p.write_text("\n".join([BIKE_HEADER, *BIKE_ROWS]) + "\n")
    df = load_bike_day(p)
    assert df.loc[0, "temp_c"] == 20.5
    assert df.loc[1, "humidity"] == 70
    assert df.loc[0, "weather"] == "mist"
    assert df["dteday"].dtype.kind == "M"


def test_load_bike_hour_timestamp(tmp_path):
    header = BIKE_HEADER.replace("mnth,", "mnth,hr,")
    rows = [
        "1,2011-01-01,1,0,1,5,0,6,0,2,0.5,0.5,0.5,0.1,3,13,16",
        "2,2011-01-01,1,0,1,6,0,6,0,1,0.25,0.2,0.7,0.2,0,1,1",
    ]
    p = tmp_path / "hour.csv"
    p.write_text("\n".join([header, *rows]) + "\n")
    df = load_bike_hour(p)
    assert df.loc[0, "timestamp"] == pd.Timestamp("2011-01-01 05:00")
    assert df.loc[1, "timestamp"] == pd.Timestamp("2011-01-01 06:00")
