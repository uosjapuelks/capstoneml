from pathlib import Path
from os import listdir, path

ROOT_DIR = (Path.cwd()).parent

DATA_DIR = Path(ROOT_DIR, "datasets")
RAW_DIR = Path(DATA_DIR, "raw")
SELF_DIR = Path(DATA_DIR, "self")
TEST_DIR = Path(DATA_DIR, "test")
ACCEL_DIR = Path(RAW_DIR, "accel")
GYRO_DIR = Path(RAW_DIR, "gyro")

MODEL_DIR = Path(ROOT_DIR, "models")
WEIGHT_DIR = Path(MODEL_DIR, "weights")

BITS_DIR = Path(ROOT_DIR, "mlp_bitstreams")
BITS_T_DIR = Path(ROOT_DIR, "mlp_bitstreams_testing")

# paths.get('<NAME>')

paths = {
    'ROOT_DIR' : (Path.cwd()).parent,
    'DATA_DIR' : Path(ROOT_DIR),
    'RAW_DIR' : Path(RAW_DIR),
    'SELF_DIR' : Path(SELF_DIR),
    'TEST_DIR' : Path(TEST_DIR),
    'ACCEL_DIR' : Path(ACCEL_DIR),
    'GYRO_DIR' : Path(GYRO_DIR),
    'MODEL_DIR' : Path(MODEL_DIR),
    'WEIGHT_DIR' : Path(WEIGHT_DIR),
    'BITS_DIR' : Path(BITS_DIR),
    'BITS_T_DIR' : Path(BITS_T_DIR)
}

for p in paths.values():
    p.mkdir(exist_ok=True)

