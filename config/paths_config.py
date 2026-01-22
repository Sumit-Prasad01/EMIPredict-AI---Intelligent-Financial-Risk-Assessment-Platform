import os

# Data Ingestion
GDRIVE_LINK : str = "https://drive.google.com/file/d/1C7tcEdnRIlxwIsFnsN6F0jkpU1FRlieS/view?usp=sharing"
SAVE_DIR : str = "artifacts/raw"
SAVE_NAME = "data.csv"


RAW_DATA_PATH = "artifacts/raw/data.csv"
PROCESSED_DATA_PATH_CL = "artifacts/processed/classification"
PROCESSED_DATA_PATH_REG = "artifacts/processed/regression"

# Features Path
## Save (Classification)
X_TRAIN_PATH_CL = os.path.join(PROCESSED_DATA_PATH_CL, "X_train.pkl")
X_TEST_PATH_CL = os.path.join(PROCESSED_DATA_PATH_CL, "X_test.pkl")
y_TRAIN_PATH_CL = os.path.join(PROCESSED_DATA_PATH_CL, "y_train.pkl")
y_TEST_PATH_CL = os.path.join(PROCESSED_DATA_PATH_CL, "y_test.pkl")
ENCODER_PATH_CL = os.path.join(PROCESSED_DATA_PATH_CL, "encoder.pkl")
PROCESSOR_PATH_CL = os.path.join(PROCESSED_DATA_PATH_CL, "processor.pkl")

## Save (Regression)
X_TRAIN_PATH_REG = os.path.join(PROCESSED_DATA_PATH_REG, "X_train.pkl")
X_TEST_PATH_REG = os.path.join(PROCESSED_DATA_PATH_REG, "X_test.pkl")
y_TRAIN_PATH_REG = os.path.join(PROCESSED_DATA_PATH_REG, "y_train.pkl")
y_TEST_PATH_REG = os.path.join(PROCESSED_DATA_PATH_REG, "y_test.pkl")
ENCODER_PATH_REG = os.path.join(PROCESSED_DATA_PATH_REG, "encoder.pkl")
PROCESSOR_PATH_REG = os.path.join(PROCESSED_DATA_PATH_REG, "processor.pkl")







# load path
X_TRAIN_LOAD_PATH = 'artifacts/processed/X_train.pkl'
X_TEST_LOAD_PATH = 'artifacts/processed/X_test.pkl'
y_TRAIN_LOAD_PATH = 'artifacts/processed/y_train.pkl'
y_TEST_LOAD_PATH = 'artifacts/processed/y_test.pkl'
# ENCODER_PATH = 'artifacts/processed/encoder.pkl'
# PROCESSOR_PATH = 'artifacts/processed/processor.pkl'



# # Model Training
# MODEL_PATH = "artifacts/models"
# os.makedirs(MODEL_PATH, exist_ok=True)
# SAVE_MODEL_PATH = os.path.join(MODEL_PATH, "xgb_model.pkl")
# SAVED_MODEL_PATH = "artifacts/models/xgb_model.pkl"