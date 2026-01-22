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
CLASS_WEIGHTS_PATH = os.path.join(PROCESSED_DATA_PATH_CL, "class_weights.pkl")

## Save (Regression)
X_TRAIN_PATH_REG = os.path.join(PROCESSED_DATA_PATH_REG, "X_train.pkl")
X_TEST_PATH_REG = os.path.join(PROCESSED_DATA_PATH_REG, "X_test.pkl")
y_TRAIN_PATH_REG = os.path.join(PROCESSED_DATA_PATH_REG, "y_train.pkl")
y_TEST_PATH_REG = os.path.join(PROCESSED_DATA_PATH_REG, "y_test.pkl")
ENCODER_PATH_REG = os.path.join(PROCESSED_DATA_PATH_REG, "encoder.pkl")
PROCESSOR_PATH_REG = os.path.join(PROCESSED_DATA_PATH_REG, "processor.pkl")







# load path(classification)
X_TRAIN_LOAD_PATH_CL = 'artifacts/processed/classification/X_train.pkl'
X_TEST_LOAD_PATH_CL = 'artifacts/processed/classification/X_test.pkl'
y_TRAIN_LOAD_PATH_CL = 'artifacts/processed/classification/y_train.pkl'
y_TEST_LOAD_PATH_CL = 'artifacts/processed/classification/y_test.pkl'
CLASS_WEIGHTS_PATH_CL = 'artifacts/processed/classification/class_weights.pkl'

# load path(regression)

# ENCODER_PATH = 'artifacts/processed/encoder.pkl'
# PROCESSOR_PATH = 'artifacts/processed/processor.pkl'



# Model Training(Classification)
CL_MODEL_PATH = "artifacts/models"
os.makedirs(CL_MODEL_PATH, exist_ok=True)
SAVE_CL_MODEL_PATH = os.path.join(CL_MODEL_PATH, "cl_model.pkl")
SAVED_CL_MODEL_PATH = "artifacts/models/cl_model.pkl.pkl"