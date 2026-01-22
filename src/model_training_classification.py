from src.logger import get_logger
from src.custom_exception import CustomException
from utils.Loader import Loader

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error


logger = get_logger(__name__)


class TrainClassificationModel:
