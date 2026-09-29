from typing import ClassVar

INVALID_REQUEST_MESSAGE = "잘못된 요청입니다."
UNSUPPORTED_MODEL_MESSAGE = "지원하지 않는 모델입니다."
EMPTY_MESSAGE_MESSAGE = "메시지를 입력하세요."
PAPER_NOT_FOUND_MESSAGE = "논문을 찾을 수 없습니다."
AI_UNAVAILABLE_MESSAGE = "AI 기능을 사용할 수 없습니다. 서버에 OpenAI API 키가 설정되지 않았습니다."
GENERATION_ERROR_MESSAGE = "AI 결과 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
STREAM_ERROR_MESSAGE = "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
DATA_LOAD_ERROR_MESSAGE = "논문 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요."
ROUTE_NOT_FOUND_MESSAGE = "요청한 주소를 찾을 수 없습니다."
SERVER_ERROR_MESSAGE = "서버 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
CSRF_FAILURE_MESSAGE = "요청을 확인하지 못했습니다(CSRF 검증 실패). 페이지를 새로고침한 뒤 다시 시도해 주세요."
RATE_LIMITED_MESSAGE = "요청이 너무 많습니다. {retry_after}초 후 다시 시도하세요."


class ApiError(Exception):
    status_code: ClassVar[int] = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidRequestError(ApiError):
    status_code = 400


class PaperNotFoundError(ApiError):
    status_code = 404

    def __init__(self, message: str = PAPER_NOT_FOUND_MESSAGE) -> None:
        super().__init__(message)


class AiUnavailableError(ApiError):
    status_code = 503

    def __init__(self, message: str = AI_UNAVAILABLE_MESSAGE) -> None:
        super().__init__(message)


class RateLimitExceededError(Exception):
    def __init__(self, retry_after: int) -> None:
        super().__init__(RATE_LIMITED_MESSAGE.format(retry_after=retry_after))
        self.retry_after = retry_after

    @property
    def message(self) -> str:
        return RATE_LIMITED_MESSAGE.format(retry_after=self.retry_after)
