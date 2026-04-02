from fastapi import HTTPException


class NotFoundError(HTTPException):
    def __init__(self, resource: str, id: str) -> None:
        super().__init__(status_code=404, detail=f"{resource} '{id}' not found")


class ValidationError(HTTPException):
    def __init__(self, message: str) -> None:
        super().__init__(status_code=422, detail=message)


class ConflictError(HTTPException):
    def __init__(self, message: str) -> None:
        super().__init__(status_code=409, detail=message)
