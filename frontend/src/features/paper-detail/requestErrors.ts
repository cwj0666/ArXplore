import { ApiError, getErrorMessage } from "../../shared/api/http";

export function describeRequestError(error: unknown, prefix: string): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  return `${prefix}: ${getErrorMessage(error, "알 수 없는 오류")}`;
}
