import { useMutation } from "@tanstack/react-query";
import { useCallback, useEffect, useRef } from "react";

import { describeRequestError } from "./requestErrors";

export type AiSectionState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready" }
  | { status: "error"; message: string }
  | { status: "cancelled" };

interface AbortableVariables<TVariables> {
  variables: TVariables;
  signal: AbortSignal;
}

export interface AbortableMutation<TVariables, TData> {
  state: AiSectionState;
  data: TData | undefined;
  start: (variables: TVariables, onSuccess?: (data: TData) => void) => void;
  cancel: () => void;
}

export function useAbortableMutation<TVariables, TData>(
  request: (variables: TVariables, signal: AbortSignal) => Promise<TData>,
  errorPrefix: string,
): AbortableMutation<TVariables, TData> {
  const controllerRef = useRef<AbortController | null>(null);
  const mutation = useMutation<TData, Error, AbortableVariables<TVariables>>({
    mutationFn: ({ variables, signal }) => request(variables, signal),
  });
  const { mutate } = mutation;

  useEffect(
    () => () => {
      controllerRef.current?.abort();
      controllerRef.current = null;
    },
    [],
  );

  const start = useCallback(
    (variables: TVariables, onSuccess?: (data: TData) => void) => {
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;
      mutate({ variables, signal: controller.signal }, onSuccess ? { onSuccess } : {});
    },
    [mutate],
  );

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
  }, []);

  let state: AiSectionState;
  switch (mutation.status) {
    case "idle":
      state = { status: "idle" };
      break;
    case "pending":
      state = { status: "loading" };
      break;
    case "success":
      state = { status: "ready" };
      break;
    case "error":
      state = mutation.variables.signal.aborted
        ? { status: "cancelled" }
        : { status: "error", message: describeRequestError(mutation.error, errorPrefix) };
      break;
  }

  return { state, data: mutation.data, start, cancel };
}
