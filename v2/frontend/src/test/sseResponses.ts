export interface ControlledStream {
  response: Response;
  push: (event: unknown) => void;
  finish: () => void;
}

export function controlledSseResponse(signal: AbortSignal | null | undefined): ControlledStream {
  const encoder = new TextEncoder();
  let controllerRef: ReadableStreamDefaultController<Uint8Array> | null = null;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controllerRef = controller;
      signal?.addEventListener("abort", () => {
        controller.error(signal.reason);
      });
    },
  });
  const write = (text: string) => {
    controllerRef?.enqueue(encoder.encode(text));
  };
  return {
    response: new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } }),
    push: (event) => {
      write(`data: ${JSON.stringify(event)}\n\n`);
    },
    finish: () => {
      write("data: [DONE]\n\n");
      controllerRef?.close();
    },
  };
}

export function jsonResponse(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

export function requestUrl(input: RequestInfo | URL): string {
  if (typeof input === "string") {
    return input;
  }
  return input instanceof URL ? input.href : input.url;
}

export function requestJsonBody(init: RequestInit | undefined): unknown {
  const body = init?.body;
  return typeof body === "string" ? JSON.parse(body) : undefined;
}
