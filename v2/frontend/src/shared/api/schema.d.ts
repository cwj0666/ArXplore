export interface paths {
  "/bootstrap.json": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    get: operations["bootstrap"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/papers/assistant/chat/": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    get?: never;
    put?: never;
    post: operations["chatWithAssistant"];
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/papers/assistant/stream/": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    get?: never;
    put?: never;
    post: operations["streamAssistantChat"];
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/papers/list.json": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    get: operations["listPapers"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/papers/{arxiv_id}/analyze/": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    get?: never;
    put?: never;
    post: operations["analyzePaper"];
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/papers/{arxiv_id}/chat/": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    get?: never;
    put?: never;
    post: operations["chatWithPaper"];
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/papers/{arxiv_id}/chat/stream/": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    get?: never;
    put?: never;
    post: operations["streamPaperChat"];
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/papers/{arxiv_id}/detail.json": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    get: operations["getPaperDetail"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/papers/{arxiv_id}/summary/": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    get?: never;
    put?: never;
    post: operations["summarizePaper"];
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
}
export type webhooks = Record<string, never>;
export interface components {
  schemas: {
    AgentChatResponse: {
      answer: string;
      citations: components["schemas"]["CitationItem"][];
    };
    AnalysisResponse: {
      cached: boolean;
      key_findings: string[];
      overview: string | null;
    };
    BootstrapResponse: {
      available_summary_models: string[];
      default_summary_model: string;
    };
    ChatHistoryMessage: {
      content: string;
      role: "user" | "assistant";
    };
    ChatRequest: {
      history?: components["schemas"]["ChatHistoryMessage"][];
      message: string;
    };
    CitationItem: {
      arxiv_id: string;
      chunk_id: number | null;
      in_answer: boolean;
      section_title: string | null;
      title: string;
      url: string;
    };
    ErrorResponse: {
      error: string;
    };
    PaperChatResponse: {
      answer: string;
      citations: components["schemas"]["CitationItem"][];
      retrieval_mode: "hybrid" | "lexical" | "first_chunks";
    };
    PaperDetail: {
      abstract: string;
      arxiv_id: string;
      authors: string[];
      pdf_url: string | null;
      published_at: string | null;
      related_papers: components["schemas"]["RelatedPaperItem"][];
      title: string;
      upvotes: number;
    };
    PaperDetailResponse: {
      paper: components["schemas"]["PaperDetail"];
    };
    PaperListItem: {
      abstract: string;
      arxiv_id: string;
      authors: string[];
      pdf_url: string | null;
      published_at: string | null;
      title: string;
      upvotes: number;
    };
    PaperListResponse: {
      items: components["schemas"]["PaperListItem"][];
      mode: "search" | "ai";
      page: number;
      page_size: number;
      query: string;
      sort: "latest" | "upvotes";
      total_items: number;
      total_pages: number;
    };
    RateLimitedResponse: {
      error: string;
      retry_after: number;
    };
    RelatedPaperItem: {
      abstract: string;
      arxiv_id: string;
      authors: string[];
      pdf_url: string | null;
      published_at: string | null;
      relation_score: number;
      source: "local" | "arxiv";
      title: string;
      upvotes: number;
    };
    StreamChunkEvent: {
      chunk: string;
    };
    StreamCitationsEvent: {
      citations: components["schemas"]["CitationItem"][];
    };
    StreamErrorEvent: {
      error: string;
    };
    SummaryRequest: {
      model: string;
    };
    SummaryResponse: {
      cached: boolean;
      model: string;
      summary: string | null;
    };
  };
  responses: never;
  parameters: never;
  requestBodies: never;
  headers: never;
  pathItems: never;
}
export type $defs = Record<string, never>;
export interface operations {
  bootstrap: {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["BootstrapResponse"];
        };
      };
    };
  };
  chatWithAssistant: {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody: {
      content: {
        "application/json": components["schemas"]["ChatRequest"];
      };
    };
    responses: {
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["AgentChatResponse"];
        };
      };
      400: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      403: {
        headers: {
          [name: string]: unknown;
        };
        content?: never;
      };
      429: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["RateLimitedResponse"];
        };
      };
      500: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      503: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
    };
  };
  streamAssistantChat: {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody: {
      content: {
        "application/json": components["schemas"]["ChatRequest"];
      };
    };
    responses: {
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": unknown;
          "text/event-stream": string;
        };
      };
      400: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      403: {
        headers: {
          [name: string]: unknown;
        };
        content?: never;
      };
      429: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["RateLimitedResponse"];
        };
      };
      500: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      503: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
    };
  };
  listPapers: {
    parameters: {
      query?: {
        q?: string;
        sort?: string;
        mode?: string;
        page?: string;
      };
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["PaperListResponse"];
        };
      };
      500: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
    };
  };
  analyzePaper: {
    parameters: {
      query?: never;
      header?: never;
      path: {
        arxiv_id: string;
      };
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["AnalysisResponse"];
        };
      };
      400: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      403: {
        headers: {
          [name: string]: unknown;
        };
        content?: never;
      };
      404: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      429: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["RateLimitedResponse"];
        };
      };
      500: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      503: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
    };
  };
  chatWithPaper: {
    parameters: {
      query?: never;
      header?: never;
      path: {
        arxiv_id: string;
      };
      cookie?: never;
    };
    requestBody: {
      content: {
        "application/json": components["schemas"]["ChatRequest"];
      };
    };
    responses: {
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["PaperChatResponse"];
        };
      };
      400: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      403: {
        headers: {
          [name: string]: unknown;
        };
        content?: never;
      };
      404: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      429: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["RateLimitedResponse"];
        };
      };
      500: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      503: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
    };
  };
  streamPaperChat: {
    parameters: {
      query?: never;
      header?: never;
      path: {
        arxiv_id: string;
      };
      cookie?: never;
    };
    requestBody: {
      content: {
        "application/json": components["schemas"]["ChatRequest"];
      };
    };
    responses: {
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": unknown;
          "text/event-stream": string;
        };
      };
      400: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      403: {
        headers: {
          [name: string]: unknown;
        };
        content?: never;
      };
      404: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      429: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["RateLimitedResponse"];
        };
      };
      500: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      503: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
    };
  };
  getPaperDetail: {
    parameters: {
      query?: never;
      header?: never;
      path: {
        arxiv_id: string;
      };
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["PaperDetailResponse"];
        };
      };
      404: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      429: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["RateLimitedResponse"];
        };
      };
      500: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
    };
  };
  summarizePaper: {
    parameters: {
      query?: never;
      header?: never;
      path: {
        arxiv_id: string;
      };
      cookie?: never;
    };
    requestBody: {
      content: {
        "application/json": components["schemas"]["SummaryRequest"];
      };
    };
    responses: {
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["SummaryResponse"];
        };
      };
      400: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      403: {
        headers: {
          [name: string]: unknown;
        };
        content?: never;
      };
      404: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      429: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["RateLimitedResponse"];
        };
      };
      500: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      503: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
    };
  };
}
