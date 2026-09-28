import { render, screen } from "@testing-library/react";
import { createMemoryRouter, RouterProvider, useLocation } from "react-router";
import { describe, expect, it } from "vitest";

import { PapersAliasRedirect } from "./PapersAliasRedirect";

function LocationProbe() {
  const location = useLocation();
  return <p>{`${location.pathname}${location.search}${location.hash}`}</p>;
}

describe("PapersAliasRedirect", () => {
  it("replaces /papers/ with the list and keeps the query and hash", async () => {
    const router = createMemoryRouter(
      [
        { path: "/papers/", Component: PapersAliasRedirect },
        { path: "/", Component: LocationProbe },
      ],
      { initialEntries: ["/start", "/papers/?q=rag&sort=upvotes&page=2#top"], initialIndex: 1 },
    );
    render(<RouterProvider router={router} />);

    expect(await screen.findByText("/?q=rag&sort=upvotes&page=2#top")).toBeInTheDocument();
    expect(router.state.historyAction).toBe("REPLACE");
  });

  it("redirects a bare /papers/ to the root", async () => {
    const router = createMemoryRouter(
      [
        { path: "/papers/", Component: PapersAliasRedirect },
        { path: "/", Component: LocationProbe },
      ],
      { initialEntries: ["/papers"] },
    );
    render(<RouterProvider router={router} />);

    expect(await screen.findByText("/")).toBeInTheDocument();
  });
});
