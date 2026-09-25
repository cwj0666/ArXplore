export function buildLoginPath(nextPath: string): string {
  return `/login/?next=${encodeURIComponent(nextPath)}`;
}
