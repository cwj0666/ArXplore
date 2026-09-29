import { Navigate, useLocation } from "react-router";

export function PapersAliasRedirect() {
  const { search, hash } = useLocation();
  return <Navigate replace to={{ pathname: "/", search, hash }} />;
}
