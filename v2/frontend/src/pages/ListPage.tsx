import { ListHeader } from "../features/papers-list/ListHeader";
import { ListResults } from "../features/papers-list/ListResults";
import { ListSearchPanel } from "../features/papers-list/ListSearchPanel";
import { useListPageState } from "../features/papers-list/useListPageState";
import styles from "./ListPage.module.css";

export default function ListPage() {
  const { params, queryInput, setQueryInput, listQuery, submitSearch, changeSort, changeMode } =
    useListPageState();

  return (
    <div className={styles.page}>
      <div className={styles.container}>
        <ListHeader />
        <ListSearchPanel
          mode={params.mode}
          queryInput={queryInput}
          onModeChange={changeMode}
          onQueryInputChange={setQueryInput}
          onSubmit={submitSearch}
        />
        <ListResults params={params} listQuery={listQuery} onSortChange={changeSort} />
      </div>
    </div>
  );
}
