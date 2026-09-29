import { useSearchParams } from "react-router";

import { AssistantChat } from "../features/assistant/AssistantChat";
import { AssistantHero } from "../features/assistant/AssistantHero";
import styles from "./AssistantPage.module.css";

export default function AssistantPage() {
  const [searchParams] = useSearchParams();

  return (
    <main className={styles.page}>
      <div className={styles.inner}>
        <AssistantHero />
        <AssistantChat initialQuery={searchParams.get("q") ?? ""} />
      </div>
    </main>
  );
}
