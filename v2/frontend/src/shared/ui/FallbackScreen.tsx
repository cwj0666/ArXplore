import type { ReactNode } from "react";

import styles from "./FallbackScreen.module.css";

interface FallbackScreenProps {
  title: string;
  message: string;
  alert?: boolean;
  children: ReactNode;
}

export function FallbackScreen({ title, message, alert = false, children }: FallbackScreenProps) {
  return (
    <main className={styles.fallback} role={alert ? "alert" : undefined}>
      <h1>{title}</h1>
      <p>{message}</p>
      <div className={styles.actions}>{children}</div>
    </main>
  );
}
