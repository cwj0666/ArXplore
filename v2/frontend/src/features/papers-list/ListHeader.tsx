import { Link } from "react-router";

import styles from "./ListHeader.module.css";

export function ListHeader() {
  return (
    <header className={styles.hero}>
      <h1>
        <Link to="/">ArXplore</Link>
      </h1>
    </header>
  );
}
