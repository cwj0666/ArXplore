import type { PaperDetail } from "../../shared/api/types";
import { authorsToText, datePart } from "../../shared/lib/text";
import { CalendarIcon, HeartIcon } from "../../shared/ui/icons";
import cards from "./cards.module.css";
import styles from "./PaperHeroCard.module.css";

export function PaperHeroCard({ paper }: { paper: PaperDetail }) {
  return (
    <div className={cards.card}>
      <div className={styles.hero}>
        <div className={styles.main}>
          <h1 className={styles.title}>{paper.title}</h1>
          <div className={styles.authors}>
            <strong>저자</strong>
            <span>{authorsToText(paper.authors)}</span>
          </div>
        </div>
        <div className={styles.side}>
          <div className={styles.sideItem}>
            <CalendarIcon />
            <span>{paper.published_at ? datePart(paper.published_at) : "-"}</span>
          </div>
          <div className={styles.sideItem}>
            <HeartIcon className={styles.heartIcon} />
            <span>{paper.upvotes}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
