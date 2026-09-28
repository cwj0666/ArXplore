import styles from "./AssistantHero.module.css";

export function AssistantHero() {
  return (
    <div className={styles.hero}>
      <h1>AI 어시스턴트</h1>
      <p>관심 있는 주제를 질문하면 관련 논문을 찾아 답변합니다. 추천 논문은 클릭 가능한 링크로 정리됩니다.</p>
    </div>
  );
}
