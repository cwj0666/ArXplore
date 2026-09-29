import styles from "./StatusScreen.module.css";

export function StatusScreen({ message, role }: { message: string; role: "status" | "alert" }) {
  return (
    <div className={styles.status} role={role}>
      {message}
    </div>
  );
}

export function PreparingScreen() {
  return <StatusScreen message="앱을 준비하는 중입니다." role="status" />;
}
