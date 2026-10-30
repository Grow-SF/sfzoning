import Credits from "@/components/Credits";
import Sources from "@/components/Sources";
import Story from "@/components/Story";
import { citedSourceIds } from "@/content/sources";
import { STORY } from "@/content/story";
import styles from "@/components/endmatter.module.css";

export default function Home() {
  return (
    <main>
      {process.env.NODE_ENV !== "production" && (
        <p className="draft-banner">Draft copy, not reviewed. Do not publish.</p>
      )}
      <Story steps={STORY} />
      <footer className={styles.endmatter}>
        <div className={styles.page}>
          <p className={styles.kicker}>Draft</p>
          <p>
            This story is a draft. Read <a href="/about">how the maps were made</a>, how accurate they are, and what is
            still under review.
          </p>
          <Sources ids={citedSourceIds(STORY)} />
          <Credits />
        </div>
      </footer>
    </main>
  );
}
