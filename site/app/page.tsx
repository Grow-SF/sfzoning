import Credits, { SOURCE_URL } from "@/components/Credits";
import Sources from "@/components/Sources";
import Story from "@/components/Story";
import Explorer from "@/components/Explorer";
import { citedSourceIds } from "@/content/sources";
import { STORY } from "@/content/story";
import styles from "@/components/endmatter.module.css";

export default function Home() {
  return (
    <main>
      <Story steps={STORY} />
      <Explorer />
      <footer className={styles.endmatter}>
        <div className={styles.page}>
          <p className={styles.kicker}>Draft</p>
          <p>
            This story is a draft. Read <a href="/about">how the maps were made</a>, how accurate they are, and what is
            still under review.
          </p>
          <p>
            The code is open source under the AGPL-3.0; the story, research and processed maps are licensed CC
            BY-NC-SA 4.0 for non-commercial reuse. See the project <a href={SOURCE_URL}>on GitHub</a>, where
            corrections and contributions are welcome.
          </p>
          <Sources ids={citedSourceIds(STORY)} />
          <Credits />
        </div>
      </footer>
    </main>
  );
}
