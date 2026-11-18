import Link from "next/link";
import SiteFooter from "@/components/SiteFooter";
import Story from "@/components/Story";
import { STORY } from "@/content/story";
import styles from "@/components/endmatter.module.css";

export default function Home() {
  return (
    <main>
      <Story steps={STORY} />
      <p className={styles.more}>
        <Link href="/maps">Explore every map →</Link>
      </p>
      <SiteFooter />
    </main>
  );
}
