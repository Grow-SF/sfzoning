import Story from "@/components/Story";
import { STORY } from "@/content/story";

export default function Home() {
  return (
    <main>
      {process.env.NODE_ENV !== "production" && (
        <p className="draft-banner">Draft copy, not reviewed. Do not publish.</p>
      )}
      <Story steps={STORY} />
    </main>
  );
}
