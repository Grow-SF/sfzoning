import MapShell from "@/components/MapShell";

export default function Home() {
  return (
    <main>
      {process.env.NODE_ENV !== "production" && (
        <p className="draft-banner">Draft copy. Not reviewed; do not publish.</p>
      )}
      <MapShell />
      <section className="story">
        <h1>How San Francisco was zoned</h1>
        <p>
          A hundred years of zoning, read straight off the city&rsquo;s own maps. The story scrolls here; the map stays
          put behind it.
        </p>
      </section>
    </main>
  );
}
