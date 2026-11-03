from sfzmaps import prune, ship


def test_plan_keeps_what_main_and_open_prs_use_and_deletes_the_rest():
    sha = lambda c: c * 64
    small, big, old = "a" * 32, "b" * 32, "c" * 32 + ".dir"
    main = ({"shipped_sha": sha("1"), "keys_sha": sha("2")},
            {"version": 2, "objects": {small: 10, big: 9_000_000}, "where": {small: sha("d")},
             "data_packs": {sha("d"): 500}, "packs": {sha("e"): 70}})
    pr = ({"shipped_sha": sha("3"), "keys_sha": sha("2")},          # a v1 manifest: all alone
          {"version": 1, "objects": {old: 5}, "packs": {sha("e"): 70}})
    used = prune.used_names(*main) | prune.used_names(*pr, records=[sha("f")])
    keep = {f"cache/shipped/{sha('1')}.json": 1, f"cache/shipped/{sha('3')}.json": 1,
            f"cache/needed/{sha('2')}.keys.gz": 1, f"cache/datapacks/{sha('d')}.zip": 500,
            f"cache/packs/{sha('e')}.zip": 70, f"cache/packs/{sha('e')}.keys.gz": 7,
            f"cache/packs/{sha('f')}.zip": 70, f"cache/packs/{sha('f')}.keys.gz": 7,
            ship.object_name(big): 9_000_000, ship.object_name(old): 5}
    gone = {ship.object_name(small): 10,                            # now in a data pack
            f"cache/shipped/{sha('4')}.json": 1, f"cache/packs/{sha('9')}.zip": 3,
            f"cache/datapacks/{sha('8')}.zip": 4}
    held = {**keep, **gone, "cache/notes.txt": 2, "tiles/1948.pmtiles": 99}
    p = prune.plan(held, used)
    assert (p.keep, p.delete, p.missing) == (keep, gone, [])
    assert p.unrecognized == {"cache/notes.txt": 2}                 # reported, never deleted
    assert prune.by_kind(gone)["dvc single objects"] == (1, 10)
    del held[ship.object_name(big)]
    assert prune.plan(held, used).missing == [ship.object_name(big)]
