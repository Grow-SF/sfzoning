{
  description = "sfzoning.com: the system tools the pipeline and site build with";

  # One pinned nixpkgs commit (nixpkgs-unstable, 2026-11-02) supplies every system tool, so
  # every checkout and machine runs the same versions: GDAL 3.13.3, poppler 26.10.0, pmtiles
  # 1.31.2. The pin matters beyond tidiness: page renders (pdftoppm) feed the vision-answer
  # cache keys, so a poppler that renders different bytes means paid re-reads. Before moving
  # the pin, check renders with `uv run sfzmaps doctor --renders 3` (pipeline/README.md).
  # This file is a dependency of every build stage (pipeline/dvc.yaml).
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/5c9de89ec55bff981cb8da1a4ee92b10f1438ffc";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in {
      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = [
            pkgs.gdal          # gdalwarp, gdal_translate, gdaladdo, gdalbuildvrt, gdal_rasterize, gdalinfo, ogr2ogr
            pkgs.poppler-utils # pdftoppm, pdfinfo (page renders)
            pkgs.pmtiles       # the go-pmtiles CLI (tile archives)
            pkgs.uv            # Python and the project's Python packages (uv.lock)
            pkgs.nodejs_22     # the site (site/package-lock.json)
            pkgs.git
          ];
          # The pipeline checks it runs with these tools (sfzmaps doctor; sfzmaps build refuses
          # otherwise), and records the toolchain it built with.
          SFZMAPS_TOOLCHAIN = "nix:${nixpkgs.rev}";
        };
      });
    };
}
