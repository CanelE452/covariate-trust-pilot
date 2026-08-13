#!/usr/bin/env bash
# Build a complete, verified recovery package for the m5dataset synthetic study.
#
# RUN THIS ON THE MACHINE THAT HAS THE REPOSITORY (the Linux host with
# ~/Documents/github/m5dataset).  It was written on the new machine, where the
# repository does not exist, so nothing here has been executed yet.
#
# It implements steps 1-16 of the recovery protocol and is deliberately
# read-only with respect to the source repository:
#   - no reset, no clean, no checkout, no restore
#   - no commit, no push, no new remote
#   - nothing is written inside the source repository
#   - nothing is excluded from the archive
#
# Usage:
#   bash make_recovery_package.sh [/path/to/m5dataset]
# Default source is ~/Documents/github/m5dataset.

set -euo pipefail

SRC="${1:-$HOME/Documents/github/m5dataset}"

# ---------------------------------------------------------------- step 0
if [ ! -d "$SRC" ]; then
    echo "SYNTHETIC_SOURCE_NOT_FOUND: $SRC does not exist" >&2
    echo "Search these before giving up:" >&2
    echo "  find \$HOME -maxdepth 5 -type d -name m5dataset" >&2
    echo "  find \$HOME -maxdepth 6 -type d \\( -name temporal_dependence -o -name decomposition_when_helps \\)" >&2
    exit 1
fi
if [ ! -d "$SRC/.git" ]; then
    echo "WARNING: $SRC has no .git; the full archive still captures everything," >&2
    echo "but the git bundle step will be skipped." >&2
fi

SRC="$(cd "$SRC" && pwd -P)"
cd "$SRC"

HEAD_SHORT="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
STAMP="$(date +%Y%m%d_%H%M%S)"

# Recovery output lives OUTSIDE the source repository, as a sibling.
OUT="$(dirname "$SRC")/_recovery_exports/m5dataset_${STAMP}"
mkdir -p "$OUT"
echo "source:   $SRC"
echo "output:   $OUT"

# ---------------------------------------------------------------- step 1-2
echo "$SRC"                                    > "$OUT/source_path.txt"
git rev-parse HEAD                             > "$OUT/git_head.txt"      2>/dev/null || true
git rev-parse --abbrev-ref HEAD                > "$OUT/git_branch.txt"    2>/dev/null || true
git status --porcelain=v1                      > "$OUT/git_status_porcelain.txt" 2>/dev/null || true
git remote -v                                  > "$OUT/git_remotes.txt"   2>/dev/null || true
git log -1 --oneline                           > "$OUT/git_log_head.txt"  2>/dev/null || true
git ls-files                                   > "$OUT/git_tracked_files.txt" 2>/dev/null || true
git ls-files --others --exclude-standard       > "$OUT/untracked_files.txt"   2>/dev/null || true
git ls-files --others --ignored --exclude-standard > "$OUT/ignored_files.txt" 2>/dev/null || true
git diff --name-only                           > "$OUT/modified_files.txt"    2>/dev/null || true
git diff --cached --name-only                  > "$OUT/staged_files.txt"      2>/dev/null || true
du -sh "$SRC"                                  > "$OUT/repo_size.txt"

# The before-state, used at the end to prove nothing changed.
cp "$OUT/git_status_porcelain.txt" "$OUT/.status_before.txt" 2>/dev/null || true

# ---------------------------------------------------------------- step 3
{
    echo "# directories whose names were recorded in the migrated audit"
    for d in experiments/temporal_dependence experiments/decomposition_when_helps \
             experiments/om_factorization_killtest experiments/unified_temporal_27_v3 \
             _docs/history results reports runs configs data src; do
        [ -e "$SRC/$d" ] && echo "EXISTS  $d" || echo "ABSENT  $d"
    done
    echo
    echo "# semantic search, names not assumed"
    find "$SRC" -path "$SRC/.git" -prune -o \
         \( -iname "*stage1*" -o -iname "*stage_1*" -o -iname "*stage3*" -o -iname "*stage_3*" \
            -o -iname "*stage4*" -o -iname "*stage_4*" -o -iname "*prereg*" -o -iname "*pre_analysis*" \
            -o -iname "*temporal_dependence*" -o -iname "*decomposition*" -o -iname "*hurdle*" \
            -o -iname "*point*" -o -iname "*factoriz*" -o -iname "*rho_interval*" -o -iname "*rho_magnitude*" \
            -o -iname "*synthetic*" -o -iname "*dgp*" -o -iname "*bootstrap*" -o -iname "*seed*" \
            -o -iname "*contrast*" -o -iname "*sweep*" -o -iname "*cell*" \) -print 2>/dev/null \
      | sed "s|^$SRC/||" | sort
} > "$OUT/scientific_candidate_paths.txt"

# ---------------------------------------------------------------- step 4
ARCHIVE="$OUT/m5dataset_full_${HEAD_SHORT}_${STAMP}.tar.gz"
echo "creating full archive (this includes .git, tracked, modified and untracked files)..."
tar czf "$ARCHIVE" -C "$(dirname "$SRC")" "$(basename "$SRC")"

# ---------------------------------------------------------------- step 5
echo "verifying archive listing..."
tar -tzf "$ARCHIVE" > "$OUT/archive_listing.txt"
BASE="$(basename "$SRC")"
ARCHIVE_OK=1
grep -q "^${BASE}/.git/HEAD$" "$OUT/archive_listing.txt" || { echo "FAIL: .git/HEAD missing"; ARCHIVE_OK=0; }
[ -s "$ARCHIVE" ] || { echo "FAIL: archive is empty"; ARCHIVE_OK=0; }
for d in experiments/temporal_dependence experiments/decomposition_when_helps _docs/history; do
    if [ -e "$SRC/$d" ]; then
        grep -q "^${BASE}/${d}/" "$OUT/archive_listing.txt" \
            || { echo "FAIL: $d exists in source but not in archive"; ARCHIVE_OK=0; }
    fi
done
# every untracked file must be inside the archive
while IFS= read -r f; do
    [ -z "$f" ] && continue
    grep -qxF "${BASE}/${f}" "$OUT/archive_listing.txt" \
        || { echo "FAIL: untracked file missing from archive: $f"; ARCHIVE_OK=0; }
done < "$OUT/untracked_files.txt"
[ "$ARCHIVE_OK" = 1 ] && echo "FULL_ARCHIVE_VERIFIED" || echo "FULL_ARCHIVE_FAIL"

# ---------------------------------------------------------------- step 6
( cd "$OUT" && sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256" \
             && sha256sum -c "$(basename "$ARCHIVE").sha256" )

# ---------------------------------------------------------------- step 7-8
if [ -d "$SRC/.git" ]; then
    BUNDLE="$OUT/m5dataset_git_${HEAD_SHORT}_${STAMP}.bundle"
    git bundle create "$BUNDLE" --all
    git bundle verify "$BUNDLE" && echo "GIT_BUNDLE_VERIFIED"
    ( cd "$OUT" && sha256sum "$(basename "$BUNDLE")" > "$(basename "$BUNDLE").sha256" \
                 && sha256sum -c "$(basename "$BUNDLE").sha256" )
fi

# ---------------------------------------------------------------- step 9
{
    echo "status,relative_path,size_bytes,sha256"
    git status --porcelain=v1 | while IFS= read -r line; do
        st="${line:0:2}"; path="${line:3}"
        path="${path%\"}"; path="${path#\"}"
        [ -f "$SRC/$path" ] || continue
        size=$(stat -c%s "$SRC/$path" 2>/dev/null || echo 0)
        hash=$(sha256sum "$SRC/$path" | cut -d' ' -f1)
        echo "\"${st}\",\"${path}\",${size},${hash}"
    done
} > "$OUT/working_tree_scientific_manifest.csv"

# ---------------------------------------------------------------- step 10-11
{
    echo "# synthetic critical artifact inventory"
    echo "# location only; no scientific interpretation is made here."
    echo
    for label in \
        "A prereg:*prereg*|*pre_analysis*|*preregist*" \
        "B paper DGP / generator:*generator*|*dgp*|*scenario*|*simulate*" \
        "C Point config/model:*point*" \
        "D Hurdle config/model:*hurdle*|*factoriz*" \
        "E Stage 1 result:*stage1*|*stage_1*" \
        "F Stage 3 diagnostic:*stage3*|*stage_3*|*prediction_curve*" \
        "G Stage 4 result:*stage4*|*stage_4*|*sweep*|*18*cell*" \
        "H bootstrap / CI:*bootstrap*|*analysis*" \
        "I seed / run manifest:*seed*|*manifest*|*run*" \
        "J plot source table:*figure*|*fig*|*plot*" \
        "K history:_docs/history/*"
    do
        name="${label%%:*}"; pats="${label#*:}"
        echo "## $name"
        found=0
        IFS='|' read -ra arr <<< "$pats"
        for p in "${arr[@]}"; do
            while IFS= read -r hit; do
                echo "  $hit"; found=1
            done < <(find "$SRC" -path "$SRC/.git" -prune -o -ipath "*$p" -print 2>/dev/null \
                     | sed "s|^$SRC/||" | grep -v '^\.git' | sort | head -40)
        done
        [ "$found" = 0 ] && echo "  NOT_FOUND"
        echo
    done
    echo "## paper-DGP misidentification check"
    echo "Any file whose text disclaims itself as the paper DGP:"
    grep -rIl --exclude-dir=.git "not the paper DGP" "$SRC" 2>/dev/null | sed "s|^$SRC/||" || echo "  (none found)"
    echo
    echo "Recorded split configurations (the paper study is the length-576 / 384-480-576 one):"
    grep -rIn --exclude-dir=.git -E "length[\"']?\s*[:=]\s*576|train_end[\"']?\s*[:=]\s*384|val_end[\"']?\s*[:=]\s*480" \
        "$SRC" 2>/dev/null | sed "s|^$SRC/||" | head -20 || echo "  (none found)"
} > "$OUT/synthetic_critical_artifact_inventory.md"

# ---------------------------------------------------------------- step 12
echo "spot-checking a few files against the archive..."
TMP="$(mktemp -d)"
{
    echo "relative_path,source_sha256,archive_sha256,match"
    grep -E '\.(json|csv|md|ya?ml|txt)$' "$OUT/archive_listing.txt" | head -8 | while IFS= read -r entry; do
        rel="${entry#${BASE}/}"
        [ -f "$SRC/$rel" ] || continue
        tar xzf "$ARCHIVE" -C "$TMP" "$entry" 2>/dev/null || continue
        a=$(sha256sum "$SRC/$rel" | cut -d' ' -f1)
        b=$(sha256sum "$TMP/$entry" | cut -d' ' -f1)
        [ "$a" = "$b" ] && m=OK || m=MISMATCH
        echo "\"$rel\",$a,$b,$m"
    done
} > "$OUT/archive_spot_check.csv"
rm -rf "$TMP"

# ---------------------------------------------------------------- step 15
{
    echo "filename,size_bytes,sha256"
    for f in "$OUT"/*; do
        [ -f "$f" ] || continue
        echo "\"$(basename "$f")\",$(stat -c%s "$f"),$(sha256sum "$f" | cut -d' ' -f1)"
    done
} > "$OUT/recovery_package_manifest.csv"

# ---------------------------------------------------------------- step 16
cd "$SRC"
git status --porcelain=v1 > "$OUT/.status_after.txt" 2>/dev/null || true
if diff -q "$OUT/.status_before.txt" "$OUT/.status_after.txt" >/dev/null 2>&1; then
    echo "SOURCE_UNTOUCHED = True"
else
    echo "SOURCE_UNTOUCHED = False  -- inspect $OUT/.status_before.txt vs .status_after.txt" >&2
fi
echo "HEAD now: $(git rev-parse HEAD 2>/dev/null || echo nogit)  (recorded: $(cat "$OUT/git_head.txt" 2>/dev/null))"
echo "branch now: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo nogit)  (recorded: $(cat "$OUT/git_branch.txt" 2>/dev/null))"

echo
echo "recovery package ready at: $OUT"
ls -la "$OUT"
