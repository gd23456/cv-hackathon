#!/bin/sh
# Build, install and optionally benchmark on a connected iPhone with a free
# Apple ID already signed in to Xcode. Usage: ios/scripts/device_run.sh [--bench]
set -eu
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
cd "$(dirname "$0")/.."
DERIVED="${DERIVED_DATA:-$(mktemp -d)}"
BUNDLE=com.gourabdash.kshana
DEVICE="${DEVICE_ID:-$(xcrun devicectl list devices 2>/dev/null | grep -Eo '[0-9A-F]{8}-[0-9A-F]{16}' | head -1)}"
TEAM="${TEAM_ID:-$(defaults read com.apple.dt.Xcode IDEProvisioningTeamByIdentifier | grep teamID | head -1 | sed -E 's/.*= "?([A-Z0-9]+)"?;/\1/')}"
[ -n "$DEVICE" ] || { echo "No iPhone found" >&2; exit 1; }
[ -n "$TEAM" ] || { echo "Sign in under Xcode > Settings > Accounts first" >&2; exit 1; }

xcodegen generate --quiet
xcodebuild -project Kshana.xcodeproj -scheme Kshana -configuration Release \
  -destination "id=$DEVICE" -derivedDataPath "$DERIVED" \
  -allowProvisioningUpdates -allowProvisioningDeviceRegistration \
  DEVELOPMENT_TEAM="$TEAM" CODE_SIGN_STYLE=Automatic build 2>&1 |
  grep -E 'error:|BUILD (SUCCEEDED|FAILED)'
# Don't uninstall here: removing a free account's last app also removes its trust.
# The app clears its own stale Neural Engine cache after a reinstall.
attempt=0
until xcrun devicectl device install app --device "$DEVICE" "$DERIVED/Build/Products/Release-iphoneos/Kshana.app" >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  [ "$attempt" -lt 4 ] || { echo "Install failed: check storage and that the phone is unlocked" >&2; exit 1; }
  sleep 10
done
echo "installed"

if [ "${1:-}" = "--bench" ]; then
  xcrun devicectl device process launch --console --terminate-existing --device "$DEVICE" "$BUNDLE" --bench
fi
