# L4 runtime wheelhouse integrity

The P33 inference lock has exactly 168 unique dependency wheels totaling 6,399,476,597 bytes, plus the 293,855,397-byte native wheel. The local artifact verifier reported 170 PASS and 0 non-PASS across all 168 dependencies and both native wheels. Each local wheel was checked against the lock's filename, size and SHA256.

The public Hub has zero of the 168 L4 dependency wheels. The exact missing filenames are recorded in `missing-public-l4-artifacts.json`; their source group and hashes are in `07-v032-runtime-manifest.json`. Public delivery is BLOCKED. The SDK does not replace Kaggle base Torch through normal pip dependencies; the locked cu129 Torch remains inside the isolated runtime overlay.
