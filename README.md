# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dikrimuhammad291-ship-it/gta6-trophy-forecast/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dikrimuhammad291-ship-it/gta6-trophy-forecast/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

Entry for [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/c/gta6-trophy-rarity-forecast-01) on make.fun: forecast every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy, scored by pinball loss on the logit scale.

## How it works

`main.py` (Python 3.13, standard library only) reads a case file and writes one forecast file. For each trophy it fits, on the logit scale, a straight line in log time over the last 24 hours of the case's history, with each trophy's slope shrunk toward its tier's mean slope. Log time bends the launch surge down instead of extrapolating it past 100%, and the recent window tracks the current pace.

Exophase refreshes a game's counts hours apart, not continuously, and a refresh usually dilutes the rates as new players are tracked. When the page has sat unchanged for a long time and the horizon is short, the forecast expects a refresh inside the horizon and shifts the median down by an estimated dilution (measured from refreshes in the history when there are any, else a prior that is large on launch day and fades as the player base settles).

Quantiles sit on the logit scale around the median, with a spread combining the fit's residual, a random-walk term that widens with the horizon, and a refresh term; the lower side is a little wider than the upper side. The program is deterministic: two runs write byte-identical files.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Validation

- Practice cases (reconstructed answers from the refresh totals the rules publish): pinball loss ≈ 0.22, against the house baseline's ≈ 1.51.
- Synthetic final-like cases (trending histories with refreshes, 6,510-minute horizon): beats both the persistence baseline and the fitted-curve baseline in every scenario tested.
- Deterministic across repeated runs; runs in ~0.1 s per case, far under the 8 s cap.
