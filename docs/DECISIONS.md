# Decisions

Short records of choices that shape the project. Add one whenever a contract,
a tool or the evaluation protocol changes. Newest at the bottom.

## D1 - Evaluate by chronological replay, not a random split (16 Nov 2026)
A random split leaks each user's future into training and cannot represent a model
ageing. Replay is slower and gives lower numbers; they are the honest ones.
Details in `EVALUATION.md`.

## D2 - Yambda-50M as the dataset (16 Nov 2026)
Global timestamps, months of data, heavy catalogue churn, audio embeddings, fits on
a laptop. Alternatives and open questions in `DATASET.md`.

## D3 - Build the offline simulation before any service (16 Nov 2026)
The simulation is the specification for the live loop. It also means the headline
result exists by M1, long before the infrastructure does.

## D4 - Ground truth is new-to-user items (16 Nov 2026)
Music listening is dominated by repeats; scoring them rewards replaying history.
Repeats-included is kept as a sensitivity run.

## D5 - The caller filters seen items, not the model (16 Nov 2026)
A model trained two weeks ago must not recommend yesterday's listens. Only the
harness (later the serving layer) has fresh history, so `recommend` takes `seen`
as an argument.

## D6 - One canonical interaction table (16 Nov 2026)
`user_id, item_id, ts, weight`; positive interactions only; sorted by time.
Event semantics are decided once, in the dataset adapter.

## D7 - No placeholder modules; dependencies arrive with the milestone that uses them (16 Nov 2026)
Empty stubs rot and heavy unused dependencies slow CI. The planned layout is in
`ARCHITECTURE.md`; a directory appears when its first real code does.

## D8 - EKS in short sessions rather than a cheaper always-on cluster (16 Nov 2026)
EKS is the service job descriptions name. Running it only for deploy, load-test and
recording sessions keeps the total near $10 to $25. Budget and guardrails in `COSTS.md`.

## D9 - Lead plans, sub-agents implement, a fresh reviewer validates (16 Nov 2026)
Briefs with disjoint file sets allow parallel work; a reviewer that did not watch
the implementation is less likely to share its blind spots. Miguel answers each
milestone's interview checkpoint before merge.

## D10 - PSI and JSD exclude unseen items (16 Nov 2026)
First design: one extra PSI bucket for items the reference never saw. Its reference
share is zero, so after smoothing it dominated the statistic (about 5% of events on
new items already pushed PSI past the usual 0.2 threshold) and tied the value to the
smoothing constant. Now PSI and JSD are computed among reference-known items only,
and novelty is measured by new-item share alone.

## D11 - Tuning windows are excluded from headline numbers (16 Nov 2026)
Drift thresholds are chosen on the first `tuning_windows` windows. Those windows are
replayed and plotted but left out of every summary number, so no policy is scored on
the data its thresholds were fitted to.

## D12 - M0 demo findings carried into M1 (16 Nov 2026)
On the synthetic demo, drift-triggered retraining behaved like a fixed two-window
schedule (the generator's drift is smooth), and PSI/JSD carried no staleness signal
(a frozen model read lower than a fresh one). Recall also falls for every policy
because new-to-user ground truth gets harder as histories grow. None of this is
tuned away; M1 tests whether real data behaves differently and adds a bursty-drift
synthetic scenario as a controlled check of the trigger.
