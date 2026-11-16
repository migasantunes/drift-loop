# AWS costs

Everything before M6 runs locally and costs nothing. This file is the budget for M6.

## Estimate (eu-west-1, before VAT, prices checked 16 Nov 2026)

| Piece | Per hour |
|---|---|
| EKS control plane | $0.10 |
| 2 x t3.medium worker nodes | about $0.09 |
| Application load balancer | about $0.03 |
| NAT gateway (avoided by using public subnets) | about $0.05 |
| **Total** | **about $0.22 without the NAT gateway, $0.27 with it** |

Not in the table, small at this scale: public IPv4 addresses (about half a cent an
hour each), disk volumes and data transfer. Use a current Kubernetes version;
versions in extended support carry a much higher control-plane fee.

- Three or four sessions of a few hours come to about $3 to $5 by the table.
  **Budget $10 to $25** to cover longer sessions, a larger node for the load test,
  VAT and leftovers found the next day.
- Left running for a month: **about $160 to $200**. This is the risk to manage.
- New AWS accounts receive sign-up credits; whether the free plan allows EKS must be
  confirmed before relying on them.

Re-check prices on the AWS pricing pages before M6; they change.

## Guardrails (all in place before the first `terraform apply`)

1. A billing alarm at $10 and another at $25.
2. Every resource tagged `project=drift-loop`.
3. `make aws-down` runs `terraform destroy`; the session checklist ends with it.
4. The day after each session, check the billing console for anything still running
   (load balancers, volumes and public IPs are the usual leftovers).

## Spend log

| Date | Session | Hours up | Cost | Notes |
|---|---|---|---|---|
| | | | | |
