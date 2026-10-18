---
title: Incident Response Runbook
department: Engineering
access: engineering, exec
updated: 2026-06-02
---
# Incident Response Runbook

## Severity Levels
P1: full outage of payments or a data breach affecting customers. P2: degraded payments (error rate above 2%) or a major feature unavailable. P3: minor bug with a workaround.

## Response Targets
For a P1 incident, the on-call engineer must acknowledge the page within 5 minutes and an Incident Commander must be assigned within 15 minutes. Customer-facing status page updates are posted every 30 minutes until resolution. P2 incidents must be acknowledged within 15 minutes.

## Escalation
Page the on-call engineer through PagerDuty using the "payments-core" service. If there is no acknowledgement within 10 minutes, escalate to the Engineering Manager on call, then to the VP of Engineering. For a suspected data breach, also page the security on-call and notify the Data Protection Officer immediately.

## Regulatory Reporting
Cyber security incidents must be reported to CERT-In within 6 hours of being noticed. The Compliance team owns this filing.

## Postmortems
A blameless postmortem is required for every P1 and P2 incident and must be published within 5 business days, with action items tracked in Jira.
