---
title: Engineering Onboarding Guide
department: Engineering
access: engineering
updated: 2026-01-12
---
# Engineering Onboarding Guide

## Week 1
Set up your laptop using the bootstrap script in the dev-setup repository, request access to GitHub, AWS (read-only) and Datadog through the access portal, and complete the secure coding training. Your onboarding buddy will pair with you on your first pull request.

## Code Review
Every pull request needs at least 2 approvals, and at least one approval must come from a code owner. Pull requests must pass CI, including unit tests, linting and the secret scanner, before merging. Keep pull requests under 400 changed lines where possible.

## Deployments
Production deployments happen through the CD pipeline only; manual deployments are not allowed. There is a deployment freeze from 25 December to 1 January and during the Diwali sale week. New engineers get production write access after 90 days and completion of the on-call shadowing program.

## Tech Stack
Backend services are written in Go and Python, the web app uses TypeScript with React, and data lives in PostgreSQL and Redis on AWS.
