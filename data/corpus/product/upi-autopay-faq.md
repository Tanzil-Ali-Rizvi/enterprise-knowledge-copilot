---
title: UPI AutoPay Product FAQ
department: Product
access: all
updated: 2026-07-18
---
# UPI AutoPay Product FAQ

## What is Northwind AutoPay?
Northwind AutoPay lets merchants collect recurring payments such as subscriptions and EMIs through UPI mandates that customers approve once in their UPI app.

## Mandate Limits
Customers can approve mandates of up to INR 15,000 per debit without entering a PIN for each debit. Debits above INR 15,000 require the customer to authenticate each debit with their UPI PIN. For insurance premiums and mutual fund SIPs the no-PIN limit is INR 1,00,000.

## Pre-debit Notification
Customers receive a pre-debit notification at least 24 hours before every recurring debit, and can pause or cancel the mandate from the notification.

## Pricing
Merchants pay a fee of 0.9% per successful AutoPay debit, with no setup fee. Enterprise merchants processing more than INR 5 crore per month can request custom pricing from their account manager.

## Failed Debits
If a debit fails, Northwind retries automatically up to 3 times over 3 days. Merchants receive a webhook event for every failed and successful attempt.
