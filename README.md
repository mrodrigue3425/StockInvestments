# Really Neat Investment Project

## Table of Contents

- [Overview](#overview)
- [Capitol Gains](#capitol-gains)

## Overview

Alex and I had a great conversation over a Catalan beer. Here are some ideas we discussed.

## Capitol Gains

### Premise

US Congress members are legally required to declare their investment portfolios to the SEC. We can imitate these portfolios, backtest them, and use these to build our own in real life.

![Stonks](figures/stonks.png)

### Goals

- Scrape this data and choose our stocks based on our findings.
- Build a systematic strategy that tracks trades made by US Congress members.
- Test whether mimicking these trades outperforms index funds.

### Key Questions

- Do congressional trades actually outperform the market?
- Is there a lag between trade and disclosure that kills edge?
- Which members historically perform best?
- Should we copy:
  - all trades?
  - only large trades?
  - only certain sectors?
- How often should we rebalance?

### Data Sources and Examples

- SEC filings: check [EDGAR](https://www.sec.gov/edgar/search/).
- [Quiverquant](https://www.quiverquant.com/): tool for tracking congress trading.
- [Capitol Trades](https://www.capitoltrades.com/): monitor for trades done by poloticians.
- [Open Secrets](https://www.opensecrets.org/about): Data on U.S. campaign finance and lobbying activities.

### Strategy Options

### 1. Copycat

- Copy all disclosed above some threshold.
- Equal-weight positions.
- Rebalance weekly/monthly.

### 2. Best Performers

- Identify top-performing poloticians historically
- Only copy their trades.
- Question: how many should we follow?

### 3. Sector Signals

- If many politicians buy a certain sector, adjust weights to favor that sector.

### 4. Delay Adjusted

- Only copy if it still makes sense.
- Filter by momentum, trend etc.

## Technical Plan

### Data Pipeline

- Extract: Scrape disclosure data.
- Transform: Parse trades.
- Load: Store in database and generate signals.

### Tools

- Selenium, BeautifulSoup.
- SQLite.
- Broker API.

## Backtesting

- Pull historical disclosure data.
- Simulate copying trades.
- Compare:
  - S&P 500
  - Nasdaq
  - random
- Metrics:
  - returns
  - drawdown
  - volatility
  - sharpe ratio

## Plan

**Phase 1: Research**

- Manually track top 5 politicians.
- Log trades.
- Simulate performance.

**Phase 2: Build Pipeline**

- Automate scraper
- Build database

**Phase 3: Backtest**

- Build back testing framework.

**Phase 4: Deploy**

- Invest cautiously
