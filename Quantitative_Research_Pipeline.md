# Designing a Quantitative Research Pipeline for Trading Disclosure Signals (U.S. Congress & Corporate Insiders)

**Objective:** Develop and refine an end-to-end research pipeline to **detect short-term stock return signals** following *public disclosures* of trades by **U.S. Congress members** (per the 2012 STOCK Act disclosures) and **U.S. corporate insiders** (SEC Form 4 filings). We will use only **free, publicly available data** (regulator websites, free APIs like Yahoo Finance, etc.), focus on **daily** prices, and emphasize **1--2 week horizons** (≈5--10 trading days). The goal is to learn and apply event study techniques, regression analysis, and machine learning (ML) methods to assess if these disclosed insider and political trades contain any **residual, exploitable predictive power** -- **strictly as a research exercise** (not as investment advice).

**Key considerations:** We will **avoid look-ahead bias** by using **disclosure dates** (not trade dates) as the information release moment[^1]. We will be mindful of **data quality issues** (survivorship bias, corporate actions, etc.), **transaction costs and liquidity constraints**, and **legal/ethical factors** surrounding the use of insider and congressional trading data. All analysis will be carefully validated to ensure results are robust and not an artifact of data mining or unmet assumptions.

Both hypotheses will be tested via an event-study framework and subsequently via predictive models. We expect **no persistent long-term advantages on average** (prior studies post-STOCK Act find little or no long-run outperformance[^2]), but we will examine whether **specific conditions** (e.g. *certain committees, crises, insider role, multiple trades*) yield short-term signals.

## 1. Data Collection and Modeling

First, we design a **data schema** and retrieval plan for the relevant datasets: **political trading disclosures, corporate insider trades, and market data**. We use *only free data sources*. Table 1 outlines the core data tables.

**Table 1. Core Data Schema and Sources**

  -------------------------------------------------------------------------
  |Dataset  |  Key Fields              |   Description & Source            |
  |---------| -------------------------| ----------------------------------|
  |Congress Trades| transaction_id, person_id, ticker, trade_date, disclosure_date, transaction_type, amount (range or est.), person_attributes (e.g. chamber, party, committee, seniority) | Public disclosures of U.S. congressional stock trades by members of the House or Senate (post-STOCK Act, 2012–present). Each record identifies who traded, what was traded (ticker), the transaction date (when trade occurred, often non-public), the disclosure date (when the trade was disclosed publicly, typically within 45 days of trade ), type (e.g. buy or sale), size (often reported as a range), and attributes of the member (e.g. their chamber, party, committee memberships). Data can be sourced from official reports (e.g. House Clerk’s office & Senate disclosures) or aggregated free repositories like Quiver Quantitative  or Capitol Trades, which compile STOCK Act filings. |
  |Insider Trades | trade_id, insider_id, ticker, trade_date, filing_date, transaction_type, shares, price, role, ownership_pct| Public Form 4 filings by corporate insiders of U.S. public companies (Section 16 officers, directors, >10% shareholders). Key fields include insider identity & role (e.g. CEO, CFO, Director), the company’s ticker, transaction date (execution date), filing date (when the Form 4 was filed on EDGAR), type (buy/sell; often only open-market buys/sales are considered for signal analysis ), number of shares traded (and potentially price or value to infer trade size), and the insider’s stake (to compute % of ownership or shares outstanding traded). Free source: the SEC EDGAR RSS feeds or bulk download from the [SEC’s Insider Transactions Data Sets]  (2006–present), which contain transactions extracted from Forms 3/4/5 filings.|
  | Market Prices | date, ticker, adj_close, volume, open, high, low, close | Daily price data for all NYSE/NASDAQ-listed U.S. stocks in the sample, including adjustments for splits/dividends (adj_close). Also include daily market and sector index levels (e.g. S&P 500, NASDAQ, or Fama-French factors) and possibly a risk-free rate (e.g. 3-month T-bill yield) for benchmarking. Free sources: Yahoo Finance or AlphaVantage APIs for daily OHLCV and index data. Historical Fama-French factor data can be downloaded from Ken French’s data library (free CSVs).|


**Keys and linking:** We will use stock **ticker** plus date fields to merge trades with price data. Specifically, for each trade disclosure record (Congress or insider), we'll link it to the corresponding stock's price and index data on the relevant dates. We must ensure tickers are consistent over time:

-   We may need to **map old tickers to new tickers** in case of corporate name/ticker changes or mergers (e.g. *FB* → *META*, etc.). Using a consistent **unique identifier** (such as a permanent company ID or CUSIP) is ideal[^7].

-   We have to include **delisted stocks** to avoid *survivorship bias* (e.g., a stock that had insider buys but later went bankrupt should remain in the sample). Free data for delisted stocks is limited, but  we might reconstruct it from sources like Nasdaq/NYSE historical  listings or use survivorship-bias-free datasets for research. At minimum, any event for which we cannot retrieve subsequent stock prices (due t delisting hould be noted to avoid overstating performance.

**Data retrieval:** We'll source data via public endpoints:

-   **Congress trades:** We can use aggregated datasets from sites like *QuiverQuant* or official disclosures. For example, *QuiverQuant* provides a public dataset of 2012--2024 congressional trades (41,851 transactions) scraped from official reports[^8].

-   **Insider trades:** Use the SEC's **EDGAR** system. The SEC provides \[bulk download files of Form 4 data from 2006 onward\][^9] in CSV form (updated quarterly). We can download these or use an API like *SEC-API* or libraries (e.g., sec-edgar-downloader) to collect Form 4 filings.

-   **Price & Reference data:** Use Yahoo Finance (via yfinance in Python) or similar free APIs to get daily prices for each ticker in our trade lists, plus indices (S&P 500 for market, sector ETFs or Fama-French factors for risk adjustment).
-   **TICKER LISTINGS** Use the SEC’s CIK (Central Index Key) as a stable and unique identifier across time. By downloading the official JSON mapping file from the SEC [^65] and matching it to the tickers we intend to analyse, we can retrieve the corresponding CIKs and build a consistent company-level identifier system. This allows us to map historical ticker changes (e.g., FB → META) to a single entity. Using the current ticker associated with each CIK, we can then pull historical price data from Yahoo Finance and merge it at the company level. Since Yahoo Finance typically adjusts for stock splits, dividends, and corporate actions, this approach enables us to construct a consistent and comprehensive historical price dataset even for firms that have experienced ticker changes over time.

**Data Volume & Storage:** Expect on the order of **tens of thousands of trades**:

-   \~42k congressional trades (2012--2024)[^10].

-   Hundreds of thousands of insider trades (the SEC Form 4 dataset had \~100k late filings alone from 2004--2020[^11], and many more timely filings).

-   Price data for thousands of stocks over \~14 years (2012--2026).

This is manageable with pandas in memory (millions of rows), but careful design helps. We can store data in **normalized tables** (as above) or use a local SQL database if needed. **Unique IDs** (transaction_id, trade_id) will help track events, and we can create an **event table** that merges key info from trades with price data for analysis.

**Illustrative Code -- Data Ingestion & Schema Setup (Pandas):**

```python
import pandas as pd

# Example: Load congressional trades (e.g., from a CSV or API)

col_names = ["person_id", "person_name", "chamber",
             "party", "committee",
             "ticker", "trade_date", "disclosure_date",
             "transaction_type", "amount_low", "amount_high"]

congress_trades = pd.read_csv("congress_trades_2012_2025.csv",
                              names=col_names,
                              parse_dates=["trade_date", "disclosure_date"])


# Load insider trades from SEC (assuming we have a CSV of Form4 data)

insider_cols = ["insider_id",
                "insider_name", "role", "ticker", "trade_date", "filing_date",
                "transaction_type", "shares", "price"]

insider_trades = pd.read_csv("insider_trades_2012_2025.csv",
                             names=insider_cols,
                             parse_dates=["trade_date", "filing_date"])


# Load daily price data (could be from Yahoo Finance or another API)

price_cols = ["date", "ticker", "adj_close", "volume",
              "open", "high", "low", "close"]

prices = pd.read_csv("daily_prices_2012_2025.csv",
                     names=price_cols,
                     parse_dates=["date"])


# Example: merge a trade with price on the disclosure date (for event-day price)

# (Here we assume 'date' in prices is trading date,
#  'disclosure_date' aligned to trading days)

event_prices = pd.merge(congress_trades, prices,
                        left_on=["ticker", "disclosure_date"],
                        right_on=["ticker", "date"],
                        how="left")
```

Note: The above is a simplified illustration; in practice, we would need to parse raw data formats, handle multiple trades per filing, etc.

### Data Preprocessing & Cleaning

Careful preprocessing is crucial to ensure correctness and avoid biases:

-   **Corporate Actions & Price Adjustment:** Use **adjusted prices** for calculating returns[^12] so that **dividends, stock splits, and mergers** don't distort return calculations. For example, if using Yahoo Finance data, prefer adj_close. This ensures a consistent price series where **returns = ln(P_t / P\_{t-1})** (log-returns) or **(P_t / P\_{t-1} - 1)** (simple returns) correctly account for splits/dividends.

-   **Date Alignment:** Ensure that **disclosure/filing dates align with trading days**. If a disclosure was filed on a non-trading day (weekend/holiday) or after market hours, define the event's effective date as the **next trading day** when the market can react[^13]. Using a calendar of trading holidays (e.g., NYSE holiday calendar) is helpful to shift dates to the next open market day.

-   **Unique Events & Grouping:** Define what constitutes one "event." Often, **multiple transactions may be grouped**: e.g., if a single person files several trades on one day (or a series of forms over a few days) for the **same stock**, treat those as one combined event to avoid double-counting the signal. For instance, if a senator discloses buying the same stock on three consecutive days, we might aggregate that into one **event window** (perhaps using the last disclosure date as day 0). The Gao & Åhlqvist (2025) study, for example, **aggregated trades by the same person in the same stock on the same day**[^14]. For insiders, if a Form 4 contains multiple transactions of the same type for one stock, those could be combined. We may also consider grouping **multiple insiders** trading *the same stock around the same time* into a single *compound event* or at least include a feature capturing this cluster (discussed more in Feature Engineering).

-   **Stock Identifier Mapping:** Resolve any issues with ticker symbols: companies may change tickers or have multiple share classes. We should map trade records to a consistent **company ID** (e.g., using CUSIP or a mapping file from the SEC or other sources)[^15]. This prevents, say, treating *GOOG* vs *GOOGL* or *TWTR* vs *TWTRQ* (if delisted) as separate companies erroneously.

-   **Missing Data & Quality Checks:** Be prepared to handle missing or dirty data:

    -   Some disclosures might list stock names instead of tickers or use outdated tickers -- these need manual or automated cleanup (e.g., mapping company names to tickers via an API or reference dataset).

    -   Remove or flag clearly erroneous entries (e.g., trades with future dates or typos, negative share counts, etc.).

    -   Ensure that for every event we have sufficient price data for the estimation and event windows. If a stock has no price data (e.g., very illiquid or delisted before we gather data) we may need to drop that event or source the price from another dataset.

-   **Survivorship Bias:** Whenever possible, include stocks that were delisted or went bankrupt during the sample (so that events affecting those stocks---often negative outcomes---are not excluded). This may require sourcing  historical prices from archives or ensuring our data provider includes delisted securities.

-   **Transaction Timing:** If possible, incorporate the time-of-day of the filing:

    -   Insider Form 4 filings often occur outside of trading hours (EDGAR timestamps). If an insider files *during trading hours*, that day's price might reflect partial reaction; otherwise, the reaction starts the next trading day.

    -   Congressional trades are reported in periodic reports (Periodical Transaction Reports, PTRs) which may be filed in batches. We'll assume the market learns of a Congress trade on the **date of the PTR's publication** (or the next trading day if after hours).

-   **Compliance & Delays:** Note that Congress members have up to 45 days to disclose trades (and many **file near the deadline**[^16]) and insiders *must file within 2 business days*[^17], although **thousands of insider trades are filed late** in practice[^18]. We may include the **reporting lag** (disclosure minus trade date) as a feature, or even choose to **exclude extremely late filings** (e.g those filed past the legal window) in our analysis to focus on more timely signals[^19]. In an case, we will use *filing/disclosure date as the event date* for analysis and **never use information that wasn't public yet** at that time (to avoid look-ahead bias)[^20].

## 2. Event Definition & Abnormal Return Calculation

With clean data in place, we define the **events** and compute **returns** and **abnormal returns** around those events.

**Defining an Event:** In this context, an "event" is a public disclosure of a trade (or group of trades) by an insider or politician:

-   For **Congress trades**, an event could be a single PTR (periodic transaction report) disclosure by a member of Congress. If a report contains multiple trades, we might treat all trades by the same person reported on the same day as separate events *if they involve different stocks*, or as one combined event per stock. *For simplicity*, we can define a **Congress event** as a single member of Congress *disclosing one or more trades in a single company on a given day.* The event date (time "0") is the **disclosure date** (when the public learns of the trade)[^21].

-   For **insiders**, a natural event unit is a **Form 4 filing** by one insider for one stock. Often, a Form 4 can include multiple transactions (e.g., several buys on sequential days, or multipleoption exercises); we can aggregate multiple same-direction trades in the filing into one event for that insider-stock pair. The event date (time "0") is the **filing date** on EDGAR (or the next market date if filed after hours).

**Event Windows:** We will analyze stock performance in windows around
each event:

-   **Estimation window (pre-event):** Typically a period before the event used to establish "normal" performance or to estimate a model of expected returns. Common practice is to take a period like \[-120, -20\] trading days before the event (ending \~a month before the event to avoid contamination)[^22]. This window may be used to fit a market model for expected returns.

-   **Event window:** The days over which we measure the **impact of the event**. We focus on short-term windows like **\[0, +5 trading days\]** and **\[0, +10 trading days\]** (approximately one and two weeks after the disclosure). Here day 0 is the disclosure/filing date (if the disclosure is after market close, day 0 return can be taken as 0 and the reaction starts on day +1). We may also look at a slightly longer window (e.g., up to +21 days, one trading month) for completeness. If interested in immediate reactions, we could include day 0 (especially for intraday or same-day effects), but given daily data and possible after-hours filings, it's safer to consider returns from day +1 onward[^23].

-   We'll also consider a short **pre-event window** (e.g., \[-10, -1\] days) to check for any **run-up or drift before the disclosure**. Ideally, there should be no systematic drift before the disclosure date (otherwise it might indicate information leakage or anticipation).

**Return Calculation:** We will compute daily stock returns and *abnormal returns (AR)* for each event:
- Let \(R_{i,t}\) be the **return** of stock *i* on day *t* (we can use log returns for additivity, or simple returns for intuitive percentages; we'll assume simple returns for explanation). We'll compute daily returns from adjusted prices as:

  $$
  R_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}},
  $$

  where \(P_{i,t}\) is the adjusted closing price on day *t*.

- We define a **benchmark return** \(R_{\text{bench},t}\) for the same day *t*. For simplicity, this could be the **S&P 500's return** (or another broad market index) on day *t*. We could also use a more tailored benchmark (e.g., a sector index matching the stock's industry, or a factor model).

- The **Abnormal Return (AR)** for stock *i* on day *t* is then:

  $AR_{i,t} = R_{i,t} - R_{\text{bench},t},$

  i.e. the stock's excess return over the market/benchmark[^24]. This assumes market efficiency under the *null hypothesis* — any truly new information should result in *excess* performance.

- A **Cumulative Abnormal Return (CAR)** over a window \([a,b]\) (with *a* and *b* in trading days relative to disclosure) is the sum of ARs over that window:

  $CAR_i[a,b] = \sum_{t=a}^{b} AR_{i,t}.$

  For example, \(CAR_i[+1,+5]\) is the total abnormal return from the first trading day after disclosure through the fifth day after.

**Expected Return Models:** For robustness, we can compute AR using different expected return benchmarks:

- **Market-adjusted:** ($AR_{i,t} = R_{i,t} - R_{\text{S\&P500},t}$). Simple and doesn't require an estimation window.

- **Market model:** Estimate a regression for each stock *i* over the *estimation window* (e.g. \([-120, -20]\)) to predict its normal relationship with the market (and possibly other factors). For example, estimate via OLS:

  $R_{i,t} = \alpha_i + \beta_i \cdot R_{\text{S\&P500},t} + \epsilon_{i,t}$

Using returns prior to the event, we estimate:

$\hat{R}_{i,t} = \hat{\alpha}_i + \hat{\beta}_i \cdot R_{\text{S\&P500},t}$

as the expected return on each event-day *t*, and define:

$AR_{i,t} = R_{i,t} - \hat{R}_{i,t}$[^25] [^26].

This controls for the stock's typical market beta.

- **Multi-factor models:** Similarly, we could use Fama-French factors for expected return[^27] (more complex, requiring factor data and regression per stock). For initial analysis, a simple market adjustment may suffice, given our short horizons.

**Illustrative Code -- Computing Cumulative Abnormal Returns:**

```python
# Assume we have DataFrames: events (with 'ticker' and 'disclosure_date'),
# prices (with 'date', 'ticker', 'ret', 'mkt_ret' columns for stock and market returns)

import pandas as pd


# Ensure returns are computed in prices DataFrame

prices.sort_values(['ticker','date'], inplace=True)

prices['ret'] = prices.groupby('ticker')['adj_close'].pct_change()

# For market benchmark (e.g., S&P500 labeled by ticker 'SPY'):

market = prices[prices['ticker']=='SPY'][['date','ret']].rename(columns={'ret':'mkt_ret'})

prices = pd.merge(prices, market, on='date', how='left')


# Compute 5-day and 10-day CAR for each event

CAR5_list = []

CAR10_list = []

for _, event in events.iterrows():

    tk = event['ticker']; t0 = event['disclosure_date']

    # slice price data for this stock for the next 10 trading days

    win = prices[(prices['ticker']==tk) &
                 (prices['date']>t0) &
                 (prices['date'] <= t0 + pd.Timedelta(days=14))]

    # calculate abnormal returns = stock ret - market ret

    win['ar'] = win['ret'] - win['mkt_ret']

    # sum AR for first 5 days and first 10 days

    CAR5 = win.iloc[0:5]['ar'].sum()

    CAR10 = win.iloc[0:10]['ar'].sum()

    CAR5_list.append(CAR5); CAR10_list.append(CAR10)


events['CAR_5d'] = CAR5_list

events['CAR_10d'] = CAR10_list


# Now 'events' has two new columns with 5-day and 10-day
# post-disclosure abnormal returns for each trade event.
```

*(This pseudocode assumes* events *is a DataFrame of event records and that we have a separate DataFrame* prices *with daily returns for each ticker. In practice, one might vectorize this or use SQL/window functions for efficiency, but the loop is shown for clarity.)*

**Quality Check:** We should verify that the distribution of CARs looks reasonable and check for any biases:

- For example, **plot the average cumulative returns** around the event (from day -20 to +20) for all events and see if there's a jump post-disclosure.

- Ensure that the average pre-event abnormal returns (days -10 to -1) are near zero (no pre-event information leakage).

- Check how many events have missing price data or extreme values; investigate outliers (possibly drop or Winsorize extreme ARs if they are likely data errors).

**Significance Testing:** To formally test hypotheses:

- Use a **t-test** (or non-parametric test like *Wilcoxon signed-rank*) to see if the mean CAR is significantly different from zero across events. For example, test \(H_0: \text{mean}(CAR[+1,+5]) = 0\) vs \(H_1: \text{mean}(CAR[+1,+5]) > 0\).

- Construct **confidence intervals** for average CARs. With large samples (~40k Congress trades, many more insider trades), the Central Limit Theorem supports using normal approximations for test statistics[^28][^29].

- We can also form portfolios: e.g., on each event day, go long the stock that was bought by an insider/politician and measure the average **buy-and-hold abnormal return** over 10 days. A classic approach is a **calendar-time portfolio regression** (averaging the returns of all event-driven positions in a given month and testing if that portfolio earns significant alpha vs the market)[^30]. This is essentially an alternative to the event-by-event CAR analysis and can be implemented with regression (the intercept of the calendar-time portfolio's excess returns can indicate abnormal performance).

## 3. Feature Engineering for Predictive Modeling

Beyond the simple average effects, we will engineer a rich set of features to feed into regression and machine learning models. These features capture various dimensions of the event's characteristics, the entities involved, and market context. **The table below summarizes key feature ideas** for both congressional and insider trading events, along with the intuition for each:

## Table 2. Candidate Features for Predictive Modeling

| Feature | Description / Rationale |
| ---------- | -------------------------- |
| Trade Type (buy vs. sell) | Indicator for a purchase vs. sale. *Insider buys are generally more **informative** than sells, which may be for diversification or taxes*[^31]*. Political "negative" trades (e.g. short-selling or selling stock) have been associated with future stock drops*[^32]*, whereas routine long purchases show little effect on average.* |
| Trade Size / Volume % | Size of the trade (shares or dollar value) relative to the stock's typical **average daily volume (ADV)** or market cap. *Large trades (especially buys) that are a high percentage of daily volume or insider's ownership may signal higher conviction or material non-public info.* |
| Reporting Lag (days) | Days between **trade date** and **disclosure date**. *Shorter lags might indicate timely compliance, whereas longer lags (especially beyond the legal deadline) might indicate an attempt to hide especially informed trades*[^33]*. Late filings by insiders have been shown to yield unusually high returns, suggesting opportunistic trades*[^34]*.* We may flag or exclude extremely late reports. |
| Insider Role (rank or category) | Seniority of the insider in the company. We can create an **ordinal variable** (e.g., CEO=5, CFO=4, COO=3, Director=2, others=1)[^35]. *Top executives (CEO/CFO) are more likely to possess valuable information, so their trades may predict stronger returns*[^36]*.* |
| Politician Influence | Indicators like *Senator vs. House*, leadership position, or high-profile status. *Senators and prominent figures might have better info or attract more market attention. (E.g., public often watches trades by famous politicians; disclosures by high-profile members can move markets slightly*[^37]*.)* |
| Committee & Sector Relevance | Indicator if a Congress member's committee oversight is **related to the traded company's industry** (e.g., a defense stock traded by someone on the Armed Services Committee)[^38]. Also, for insiders, a dummy if multiple insiders in the same firm trade in the **same month** (cluster effect). *These scenarios might indicate especially strong information -- either policy-related insights or shared insider confidence*[^39]*.* |
| Multiple Trades (Clustering) | Count of how many insiders (or politicians) bought/sold the *same stock around the same time* (e.g., within a week). *Trades by multiple insiders in a short window have stronger predictive power than single isolated trades*[^40] *(a coordinated signal that something is up at the company).* Similarly, if **many politicians** are trading the same stock, that might amplify the signal. |
| Past Stock Momentum | Recent stock performance leading up to the event: e.g., returns over the past 1 month, 3 months, etc. *This gauges whether the insider/politician is potentially **buying the dip** (negative recent performance) or **buying into momentum** (positive recent performance). Momentum or reversal patterns could influence post-event returns -- e.g., an insider buying after a price drop might signal a rebound, whereas a buy after a price surge might indicate continued momentum*[^41]*.* |
| Volatility & Liquidity | Measures of recent volatility (e.g., 20-day or 60-day realized volatility of stock returns) and trading liquidity (average volume or turnover). *High volatility might indicate more information uncertainty (potential for larger moves). Low liquidity (e.g., microcaps) can mean **slower information diffusion***[^42]*, so public disclosure of trades in such stocks could have larger price impact (and also higher trading costs). We may include the stock's market cap or volume rank as a feature.* |
| Valuation & Fundamentals (optional) | Basic fundamental or valuation metrics such as P/E ratio, recent earnings surprise, etc., if available from free sources. *This can control for whether an insider is buying a seemingly undervalued stock (which might subsequently correct upwards).* |
| Temporal Features | Time-of-year or specific period indicators. E.g., a dummy if the trade was during an earnings quiet period, or in early 2020 (start of COVID) or other crisis periods[^43]. *This can capture scenarios where certain information advantages might be more pronounced (such as before a major policy announcement or during market turmoil).* |

*Feature data notes:* Many of these features can be derived from our core datasets. For instance, **trade size vs. ADV** can be computed by taking the shares traded in an insider trade and dividing by the stock's average daily volume over a prior window (say 20 days) from the prices data. **Committee-sector relevance** for Congress trades can be determined by mapping the company's industry sector (from a freely available source like Yahoo Finance or FINRA sector classifications) and checking if it matches the politician's committee domain (we'd maintain a mapping of committees to industries; e g., Financial Services Committee ↔ Banking sector). **Multiple insider trades** can be computed by counting the number of distinct insiders trading the same stock within, say, a 5-day window of the event (excluding the event insider themselves, or including them for multiple trades by the same insider). We will also engineer any needed transforms (e.g., winsorize or cap extreme values of features like size or volatility to reduce the influence of outliers).

**Illustrative Code -- Feature Engineering Examples:**

```python
# Example: add a feature for trade size as a fraction of avg daily volume (ADV) over prior 20 days

# Assuming we have 'shares' in insider_trades and 'volume' in prices

# First, compute 20-day average volume for each stock and day

prices['vol_avg20'] = prices.groupby('ticker')['volume'].rolling(window=20, min_periods=5).mean().reset_index(level=0, drop=True)

# Merge this into the insider_trades data on matching date (need the last available date <= trade_date)

insider_trades = pd.merge_asof(insider_trades.sort_values('trade_date'),
                               prices[['date','ticker','vol_avg20']].sort_values('date'),
                               left_on='trade_date', right_on='date',
                               by='ticker', direction='backward')

insider_trades['size_vs_ADV'] = insider_trades['shares'] / insider_trades['vol_avg20']


# Example: add an insider role score based on a mapping

role_rank = {"CEO":5, "CFO":4, "COO":3, "Director":2}

insider_trades['role_score'] = insider_trades['role'].map(role_rank).fillna(1)


# Example: add a political committee-industry match flag

# Suppose we have a mapping dict of committee to sectors of interest

committee_to_sector = {

"Senate Finance": ["Financials"],

"House Energy and Commerce": ["Energy", "Utilities", "Healthcare"],

# ... etc.

}

# We would need a function to map a stock ticker to its sector (from an industry classification dataset)

def stock_sector(ticker):
    return stock_info_df.loc[ticker, 'sector']  # assuming we have this lookup

# Now create the feature

congress_trades['sector'] = congress_trades['ticker'].map(stock_sector)

congress_trades['committee_sector_match'] = congress_trades.apply(
    lambda row: 1 if any(row['sector'] in sectors
                         for sectors in committee_to_sector.get(row['committee'], [])) else 0,
    axis=1)
```

*(The above code is for illustration; in practice, obtaining sectors would require a lookup, e.g., using a free API or mapping file for ticker-to-sector. Also,* pd.merge_asof *is used to align the nearest prior volume data to the trade date for computing size vs ADV.)*

**Responsible Data Usage:** Throughout, we must remember that these features and signals, even if statistically significant, **do not guarantee real-world trading profits**. We will explicitly account for practical considerations:

- **Look-Ahead Exclusion:** All features must be constructed using *only data available before or at the disclosure date*[^44]. For example, if our event is a Form 4 filed on 2021-09-01, we use price and other information up to 2021-09-01 (excluded the subsequent returns) when computing features. We won't inadvertently use future data (like future price movements or future disclosures) in feature construction.

- **Multicollinearity & Overfitting:** With many features and relatively limited events per unit time, use regularization (for regression or tree-based algorithms) and feature selection techniques. We might perform **feature importance analysis** (for trees) or **remove highly correlated features** to make models more interpretable.

- **Normalization:** Many features (trade size, market cap, etc.) will have skewed distributions. Consider transformations (log-scale, ranking, or capping outliers) to prevent a few massive trades from unduly influencing results.

## 4. Modeling Approaches for Signal Extraction

We will adopt a **progressive modeling approach**, starting from simple analyses and building up to more complex models. This ensures we validate each step and understand the signal before using black-box methods.

### 4.1 Baseline Event Study & Statistical Tests

In the initial phase, treat it purely as a classic event study and statistical inference problem:

- **Compute average CARs** for different subsets of the data:
  - All Congress buy events vs all Congress sell events.
  - All insider buys vs sells.
  - Stratify by subsets: e.g., insider CEO buys vs insider non-CEO buys; Congress trades by committee type; large vs small trade size; etc.

- For each group, calculate the **mean CAR** over the 5-day and 10-day windows, and compute a **t-statistic** for whether this mean is significantly different from zero. With large samples, a simple one-sample *t-test* is appropriate[^45]. For smaller subsamples, we could use a bootstrap to get confidence intervals for the mean CAR.

- Also calculate the **percentage of events with positive CAR** (for buys, we'd expect >50% if there's a genuine signal) or negative CAR (for sells).

- Use **visualizations**: e.g., plot the *cumulative average abnormal return (CAAR)* across time for various groups. This is a standard event study output where you plot the average of cumulative returns from some days before through after the event. Significant divergence above 0 after time 0 would indicate a positive signal. (Ensure to include confidence bands or error bars around the average line).

**Illustrative Code -- Baseline Analysis (Averaging CARs):**
```python 
import numpy as np

from scipy import stats


# Group events by some category, e.g., insider CEO buys vs others

events['is_CEO'] = (events['insider_role'] == 'CEO').astype(int)

# Calculate mean 5-day CAR for CEO vs non-CEO buys

mean_car_ceo = events.query("transaction_type=='Buy' and is_CEO==1")['CAR_5d'].mean()

mean_car_nonceo = events.query("transaction_type=='Buy' and is_CEO==0")['CAR_5d'].mean()


# t-test for difference from 0

t_stat, p_value = stats.ttest_1samp(
    events.query("transaction_type=='Buy' and is_CEO==1")['CAR_5d'],
    popmean=0
)

print(f"Mean 5-day CAR for CEO buys: {mean_car_ceo:.4f}, t-stat={t_stat:.2f}, p={p_value:.3f}")


# Example output might be:

# Mean 5-day CAR for CEO buys: 0.0125 (i.e., +1.25%), t-stat=2.5, p=0.013
```

You would repeat this kind of analysis for various hypotheses. For example, test if **Congress buy disclosures have positive mean CAR** over 5 days (H1) or if **insider CEO buys > others** (H2). If the t-statistic is large (p < 0.05), that suggests statistically significant abnormal performance.

We might find, for instance, that *on average, Congress buy disclosures do not show significant CARs* (consistent with recent studies finding no overall outperformance post-2012[^46]), *except* maybe in specific sectors or conditions. On the other hand, we might find *insider purchases* (especially by multiple high-ranking insiders) have an average positive CAR (historically, insider purchases have been associated with ~3-4% abnormal 6-month returns on average[^47], though short-horizon effects may be smaller).

- Additionally, run **cross-sectional regressions** for CAR on features to identify which factors correlate with stronger post-event performance. For example:

  \[
  CAR_{i,[1,10]} = \alpha + \beta_1 \cdot \text{(InsiderCEO}_i) + \beta_2 \cdot \text{(TradeSize%}_i) + \beta_3 \cdot \text{(Momentum}_i) + \cdots + \epsilon_i.
  \]

  *Here \(CAR_{i,[1,10]}\) is the 10-day post-event CAR for event* \(i\), *and the X variables are features (InsiderCEO = 1 if a CEO trade, TradeSize% = trade size relative to volume or market cap, etc.). We can include a variety of features to see what drives CAR. Significant \(\beta\) estimates (t-statistics) will tell us which features have predictive power. We can run these regressions using OLS with robust standard errors (to account for heteroskedasticity or clustering by date/company as needed).*

**Illustrative Code -- Cross-Sectional Regression (using vstatsmodels):**

```python 
import statsmodels.api as sm


# Prepare design matrix X and response y

features = ["is_CEO", "size_vs_ADV", "prior_1m_return", "vol_20d", "committee_sector_match"]  # etc.

X = events[features]

X = sm.add_constant(X)  # add intercept term

y = events['CAR_5d']  # using 5-day CAR as target


model = sm.OLS(y, X).fit(cov_type='HC3')  # HC3: robust standard errors

print(model.summary())

# This will show coefficients, t-stats, and p-values for each feature.
```

- Consider a **panel regression or Fama--MacBeth approach** to control for time effects:

  - A **Fama--MacBeth regression** involves running a cross-sectional regression for each period (e.g., each month or each day's events) and then averaging the coefficients across periods, assessing their statistical significance. This can account for time-variation in relationships.

  - Alternatively, use a **panel data regression** with fixed effects (e.g., add year-quarter dummies to control for market cycles, or firm fixed effects if a firm appears many times) and cluster standard errors by firm and date. This would help ensure that any detected signal isn't just picking up an unrelated effect (like particular hot sectors or a few firms driving results).

By this stage, we should have a sense of whether simple linear patterns exist (for example, we might confirm **insider buys have positive average CAR**, or that **Congress trades in certain sectors during crises had significant CAR**[^48] [^49]). If nothing stands out, it may indicate no easy linear signal; if some factors show promise (e.g., an insider's trade size or a politician's committee role correlates with CAR), those will guide our next, more complex modeling.

### 4.2 Machine Learning Models

The next step is to build predictive models that can potentially capture **non-linear interactions** and more complex patterns in the data, to see if any *hidden* signals exist. We will ensure to do this in a way that mimics a real-time predictive scenario (to avoid look-ahead bias in model training and evaluation).

**Model targets:** We can set up the ML task in two ways:

- **Regression target:** the *magnitude* of the post-event return (e.g., predicting (CAR\[+1,+5\]) as a continuous outcome).

- **Classification target:** a binary indicator of whether the event was "positive" (e.g., 1 if (CAR\[+1,+5\] > 0) or above some percentile threshold). Classification (signal vs. no-signal) is often easier, but we can try both. For example, Hangyi Zhao (2025) defined a binary target for microcap insider buys as 1 if 30-day CAR > 10%[^50] [^51].

We will start with classification (since ultimately we care about identifying outperforming trades):

- Define (Y_i = 1) if the **cumulative abnormal return** for event (i) over the next 10 days is positive (or above +X%, if we want to require a minimum threshold), and (Y_i = 0) otherwise. We must be careful if using a threshold to ensure enough positives in the data (to avoid an extremely imbalanced target).

**Model features (X):** use the feature set from **Table 2** (and possibly more). We may need to encode categorical variables (one-hot encode committee names or sectors, etc.) and scale numeric features if using certain models (not as necessary for tree-based models, but important for regression or neural networks).

**Avoiding Data Leakage:** It is crucial that when training ML models:

- We simulate **real-time prediction**. This means when training on past events to predict future events, we do not include future data. We'll likely split the data **chronologically**: e.g., use 2012--2018 events for training, 2019 for validation (model tuning), 2020--2021 for testing out-of-sample performance, and possibly keep 2022--2025 as a further holdout. We might also use a **rolling window** approach for training: for example, train on events up to 2018, test on 2019; then train on up to 2019, test on 2020, etc., and aggregate the test results. This mimics how the model would be updated in a real trading scenario.

- We ensure that if multiple events from the *same date* appear in both training and test, we handle carefully. Ideally, split by event *date* to avoid training on an event that happened *after* another event that's supposed to be in the test set.

**Candidate Models:**

1. **Logistic Regression (baseline ML):** A simple baseline classifier using all the features. This tests if a linear combination of features can predict the binary outcome. We'd apply regularization (Ridge/Lasso) to prevent overfit and possibly pick important features.

2. **Gradient Boosted Trees:** e.g., using **XGBoost** or another boosting library, which often performs well on tabular data[^52]. This can capture non-linear interactions between features (e.g., maybe an insider trade's effect is strong *only if* it's a CEO *and* the stock had a recent price drop, etc.). We'd tune parameters like the number of trees, depth, learning rate via our validation set or cross-validation.

3. **Random Forests:** As a comparison to boosted trees, a random forest (bagging of decision trees) could be used. It's often robust and provides an importance measure for features.

4. **Neural Network (optional):** A shallow feed-forward network (with a couple of hidden layers) could detect interactions as well. However, with limited data and structured features, tree-based models might suffice. (Neural networks, as shown in recent research[^53], can capture complex interactions but require careful tuning and possibly more data.)

**Model Training & Validation:** We will use *time-aware cross-validation*. For example:

- Split the dataset by year: train on 2012--2018, validate on 2019, test on 2020--2021, etc., to mimic going forward in time.

- Alternatively, use a **rolling window** CV: e.g., train on the first N years, test on the next year, then roll forward.

- Use appropriate scoring metrics: for classification, **AUC (Area Under ROC)** and **F1-score** (especially if classes are imbalanced) are useful; for regression, use **R\^2** or MSE, but ultimately we care about whether the top predictions actually yield positive returns, so classification metrics might align better with the end goal.

- **Feature importance:** For tree models, examine feature importances or SHAP values to see which features contribute most. For example, Hangyi Zhao's model found *"distance from 52-week high" as a top predictor in microcap insider trades*[^54] [^55]*.* In our broader set, we might find, say, that **trade size relative to volume** or **multiple insider flags** are highly predictive features.

**Illustrative Code -- Training a Model with Time-based Split:**
```python 
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score


# Suppose we have our events DataFrame with features and 'y' (binary outcome)

# Split data into training (2012-2018), validation (2019), test (2020-2021)

train_data = events[events['disclosure_date'] < '2019-01-01']

val_data = events[(events['disclosure_date'] >= '2019-01-01') & (events['disclosure_date'] < '2020-01-01')]

test_data = events[events['disclosure_date'] >= '2020-01-01']


X_train, y_train = train_data[features], train_data['y']

X_val, y_val = val_data[features], val_data['y']

X_test, y_test = test_data[features], test_data['y']


# Train a gradient boosting classifier

model = XGBClassifier(

    n_estimators=100, max_depth=5, learning_rate=0.1,

    subsample=0.8, colsample_bytree=0.8, random_state=42

)

model.fit(X_train, y_train, early_stopping_rounds=10,

          eval_set=[(X_val, y_val)], verbose=False)


# Evaluate on test set

test_proba = model.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, test_proba)

print(f"Test AUC: {auc:.3f}")
```

#### Note

The code above demonstrates a train/validation/test split by date and training of an XGBoost model with early stopping on the validation set. In practice, we might refine this approach with cross-validation and more detailed hyperparameter tuning.

**Defining a Signal Score:** Once a model is trained (or even from a simpler method), we need to define a *signal* that can be used in a backtest:

- For a regression model predicting (CAR\[+1,+5\]), the predicted value itself can be the **signal score** (expected abnormal return). We could rank events by this predicted CAR.

- For a classifier predicting probability of positive CAR, we could use the **predicted probability** ( \hat{P}(CAR \> 0) ) as a signal score. Alternatively, if we want a single ranking that considers both probability and magnitude, we might multiply the probability by the expected magnitude (effectively the model's estimate of *expected return*).

- Simpler: Even before complex ML, a *hand-crafted signal* could be tested. For example, one might compute a score = role_score + normalized_trade_size + (# insiders trading same stock in last week) as a simple heuristic, and check if higher scores correlate with higher subsequent returns.

**Model Interpretability:** After fitting any model, we will analyze which factors are driving the predictions:

- For linear models, look at the coefficients (\beta).

- For tree-based models, use feature importances or SHAP values to confirm which features are most influential. For instance, we might discover that *"Insider role = CEO" combined with "large trade size"* and *"recent price drop"* is a potent mix for predicting positive CARs, which would align with intuition and prior studies[^56].

- We should be cautious to avoid overinterpreting any single model run. Use the validation set and re-training on different periods to see if the same features consistently appear important.

## 5. Backtesting & Performance Evaluation

Finally, we design a **backtesting framework** to simulate how an investor might attempt to use these disclosure-based signals in real time. This will help evaluate *out-of-sample performance* under realistic conditions and test if the signals hold up after considering trading constraints.

### 5.1 Backtest Design

**Event-driven Portfolio Construction:** We will simulate a simple *rolling portfolio* that takes positions in stocks following the events:

- **Entry rule:** On each disclosure **date t (day 0)**, identify all new events (insider or political disclosures) on that day. Calculate each event's **signal score** (e.g., from our model or a simpler heuristic). **Select the top N** events (or top X%) for investment on that day. For Congress and insiders, we might separate strategies or combine them, but initially consider them separately.

- **Position sizing:** For simplicity, allocate equal weight to each selected event (e.g. if 5 events are selected on a day, invest 20% of the notional portfolio in each). Alternatively, weight by confidence (higher score = larger weight), but equal-weight avoids overfitting and is easier to analyze.

- **Holding period:** Hold each position for a fixed **horizon** (e.g., 5 trading days or 10 trading days). After the holding period, remove the position (take profit/loss). We can experiment with holding length; given our focus, 5 or 10 days are natural choices. We can also test both (forming two strategies).

- **Overlap and rebalancing:** Since events (and thus entries) can happen almost daily, our portfolio will potentially hold positions initiated on different days. We will **rebalance once per day**: each day's closing prices, we check:

  - Remove any positions that have reached their exit date (e.g., positions held for 5 days are sold on the 5th day's close).

  - Add new positions for today's signals (using that day's closing price as the entry).

  - Rebalance weights if needed (to maintain equal weighting among current positions). However, if using equal initial weights and not rebalancing, we can simply track each position's returns independently (this is easier to implement -- essentially simulating separate *trade P&Ls* and averaging them).

- **Long/short or long-only:** If our signal can identify negative outcomes (especially for Congress "sells" or insider sales), we could simulate taking short positions for negative signals. However, shorting may be unrealistic for some small stocks (and adds complexity). We might start with a **long-only strategy** focusing on purchase signals (and just avoid the others). We can later consider a long-short approach (long top decile of predicted CARs, short bottom decile) to see if that boosts returns --- but this requires caution due to real-world shorting costs.

**Backtesting Example:**

Suppose on Jan 15, 2024 there are 3 new insider buy filings and 1 congressional PTR disclosure of a purchase. Our model assigns scores and we pick the top 2 events (say they are buys of Company A and Company B). We will **buy shares of A and B at Jan 15's close or Jan 16's open** (depending on data availability/assumption) and hold for, e.g., 5 trading days, selling at the Jan 22 close. We repeat this daily. Thus, on Jan 16 we might enter new positions that will be sold on Jan 23, and so on. At any given time the portfolio may hold positions entered over the last 5 days.

We need to track the performance of this strategy through time.

**Illustrative Pseudo-Code -- Backtesting Loop:**

```python 
holding_period = 5  # hold for 5 trading days after entry

initial_capital = 1_000_000  # $1,000,000 for example

portfolio = []  # list to keep current open positions with details

portfolio_values = []  # to track portfolio value over time

equity = initial_capital


# Assume we have a list of all trading days in our sample

for current_date in trading_calendar:

    # 1. Remove expired positions (held for >= holding_period days)

    new_portfolio = []

    for pos in portfolio:

        if (current_date - pos['entry_date']).days >= holding_period or pos['exit_flag']:

            # calculate profit for this position up to yesterday's close

            exit_price = get_price(pos['ticker'], current_date)  # price at exit (current_date open or prev close)

            pos_return = (exit_price / pos['entry_price']) - 1

            equity *= (1 + pos_return * pos['weight'])  # update equity with this position's return

        else:

            new_portfolio.append(pos)  # keep position if still in holding window

    portfolio = new_portfolio


    # 2. Identify today's events and generate signals

    todays_events = events[events['disclosure_date'] == current_date]

    if not todays_events.empty:

        # Score each event (using trained model or heuristic)

        X_today = todays_events[feature_cols]

        scores = model.predict_proba(X_today)[:,1]  # probability of positive CAR

        todays_events = todays_events.assign(signal_score=scores)

        # select top N or top percentile of signals

        selected_events = todays_events.nlargest(5, 'signal_score')  # pick top 5 signals today, for example


        # 3. Enter new positions for selected events

        # Determine allocation per position (equal-weight)

        if selected_events.shape[0] > 0:

            alloc = equity / selected_events.shape[0]

            for _, ev in selected_events.iterrows():

                entry_px = get_price(ev['ticker'], current_date)  # buy at today's closing price

                position = {

                    'ticker': ev['ticker'],

                    'entry_date': current_date,

                    'entry_price': entry_px,

                    'weight': alloc / equity,  # fraction of portfolio

                    'exit_flag': False

                }

                portfolio.append(position)

    # 4. Record current equity value (assuming positions valued at current closing prices)

    portfolio_value = equity

    for pos in portfolio:

        current_px = get_price(pos['ticker'], current_date)

        pos_value = pos['weight'] * equity * (current_px / pos['entry_price'])

        portfolio_value += pos_value - pos['weight'] * equity  # add P&L of each open position

    portfolio_values.append({'date': current_date, 'equity': portfolio_value})
```

*(This high-level pseudocode outlines a daily loop. In practice, you might vectorize this or use event-time indexing. Note:* get_price(ticker, date) *is a placeholder for retrieving price; in practice, we'd index into a price DataFrame or dictionary. The logic updates* equity *as positions are closed and tracks the portfolio's value over time in* portfolio_values*.)*

### 5.2 Performance Evaluation Metrics

After running the backtest, we will evaluate the strategy's performance with appropriate metrics, always comparing against benchmarks:

- **Cumulative return** -- Compute the growth of $1 (or the initial $1,000,000) over time to visualize the performance. Compare this to the S&P 500's growth over the same period for context.

- **Annualized Return and Volatility** -- Calculate the mean and standard deviation of daily returns, scale them to annual figures (approximately by multiplying the daily mean by 252 and daily stdev by (\sqrt{252})). From these, compute the **annualized Sharpe ratio** = (annual return minus risk-free rate) / annual volatility. This measures risk-adjusted performance. A strategy that looks good on an absolute basis might still fare poorly once volatility (risk) is considered.

- **Max Drawdown** -- The worst peak-to-trough decline in the equity curve. A high drawdown might indicate the strategy is risky or has large occasional losses.

- **Hit Rate (Win %)** -- The fraction of events or trades that resulted in positive returns. We can look at hits for 5-day and 10-day horizons. A win rate significantly above 50% (after accounting for any negative positions) would indicate a real edge.

- **Average Trade Return and Distribution** -- The mean and median CAR for the events we actually traded. Also look at the distribution of these returns (e.g., 25th percentile, 75th percentile, etc.) to understand the consistency of the signal.

- **t-Statistics for Strategy Returns** -- We can compute if the daily *excess returns* of the strategy (portfolio returns minus market returns) are significantly positive (this is like checking if the strategy has non-zero alpha). A simple approach is regressing the **daily strategy returns against market returns**; a positive intercept ((\alpha)) that is statistically significant would indicate abnormal performance not explained by market moves.

We will present results in summary tables. For example, a final result
might look like:

| Strategy (5-day hold) | Annual Return | Annual Volatility | Sharpe (rf=0%) | Max Drawdown | Hit Rate (5-day CAR > 0) |
|------------------------|--------------|-------------------|----------------|--------------|---------------------------|
| Congress Buy Disclosures (Top 10%) | +5.2% | 8%  | 0.65 | -15% | 53% |
| Insider CEO Buys (Top 10%) | +12.4% | 10% | 1.24 | -20% | 58% |
| *Market (S&P 500)* | *+8.0%* | *15%* | *0.53* | *-30%* | *~54%* |
*(Table above is an **illustrative example** -- actual results would be computed from the data. The "Market" row is shown for baseline comparison, with hypothetical 8% annual return, 15% vol, and ~0.5 Sharpe.)*

Additionally, we will analyze **turnover and trading costs**:

- **Turnover:** How many trades (positions) per year does the strategy make? With daily PTRs and many insider filings, a top-decile strategy could trade frequently. If each position is held 10 days, the portfolio turnover could be high. We'll quantify this (e.g., percent of portfolio traded per month).

- **Transaction Costs Impact:** We will estimate how a reasonable transaction cost assumption (e.g., 0.1% per trade each side, or a $0.005/share commission+slippage) would affect the net returns. High turnover strategies can see performance evaporate once costs are included. For instance, if the average trade has a 1% gross CAR but one round-trip costs 0.2%, the net might drop significantly.

### 5.3 Robustness Tests and Extensions

To ensure our findings are **not artifacts** or overfit, we'll perform several robustness checks:

- **Subperiod Analysis:** Split the sample into subperiods (e.g., 2012--2015 vs 2016--2019 vs 2020--2023) to see if any detected signal is consistent or concentrated in a particular era. This can reveal, for example, if any apparent predictability was driven by an early period (perhaps right after the STOCK Act's 2012 implementation) and then faded[^57].

- **Sector and Market Regimes:** Check performance across different market conditions. Did the signal work better in bear markets vs bull markets? (Possibly, insiders buying during market-wide crashes like March 2020 could yield strong returns if they timed the bottom.) Did congressional trades from certain industries (tech vs finance vs healthcare) behave differently?

- **Signal Decay and Holding Periods:** Try different holding periods (e.g., 3-day, 5-day, 10-day, 21-day) to see where the signal is strongest or if it mean-reverts. If a signal dissipates after, say, 7 days, it suggests an optimal holding period around a week.

- **Placebo Tests:** A rigorous step is to test *randomized fake events*. For example, randomly assign "placebo" event dates to the same set of stocks, or randomize the pairing of trades to disclosure dates, and recompute the strategy returns. This helps confirm that any observed performance isn't simply due to general market movement or luck. The expectation is that placebo signals should show no outperformance on average; if our strategy's performance is significantly higher than the placebo's, that adds confidence.

- **Out-of-Sample Validation:** We can hold out the most recent data (e.g., 2022--2023) entirely for a final test of our ML model's performance, to see if the signal holds in truly unseen data. This guards against overfitting our entire research process to historical quirks.

- **Alternate Definitions:** Redefine events and see if results change. For instance, grouping all congressional trades in a week for the same stock as one event vs. treating each disclosure as separate. If results are very sensitive to such choices, that suggests the signal might not be robust.

- **Ethical & Legal Review:** While not a typical "robustness test," we must continually consider the ethical and legal implications. Even if a pattern is found, **insider and political trading data involve ethical concerns**. Lawmakers are debating stricter rules (especially if evidence suggests they benefit from non-public info)[^58]; corporate insiders are bound by fiduciary duties and laws against trading on material non-public information. **Our research is purely academic** -- any real-world exploitation could face regulatory scrutiny, especially for political trades if laws change. Additionally, using this data must respect usage policies (e.g., if using an API or website data, ensure compliance with their terms).

## Conclusion and Next Steps

**Summary:** In this project, we outlined a comprehensive pipeline to investigate short-term market reactions to disclosed **insider and political trades**. We formalized hypotheses (e.g., **Congressional purchase disclosures leading to +CAR**, and **insider CEO buys outperforming**), structured a data acquisition and cleaning process, and designed analysis techniques ranging from classical event studies to modern machine learning models. The approach emphasizes *data integrity*, *avoiding look-ahead bias*, and *robust validation* at every step, given the pitfalls of financial data mining.

**Preliminary expectations:** Based on the literature, we might anticipate:

- Little to no **average** abnormal return for broad sets of congressional trades post-2012[^59], but potential pockets of predictability (e.g., crisis periods or certain committees[^60], and notably, some evidence of market reaction to disclosures of trades by high-profile Congress members[^61]).

- For corporate insiders, consistent with many studies: **insider buys** (especially by senior insiders and in groups) tend to precede positive stock performance[^62], whereas sells are less informative on the upside (sales may sometimes predict declines or lack future outperformance). We expect our models to pick up on known signals like *trade size, insider role, multiple-buy clusters, and recent price patterns* as key predictors of success.

**Next Steps:** With the pipeline design in place, actual implementation would proceed iteratively:

1. **Data Collection & Wrangling** -- Write scripts to fetch and parse Form 4 data (e.g., using Python's requests or sec-api for EDGAR, or by downloading SEC's CSVs) and congressional trade data (from a site like QuiverQuant or by scraping official disclosures). Clean and merge with price data.

2. **Exploratory Data Analysis** -- Plot and summarize the data (distributions of trade sizes, delays, etc.). Check how many events are in various categories. Visualize some example event stock price paths.

3. **Baseline Tests** -- Calculate CARs and test hypotheses H1 and H2. Present results in a table (e.g., mean 5-day and 10-day CAR for different subsets with t-stats).

4. **Iterative Feature Refinement** -- Based on EDA and initial results, create additional features or refine existing ones. For instance, if we see that many political trades happen in tech stocks, ensure our model accounts for sector; if insider sells show no signal, perhaps focus on buys.

5. **Train ML Models** -- Begin with logistic regression or simple classifiers, then move to more complex models like XGBoost. Use cross-validation to tune parameters and avoid overfitting. Evaluate performance on validation and test sets.

6. **Backtest Strategies** -- Using the trained model (or even simpler rules discovered), perform the rolling backtest as described. Calculate performance metrics and compare against benchmarks.

7. **Robustness & Sensitivity** -- Try variations: different time periods, sub-samples, alternative model specifications, different thresholds for signals, inclusion of transaction cost assumptions, etc., to see if the signal persists.

8. **Document Findings** -- Summarize which (if any) *exploitable signals* were identified. For example, we might conclude something like: *"Insider purchases by C-suite executives following price drops showed an average +2% abnormal return in 5 days, significant at 95% confidence, but this shrank to <1% after accounting for the bid-ask spreads of those stocks."* We will also highlight if *no* significant signal was found for certain types of trades (which is itself a valuable finding, indicating market efficiency in those areas).

Throughout the project, the emphasis remains on **learning and validation** rather than profit. Any positive findings will be tempered with discussions of **real-world frictions** (e.g., *trading costs, market impact, capacity, risk management*) and the fact that after public disclosure, much of the private information advantage may already be reflected in prices[^63] [^64]. In other words, even if our models find statistical patterns, **these are not investment recommendations** but insights into market behavior.

By following this structured approach, we ensure that our analysis is thorough, credible, and true to the spirit of an educational **quantitative research project**. Each stage -- from data gathering to hypothesis testing to modeling and backtesting -- will reinforce good research practices and help determine whether any **short-term predictive signals exist** in political and insider trading disclosures under real-world constraints.

[^1]: <https://arxiv.org/pdf/2602.06198>

[^2]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^3]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^4]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^5]: <https://arxiv.org/pdf/2602.06198>

[^6]: <https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets>

[^7]: <https://arxiv.org/pdf/2602.06198>

[^8]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^9]: <https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets>

[^10]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^11]: <https://www.hbs.edu/faculty/Shared%20Documents/conferences/2023-IMO Session8-late_filings.pdf>

[^12]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^13]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^14]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^15]: <https://arxiv.org/pdf/2602.06198>

[^16]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^17]: <https://www.hbs.edu/faculty/Shared%20Documents/conferences/2023-IMO/Session8-late_filings.pdf>

[^18]: <https://www.hbs.edu/faculty/Shared%20Documents/conferences/2023-IMO/Session8-late_filings.pdf>

[^19]: <https://arxiv.org/pdf/2602.06198>

[^20]: <https://arxiv.org/pdf/2602.06198>

[^21]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^22]: <https://www.princeton.edu/~otorres/AbnormalReturns.pdf>

[^23]: <https://arxiv.org/pdf/2602.06198>

[^24]: <https://arxiv.org/pdf/2602.06198>

[^25]: <https://www.princeton.edu/~otorres/AbnormalReturns.pdf>

[^26]: <https://www.princeton.edu/~otorres/AbnormalReturns.pdf>

[^27]: <https://arxiv.org/pdf/2602.06198>

[^28]: <https://www.princeton.edu/~otorres/AbnormalReturns.pdf>

[^29]: <https://www.princeton.edu/~otorres/AbnormalReturns.pdf>

[^30]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^31]: <https://arxiv.org/pdf/2602.06198>

[^32]: <https://corpgov.law.harvard.edu/2024/07/29/negative-trading-in-congress/>

[^33]: <https://www.hbs.edu/faculty/Shared%20Documents/conferences/2023-IMO/Session8-late_filings.pdf>

[^34]: <https://www.hbs.edu/faculty/Shared%20Documents/conferences/2023-IMO/Session8-late_filings.pdf>

[^35]: <https://arxiv.org/pdf/2602.06198>

[^36]: <https://arxiv.org/pdf/2602.06198>

[^37]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^38]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^39]: <https://arxiv.org/pdf/2602.06198>

[^40]: <https://arxiv.org/pdf/2602.06198>

[^41]: <https://arxiv.org/pdf/2602.06198>

[^42]: <https://arxiv.org/pdf/2602.06198>

[^43]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^44]: <https://arxiv.org/pdf/2602.06198>

[^45]: <https://www.princeton.edu/~otorres/AbnormalReturns.pdf>

[^46]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^47]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^48]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^49]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^50]: <https://arxiv.org/pdf/2602.06198>

[^51]: <https://arxiv.org/pdf/2602.06198>

[^52]: <https://arxiv.org/pdf/2602.06198>

[^53]: <https://arxiv.org/pdf/2602.06198>

[^54]: <https://arxiv.org/pdf/2602.06198>

[^55]: <https://arxiv.org/pdf/2602.06198>

[^56]: <https://arxiv.org/pdf/2602.06198>

[^57]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^58]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^59]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^60]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^61]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^62]: <https://arxiv.org/pdf/2602.06198>

[^63]: <http://arc.hhs.se/download.aspx?MediumId=6365>

[^64]: <https://corpgov.law.harvard.edu/2024/07/29/negative-trading-in-congress/>

[^65] https://www.sec.gov/file/company-tickers

