from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "customer-shopping-behavior.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Data Loading
# =========================================================

def load_data():
    """Load the raw customer shopping behaviour dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")

    return df


# =========================================================
# Data Cleaning
# =========================================================

def clean_data(df):
    """Clean and standardise the dataset."""

    df = df.copy()

    # -----------------------------------------------------
    # Standardise column names
    # -----------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
    )

    # -----------------------------------------------------
    # Remove duplicate records
    # -----------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates()

    duplicates_removed = (
        before_duplicates - len(df)
    )

    print(
        f"Duplicates removed: {duplicates_removed}"
    )

    # -----------------------------------------------------
    # Convert numeric columns
    # -----------------------------------------------------

    numeric_columns = [
        "age",
        "purchase_amount_usd",
        "review_rating",
        "previous_purchases",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # -----------------------------------------------------
    # Handle missing review ratings
    # -----------------------------------------------------

    if "review_rating" in df.columns:

        missing_ratings = (
            df["review_rating"].isna().sum()
        )

        median_rating = (
            df["review_rating"].median()
        )

        df["review_rating"] = (
            df["review_rating"]
            .fillna(median_rating)
        )

        print(
            "Missing review ratings imputed: "
            f"{missing_ratings}"
        )

    # -----------------------------------------------------
    # Convert Yes/No variables into binary variables
    # -----------------------------------------------------

    binary_columns = [
        "discount_applied",
        "promo_code_used",
        "subscription_status",
    ]

    for column in binary_columns:

        if column in df.columns:

            df[f"{column}_binary"] = (
                df[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(
                    {
                        "yes": 1,
                        "no": 0,
                    }
                )
            )

    # Cleaner subscription indicator
    if "subscription_status_binary" in df.columns:

        df["subscriber_binary"] = (
            df["subscription_status_binary"]
        )

    return df


# =========================================================
# Feature Engineering
# =========================================================

def create_features(df):
    """Create analytical variables for research."""

    df = df.copy()

    # =====================================================
    # Age Groups
    # =====================================================

    df["age_group"] = pd.cut(
        df["age"],
        bins=[
            0,
            24,
            34,
            44,
            54,
            100,
        ],
        labels=[
            "18-24",
            "25-34",
            "35-44",
            "45-54",
            "55+",
        ],
    )

    # =====================================================
    # Spending Bands
    # =====================================================

    df["spending_band"] = pd.cut(
        df["purchase_amount_usd"],
        bins=[
            0,
            40,
            70,
            100,
            np.inf,
        ],
        labels=[
            "Low",
            "Medium",
            "High",
            "Very High",
        ],
    )

    # =====================================================
    # Purchase History Bands
    # =====================================================

    df["purchase_history_band"] = pd.cut(
        df["previous_purchases"],
        bins=[
            -1,
            10,
            25,
            40,
            np.inf,
        ],
        labels=[
            "Low History",
            "Moderate History",
            "High History",
            "Very High History",
        ],
    )

    # =====================================================
    # Customer Value Score
    # =====================================================

    df["customer_value_score"] = (
        df["purchase_amount_usd"] * 0.6
        + df["previous_purchases"] * 0.4
    )

    # =====================================================
    # Retention Risk Proxy
    # =====================================================

    # IMPORTANT:
    #
    # The dataset does not contain an observed churn label
    # or longitudinal customer history.
    #
    # Therefore this is NOT a churn prediction model.
    #
    # It is a behavioural retention-risk proxy based on:
    #
    # 1. Low previous purchases
    # 2. Infrequent purchasing
    # 3. Non-subscription status

    low_history = (
        df["previous_purchases"] <= 10
    )

    low_frequency = (
        df["frequency_of_purchases"].isin(
            [
                "Annually",
                "Quarterly",
            ]
        )
    )

    non_subscriber = (
        df["subscriber_binary"] == 0
    )

    df["retention_risk_proxy"] = np.select(
        [
            (
                low_history
                & low_frequency
                & non_subscriber
            ),

            (
                low_history
                & (
                    low_frequency
                    | non_subscriber
                )
            ),
        ],
        [
            "High",
            "Medium",
        ],
        default="Low",
    )

    # =====================================================
    # Retention Risk Score
    # =====================================================

    # Transparent 0-3 behavioural score.

    df["retention_risk_score"] = 0

    df.loc[
        low_history,
        "retention_risk_score",
    ] += 1

    df.loc[
        low_frequency,
        "retention_risk_score",
    ] += 1

    df.loc[
        non_subscriber,
        "retention_risk_score",
    ] += 1

    return df


# =========================================================
# Data Quality Report
# =========================================================

def data_quality_report(df):
    """Generate a data-quality report."""

    quality = pd.DataFrame(
        {
            "column": df.columns,

            "data_type": [
                str(dtype)
                for dtype in df.dtypes
            ],

            "missing_values": [
                df[column].isna().sum()
                for column in df.columns
            ],

            "unique_values": [
                df[column].nunique()
                for column in df.columns
            ],
        }
    )

    quality["missing_percentage"] = (
        quality["missing_values"]
        / len(df)
        * 100
    )

    quality.to_csv(
        OUTPUT_DIR / "data_quality_report.csv",
        index=False,
    )

    return quality


# =========================================================
# Business Summary
# =========================================================

def generate_summary(df):
    """Generate key business metrics."""

    summary = {
        "customers":
            df["customer_id"].nunique(),

        "transactions":
            len(df),

        "total_revenue":
            df["purchase_amount_usd"].sum(),

        "average_purchase":
            df["purchase_amount_usd"].mean(),

        "median_purchase":
            df["purchase_amount_usd"].median(),

        "average_rating":
            df["review_rating"].mean(),

        "subscriber_rate":
            df["subscriber_binary"].mean(),

        "high_risk_customers":
            (
                df["retention_risk_proxy"]
                == "High"
            ).sum(),

        "high_risk_rate":
            (
                (
                    df["retention_risk_proxy"]
                    == "High"
                ).mean()
            ),
    }

    summary_df = pd.DataFrame(
        summary.items(),
        columns=[
            "metric",
            "value",
        ],
    )

    summary_df.to_csv(
        OUTPUT_DIR / "business_summary.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("BUSINESS SUMMARY")
    print("=" * 60)

    print(
        summary_df.to_string(
            index=False
        )
    )

    return summary_df


# =========================================================
# Executive KPI Summary
# =========================================================

def executive_kpis(df):
    """Create executive-level KPI metrics."""

    high_value_threshold = (
        df["purchase_amount_usd"]
        .quantile(0.75)
    )

    high_value = (
        df["purchase_amount_usd"]
        >= high_value_threshold
    )

    high_risk = (
        df["retention_risk_proxy"]
        == "High"
    )

    kpis = {
        "Total Customers":
            df["customer_id"].nunique(),

        "Total Revenue":
            df["purchase_amount_usd"].sum(),

        "Average Purchase":
            df["purchase_amount_usd"].mean(),

        "Median Purchase":
            df["purchase_amount_usd"].median(),

        "Average Rating":
            df["review_rating"].mean(),

        "Subscriber Rate":
            df["subscriber_binary"].mean() * 100,

        "High Risk Customers":
            high_risk.sum(),

        "High Risk Customer Rate":
            high_risk.mean() * 100,

        "High Value Customers":
            high_value.sum(),

        "High Value Revenue":
            df.loc[
                high_value,
                "purchase_amount_usd",
            ].sum(),

        "High Value Revenue Share":
            (
                df.loc[
                    high_value,
                    "purchase_amount_usd",
                ].sum()
                / df["purchase_amount_usd"].sum()
                * 100
            ),
    }

    result = pd.DataFrame(
        {
            "kpi": list(kpis.keys()),
            "value": list(kpis.values()),
        }
    )

    result.to_csv(
        OUTPUT_DIR / "executive_kpis.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("EXECUTIVE KPIs")
    print("=" * 60)

    print(
        result.to_string(
            index=False
        )
    )

    return result


# =========================================================
# Category Analysis
# =========================================================

def category_analysis(df):
    """Analyse revenue and customer behaviour by category."""

    result = (
        df
        .groupby(
            "category",
            observed=True,
        )
        .agg(
            customers=(
                "customer_id",
                "nunique",
            ),

            transactions=(
                "customer_id",
                "count",
            ),

            revenue=(
                "purchase_amount_usd",
                "sum",
            ),

            average_purchase=(
                "purchase_amount_usd",
                "mean",
            ),

            median_purchase=(
                "purchase_amount_usd",
                "median",
            ),

            average_rating=(
                "review_rating",
                "mean",
            ),
        )
        .sort_values(
            "revenue",
            ascending=False,
        )
        .reset_index()
    )

    result["revenue_share"] = (
        result["revenue"]
        / result["revenue"].sum()
        * 100
    )

    result.to_csv(
        OUTPUT_DIR / "category_analysis.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("CATEGORY ANALYSIS")
    print("=" * 60)

    print(
        result.to_string(
            index=False
        )
    )

    return result


# =========================================================
# Season Analysis
# =========================================================

def season_analysis(df):
    """Analyse purchasing behaviour by season."""

    result = (
        df
        .groupby(
            "season",
            observed=True,
        )
        .agg(
            customers=(
                "customer_id",
                "nunique",
            ),

            revenue=(
                "purchase_amount_usd",
                "sum",
            ),

            average_purchase=(
                "purchase_amount_usd",
                "mean",
            ),

            median_purchase=(
                "purchase_amount_usd",
                "median",
            ),

            average_rating=(
                "review_rating",
                "mean",
            ),
        )
        .sort_values(
            "revenue",
            ascending=False,
        )
        .reset_index()
    )

    result["revenue_share"] = (
        result["revenue"]
        / result["revenue"].sum()
        * 100
    )

    result.to_csv(
        OUTPUT_DIR / "season_analysis.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("SEASON ANALYSIS")
    print("=" * 60)

    print(
        result.to_string(
            index=False
        )
    )

    return result


# =========================================================
# Subscription Analysis
# =========================================================

def subscription_analysis(df):
    """Analyse subscriber and non-subscriber behaviour."""

    result = (
        df
        .groupby(
            "subscription_status",
            observed=True,
        )
        .agg(
            customers=(
                "customer_id",
                "nunique",
            ),

            revenue=(
                "purchase_amount_usd",
                "sum",
            ),

            average_purchase=(
                "purchase_amount_usd",
                "mean",
            ),

            median_purchase=(
                "purchase_amount_usd",
                "median",
            ),

            average_previous_purchases=(
                "previous_purchases",
                "mean",
            ),

            average_rating=(
                "review_rating",
                "mean",
            ),
        )
        .reset_index()
    )

    result.to_csv(
        OUTPUT_DIR / "subscription_analysis.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("SUBSCRIPTION ANALYSIS")
    print("=" * 60)

    print(
        result.to_string(
            index=False
        )
    )

    return result


# =========================================================
# Purchase Frequency Analysis
# =========================================================

def frequency_analysis(df):
    """Analyse customer purchase frequency."""

    result = (
        df
        .groupby(
            "frequency_of_purchases",
            observed=True,
        )
        .agg(
            customers=(
                "customer_id",
                "nunique",
            ),

            revenue=(
                "purchase_amount_usd",
                "sum",
            ),

            average_purchase=(
                "purchase_amount_usd",
                "mean",
            ),

            average_previous_purchases=(
                "previous_purchases",
                "mean",
            ),
        )
        .sort_values(
            "customers",
            ascending=False,
        )
        .reset_index()
    )

    result.to_csv(
        OUTPUT_DIR / "frequency_analysis.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("PURCHASE FREQUENCY ANALYSIS")
    print("=" * 60)

    print(
        result.to_string(
            index=False
        )
    )

    return result


# =========================================================
# Customer Segmentation
# =========================================================

def customer_segmentation(df):
    """
    Create data-driven customer value segments.

    Because this dataset contains one record per customer,
    total_spend is effectively the customer's observed
    purchase amount in this dataset.
    """

    customer = (
        df
        .groupby("customer_id")
        .agg(
            total_spend=(
                "purchase_amount_usd",
                "sum",
            ),

            purchase_count=(
                "customer_id",
                "count",
            ),

            average_purchase=(
                "purchase_amount_usd",
                "mean",
            ),

            previous_purchases=(
                "previous_purchases",
                "max",
            ),

            average_rating=(
                "review_rating",
                "mean",
            ),

            subscriber=(
                "subscriber_binary",
                "max",
            ),

            retention_risk_score=(
                "retention_risk_score",
                "max",
            ),
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # Data-driven quartile segmentation
    # -----------------------------------------------------

    customer["customer_segment"] = pd.qcut(
        customer["total_spend"],
        q=4,
        labels=[
            "Low Value",
            "Developing",
            "High Value",
            "Premium",
        ],
        duplicates="drop",
    )

    # -----------------------------------------------------
    # Segment summary
    # -----------------------------------------------------

    segment_summary = (
        customer
        .groupby(
            "customer_segment",
            observed=True,
        )
        .agg(
            customers=(
                "customer_id",
                "nunique",
            ),

            total_spend=(
                "total_spend",
                "sum",
            ),

            average_spend=(
                "total_spend",
                "mean",
            ),

            average_previous_purchases=(
                "previous_purchases",
                "mean",
            ),

            subscriber_rate=(
                "subscriber",
                "mean",
            ),

            average_risk_score=(
                "retention_risk_score",
                "mean",
            ),
        )
        .reset_index()
    )

    segment_summary["revenue_share"] = (
        segment_summary["total_spend"]
        / segment_summary["total_spend"].sum()
        * 100
    )

    # -----------------------------------------------------
    # Save customer-level segmentation
    # -----------------------------------------------------

    customer.to_csv(
        OUTPUT_DIR / "customer_segments.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Save segment-level summary
    # -----------------------------------------------------

    segment_summary.to_csv(
        OUTPUT_DIR
        / "customer_segment_summary.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("CUSTOMER SEGMENTATION")
    print("=" * 60)

    print(
        segment_summary.to_string(
            index=False
        )
    )

    return customer


# =========================================================
# Statistical Research
# =========================================================

def statistical_research(df):
    """
    Run statistical tests to investigate relationships
    within customer purchasing behaviour.

    Tests:
        1. Welch independent t-test
        2. One-way ANOVA by category
        3. One-way ANOVA by season
        4. Spearman correlation
    """

    results = []

    # =====================================================
    # 1. Subscription vs Purchase Amount
    # =====================================================

    subscribers = df.loc[
        df["subscriber_binary"] == 1,
        "purchase_amount_usd",
    ].dropna()

    non_subscribers = df.loc[
        df["subscriber_binary"] == 0,
        "purchase_amount_usd",
    ].dropna()

    t_stat, t_p_value = stats.ttest_ind(
        subscribers,
        non_subscribers,
        equal_var=False,
    )

    results.append(
        {
            "research_question":
                "Does subscription status affect purchase amount?",

            "test":
                "Welch independent t-test",

            "statistic":
                t_stat,

            "p_value":
                t_p_value,

            "significant_at_0.05":
                t_p_value < 0.05,
        }
    )

    # =====================================================
    # 2. Category vs Purchase Amount
    # =====================================================

    category_groups = [
        group[
            "purchase_amount_usd"
        ].dropna().values

        for _, group in df.groupby(
            "category",
            observed=True,
        )

        if len(group) > 1
    ]

    if len(category_groups) >= 2:

        category_f, category_p = (
            stats.f_oneway(
                *category_groups
            )
        )

        results.append(
            {
                "research_question":
                    "Does purchase amount differ across categories?",

                "test":
                    "One-way ANOVA",

                "statistic":
                    category_f,

                "p_value":
                    category_p,

                "significant_at_0.05":
                    category_p < 0.05,
            }
        )

    # =====================================================
    # 3. Season vs Purchase Amount
    # =====================================================

    season_groups = [
        group[
            "purchase_amount_usd"
        ].dropna().values

        for _, group in df.groupby(
            "season",
            observed=True,
        )

        if len(group) > 1
    ]

    if len(season_groups) >= 2:

        season_f, season_p = (
            stats.f_oneway(
                *season_groups
            )
        )

        results.append(
            {
                "research_question":
                    "Does purchase amount differ across seasons?",

                "test":
                    "One-way ANOVA",

                "statistic":
                    season_f,

                "p_value":
                    season_p,

                "significant_at_0.05":
                    season_p < 0.05,
            }
        )

    # =====================================================
    # 4. Previous Purchases vs Current Purchase
    # =====================================================

    correlation_data = df[
        [
            "previous_purchases",
            "purchase_amount_usd",
        ]
    ].dropna()

    correlation, correlation_p = (
        stats.spearmanr(
            correlation_data[
                "previous_purchases"
            ],

            correlation_data[
                "purchase_amount_usd"
            ],
        )
    )

    results.append(
        {
            "research_question":
                "Is purchase history associated with current purchase amount?",

            "test":
                "Spearman correlation",

            "statistic":
                correlation,

            "p_value":
                correlation_p,

            "significant_at_0.05":
                correlation_p < 0.05,
        }
    )

    # =====================================================
    # Save Results
    # =====================================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR
        / "statistical_research_results.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("STATISTICAL RESEARCH RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    return results_df


# =========================================================
# Retention Risk Analysis
# =========================================================

def retention_risk_analysis(df):
    """
    Analyse the engineered retention-risk proxy.

    This is NOT observed customer churn.
    """

    result = (
        df
        .groupby(
            "retention_risk_proxy",
            observed=True,
        )
        .agg(
            customers=(
                "customer_id",
                "nunique",
            ),

            revenue=(
                "purchase_amount_usd",
                "sum",
            ),

            average_purchase=(
                "purchase_amount_usd",
                "mean",
            ),

            average_previous_purchases=(
                "previous_purchases",
                "mean",
            ),

            average_risk_score=(
                "retention_risk_score",
                "mean",
            ),
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # Revenue share
    # -----------------------------------------------------

    result["revenue_share"] = (
        result["revenue"]
        / result["revenue"].sum()
        * 100
    )

    # -----------------------------------------------------
    # Logical risk ordering
    # -----------------------------------------------------

    risk_order = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
    }

    result["risk_order"] = (
        result[
            "retention_risk_proxy"
        ].map(risk_order)
    )

    result = (
        result
        .sort_values("risk_order")
        .drop(
            columns="risk_order"
        )
    )

    result.to_csv(
        OUTPUT_DIR
        / "retention_risk_analysis.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("RETENTION RISK ANALYSIS")
    print("=" * 60)

    print(
        result.to_string(
            index=False
        )
    )

    return result


# =========================================================
# Promotion & Discount Analysis
# =========================================================

def behaviour_analysis(df):
    """Analyse discount and promotion behaviour."""

    result = (
        df
        .groupby(
            [
                "discount_applied",
                "promo_code_used",
            ],
            observed=True,
        )
        .agg(
            customers=(
                "customer_id",
                "nunique",
            ),

            revenue=(
                "purchase_amount_usd",
                "sum",
            ),

            average_purchase=(
                "purchase_amount_usd",
                "mean",
            ),

            average_rating=(
                "review_rating",
                "mean",
            ),
        )
        .reset_index()
    )

    result.to_csv(
        OUTPUT_DIR
        / "promotion_discount_analysis.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("PROMOTION & DISCOUNT ANALYSIS")
    print("=" * 60)

    print(
        result.to_string(
            index=False
        )
    )

    return result


# =========================================================
# Location Analysis
# =========================================================

def location_analysis(df):
    """Analyse revenue and purchasing behaviour by location."""

    result = (
        df
        .groupby(
            "location",
            observed=True,
        )
        .agg(
            customers=(
                "customer_id",
                "nunique",
            ),

            revenue=(
                "purchase_amount_usd",
                "sum",
            ),

            average_purchase=(
                "purchase_amount_usd",
                "mean",
            ),

            average_rating=(
                "review_rating",
                "mean",
            ),

            average_previous_purchases=(
                "previous_purchases",
                "mean",
            ),
        )
        .sort_values(
            "revenue",
            ascending=False,
        )
        .reset_index()
    )

    result.to_csv(
        OUTPUT_DIR
        / "location_analysis.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("LOCATION ANALYSIS")
    print("=" * 60)

    print(
        result.to_string(
            index=False
        )
    )

    return result


# =========================================================
# High Value Customer Analysis
# =========================================================

def high_value_analysis(df):
    """Identify high-value purchase records."""

    threshold = (
        df["purchase_amount_usd"]
        .quantile(0.75)
    )

    high_value = (
        df[
            df["purchase_amount_usd"]
            >= threshold
        ]
        .copy()
    )

    result = pd.DataFrame(
        {
            "metric": [
                "75th percentile purchase threshold",
                "high-value customers",
                "high-value revenue",
                "high-value average purchase",
                "high-value subscriber rate",
            ],

            "value": [
                threshold,

                high_value[
                    "customer_id"
                ].nunique(),

                high_value[
                    "purchase_amount_usd"
                ].sum(),

                high_value[
                    "purchase_amount_usd"
                ].mean(),

                high_value[
                    "subscriber_binary"
                ].mean(),
            ],
        }
    )

    result.to_csv(
        OUTPUT_DIR
        / "high_value_customer_analysis.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("HIGH VALUE CUSTOMER ANALYSIS")
    print("=" * 60)

    print(
        result.to_string(
            index=False
        )
    )

    return result


# =========================================================
# Main Analysis Pipeline
# =========================================================

def main():

    print("=" * 60)

    print(
        "CUSTOMER BEHAVIOUR & RETENTION ANALYTICS"
    )

    print("=" * 60)

    # -----------------------------------------------------
    # 1. Load
    # -----------------------------------------------------

    df = load_data()

    # -----------------------------------------------------
    # 2. Clean
    # -----------------------------------------------------

    df = clean_data(df)

    # -----------------------------------------------------
    # 3. Feature Engineering
    # -----------------------------------------------------

    df = create_features(df)

    # -----------------------------------------------------
    # 4. Data Quality
    # -----------------------------------------------------

    data_quality_report(df)

    # -----------------------------------------------------
    # 5. Business Summary
    # -----------------------------------------------------

    generate_summary(df)

    # -----------------------------------------------------
    # 6. Executive KPIs
    # -----------------------------------------------------

    executive_kpis(df)

    # -----------------------------------------------------
    # 7. Category Analysis
    # -----------------------------------------------------

    category_analysis(df)

    # -----------------------------------------------------
    # 8. Season Analysis
    # -----------------------------------------------------

    season_analysis(df)

    # -----------------------------------------------------
    # 9. Subscription Analysis
    # -----------------------------------------------------

    subscription_analysis(df)

    # -----------------------------------------------------
    # 10. Purchase Frequency
    # -----------------------------------------------------

    frequency_analysis(df)

    # -----------------------------------------------------
    # 11. Customer Segmentation
    # -----------------------------------------------------

    customer_segmentation(df)

    # -----------------------------------------------------
    # 12. Statistical Research
    # -----------------------------------------------------

    statistical_research(df)

    # -----------------------------------------------------
    # 13. Retention Risk
    # -----------------------------------------------------

    retention_risk_analysis(df)

    # -----------------------------------------------------
    # 14. Promotion / Discount Analysis
    # -----------------------------------------------------

    behaviour_analysis(df)

    # -----------------------------------------------------
    # 15. Location Analysis
    # -----------------------------------------------------

    location_analysis(df)

    # -----------------------------------------------------
    # 16. High Value Analysis
    # -----------------------------------------------------

    high_value_analysis(df)

    # -----------------------------------------------------
    # 17. Save Clean Analytical Dataset
    # -----------------------------------------------------

    df.to_csv(
        OUTPUT_DIR
        / "customer_behavior_clean.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Completion
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Outputs saved to: {OUTPUT_DIR}"
    )


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    main()