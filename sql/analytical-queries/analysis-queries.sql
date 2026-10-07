/*
============================================================
CUSTOMER BEHAVIOUR, RETENTION & REVENUE ANALYTICS
SQL ANALYTICAL QUERIES
============================================================

Schema:
    DIM_Products
    DIM_Customers
    FACT_Sales
    Customer_Reviews

Purpose:
    Business-focused analysis using a relational star schema.

Analysis areas:
    - Revenue KPIs
    - Product/category performance
    - Customer behaviour
    - Subscription analysis
    - Geographic performance
    - Customer value segmentation
    - High-value customers
    - Ranking and window functions
    - Retention-risk proxy

Important limitation:
    The dataset does not contain longitudinal customer transactions
    suitable for measuring observed churn. Retention analysis is
    therefore treated as a behavioural proxy.
*/


/* ==========================================================
1. OVERALL BUSINESS KPIs
========================================================== */

SELECT
    COUNT(DISTINCT CustomerID) AS TotalCustomers,
    COUNT(*) AS TotalTransactions,
    SUM(PurchaseAmount) AS TotalRevenue,
    ROUND(AVG(PurchaseAmount), 2) AS AveragePurchase,
    ROUND(AVG(ReviewRating), 2) AS AverageRating
FROM FACT_Sales;


/* ==========================================================
2. REVENUE BY PRODUCT CATEGORY
========================================================== */

SELECT
    p.Category,
    COUNT(*) AS Transactions,
    SUM(f.PurchaseAmount) AS Revenue,
    ROUND(AVG(f.PurchaseAmount), 2) AS AveragePurchase,
    ROUND(
        SUM(f.PurchaseAmount) * 100.0 /
        SUM(SUM(f.PurchaseAmount)) OVER (),
        2
    ) AS RevenueSharePct
FROM FACT_Sales f
JOIN DIM_Products p
    ON f.ProductID = p.ProductID
GROUP BY p.Category
ORDER BY Revenue DESC;


/* ==========================================================
3. PRODUCT PERFORMANCE
========================================================== */

SELECT
    p.ProductID,
    p.ItemName,
    p.Category,
    COUNT(*) AS Transactions,
    SUM(f.PurchaseAmount) AS Revenue,
    ROUND(AVG(f.PurchaseAmount), 2) AS AveragePurchase,
    ROUND(AVG(f.ReviewRating), 2) AS AverageRating
FROM FACT_Sales f
JOIN DIM_Products p
    ON f.ProductID = p.ProductID
GROUP BY
    p.ProductID,
    p.ItemName,
    p.Category
ORDER BY Revenue DESC;


/* ==========================================================
4. CUSTOMER SUBSCRIPTION ANALYSIS
========================================================== */

SELECT
    c.IsSubscriber,
    COUNT(DISTINCT c.CustomerID) AS Customers,
    SUM(f.PurchaseAmount) AS Revenue,
    ROUND(AVG(f.PurchaseAmount), 2) AS AveragePurchase,
    ROUND(AVG(f.ReviewRating), 2) AS AverageRating
FROM DIM_Customers c
JOIN FACT_Sales f
    ON c.CustomerID = f.CustomerID
GROUP BY c.IsSubscriber
ORDER BY Revenue DESC;


/* ==========================================================
5. GEOGRAPHIC PERFORMANCE
========================================================== */

SELECT
    c.Location,
    COUNT(DISTINCT c.CustomerID) AS Customers,
    SUM(f.PurchaseAmount) AS Revenue,
    ROUND(AVG(f.PurchaseAmount), 2) AS AveragePurchase,
    ROUND(AVG(f.ReviewRating), 2) AS AverageRating
FROM DIM_Customers c
JOIN FACT_Sales f
    ON c.CustomerID = f.CustomerID
GROUP BY c.Location
ORDER BY Revenue DESC;


/* ==========================================================
6. CUSTOMER VALUE SEGMENTATION
========================================================== */

WITH CustomerSpend AS (

    SELECT
        c.CustomerID,
        c.Age,
        c.Location,
        c.IsSubscriber,
        SUM(f.PurchaseAmount) AS TotalSpend

    FROM DIM_Customers c

    JOIN FACT_Sales f
        ON c.CustomerID = f.CustomerID

    GROUP BY
        c.CustomerID,
        c.Age,
        c.Location,
        c.IsSubscriber
),

CustomerSegments AS (

    SELECT
        *,
        CASE
            WHEN TotalSpend < 40
                THEN 'Low Value'

            WHEN TotalSpend < 60
                THEN 'Developing'

            WHEN TotalSpend < 80
                THEN 'High Value'

            ELSE 'Premium'
        END AS CustomerSegment

    FROM CustomerSpend
)

SELECT
    CustomerSegment,
    COUNT(*) AS Customers,
    ROUND(AVG(TotalSpend), 2) AS AverageSpend,
    SUM(TotalSpend) AS Revenue
FROM CustomerSegments
GROUP BY CustomerSegment
ORDER BY AverageSpend;


/* ==========================================================
7. CUSTOMER VALUE QUARTILES
========================================================== */

WITH CustomerSpend AS (

    SELECT
        c.CustomerID,
        SUM(f.PurchaseAmount) AS TotalSpend

    FROM DIM_Customers c

    JOIN FACT_Sales f
        ON c.CustomerID = f.CustomerID

    GROUP BY c.CustomerID
)

SELECT
    CustomerID,
    TotalSpend,

    NTILE(4) OVER (
        ORDER BY TotalSpend
    ) AS ValueQuartile

FROM CustomerSpend

ORDER BY TotalSpend DESC;


/* ==========================================================
8. TOP 25 CUSTOMERS BY REVENUE
========================================================== */

WITH CustomerSpend AS (

    SELECT
        c.CustomerID,
        c.Age,
        c.Location,
        c.IsSubscriber,
        SUM(f.PurchaseAmount) AS TotalSpend

    FROM DIM_Customers c

    JOIN FACT_Sales f
        ON c.CustomerID = f.CustomerID

    GROUP BY
        c.CustomerID,
        c.Age,
        c.Location,
        c.IsSubscriber
),

RankedCustomers AS (

    SELECT
        *,
        RANK() OVER (
            ORDER BY TotalSpend DESC
        ) AS RevenueRank

    FROM CustomerSpend
)

SELECT *
FROM RankedCustomers
WHERE RevenueRank <= 25
ORDER BY RevenueRank;


/* ==========================================================
9. CATEGORY REVENUE RANKING
========================================================== */

WITH CategoryRevenue AS (

    SELECT
        p.Category,
        SUM(f.PurchaseAmount) AS Revenue

    FROM FACT_Sales f

    JOIN DIM_Products p
        ON f.ProductID = p.ProductID

    GROUP BY p.Category
)

SELECT
    Category,
    Revenue,

    RANK() OVER (
        ORDER BY Revenue DESC
    ) AS RevenueRank

FROM CategoryRevenue

ORDER BY RevenueRank;


/* ==========================================================
10. CATEGORY CONTRIBUTION TO TOTAL REVENUE
========================================================== */

WITH CategoryRevenue AS (

    SELECT
        p.Category,
        SUM(f.PurchaseAmount) AS Revenue

    FROM FACT_Sales f

    JOIN DIM_Products p
        ON f.ProductID = p.ProductID

    GROUP BY p.Category
)

SELECT
    Category,
    Revenue,

    ROUND(
        Revenue * 100.0 /
        SUM(Revenue) OVER (),
        2
    ) AS RevenueSharePct

FROM CategoryRevenue

ORDER BY Revenue DESC;


/* ==========================================================
11. HIGH-VALUE CUSTOMERS
========================================================== */

WITH CustomerSpend AS (

    SELECT
        CustomerID,
        SUM(PurchaseAmount) AS TotalSpend

    FROM FACT_Sales

    GROUP BY CustomerID
)

SELECT
    CustomerID,
    TotalSpend

FROM CustomerSpend

WHERE TotalSpend >= (
    SELECT
        PERCENTILE_CONT(0.75)
        WITHIN GROUP (
            ORDER BY TotalSpend
        )
    FROM CustomerSpend
)

ORDER BY TotalSpend DESC;


/* ==========================================================
12. HIGH-VALUE CUSTOMERS BY SUBSCRIPTION STATUS
========================================================== */

WITH CustomerSpend AS (

    SELECT
        c.CustomerID,
        c.IsSubscriber,
        SUM(f.PurchaseAmount) AS TotalSpend

    FROM DIM_Customers c

    JOIN FACT_Sales f
        ON c.CustomerID = f.CustomerID

    GROUP BY
        c.CustomerID,
        c.IsSubscriber
)

SELECT
    IsSubscriber,
    COUNT(*) AS HighValueCustomers,
    SUM(TotalSpend) AS HighValueRevenue,
    ROUND(AVG(TotalSpend), 2) AS AverageHighValueSpend

FROM CustomerSpend

WHERE TotalSpend >= 81

GROUP BY IsSubscriber

ORDER BY HighValueRevenue DESC;


/* ==========================================================
13. RETENTION RISK PROXY
==========================================================

This query uses transaction history as a behavioural indicator.

Because the dataset does not provide true longitudinal churn
outcomes, this should NOT be interpreted as actual churn.
*/

WITH CustomerHistory AS (

    SELECT
        CustomerID,
        COUNT(*) AS PurchaseCount,
        SUM(PurchaseAmount) AS TotalSpend

    FROM FACT_Sales

    GROUP BY CustomerID
)

SELECT
    CASE
        WHEN PurchaseCount >= 10
            THEN 'Low Risk'

        WHEN PurchaseCount >= 5
            THEN 'Medium Risk'

        ELSE 'High Risk'
    END AS RetentionRisk,

    COUNT(*) AS Customers,
    SUM(TotalSpend) AS Revenue,
    ROUND(AVG(TotalSpend), 2) AS AverageCustomerSpend

FROM CustomerHistory

GROUP BY

    CASE
        WHEN PurchaseCount >= 10
            THEN 'Low Risk'

        WHEN PurchaseCount >= 5
            THEN 'Medium Risk'

        ELSE 'High Risk'
    END

ORDER BY
    AverageCustomerSpend DESC;


/* ==========================================================
14. HIGH-VALUE CUSTOMERS WITH RETENTION RISK
========================================================== */

WITH CustomerHistory AS (

    SELECT
        CustomerID,
        COUNT(*) AS PurchaseCount,
        SUM(PurchaseAmount) AS TotalSpend

    FROM FACT_Sales

    GROUP BY CustomerID
)

SELECT
    CustomerID,
    PurchaseCount,
    TotalSpend,

    CASE
        WHEN PurchaseCount < 5
             AND TotalSpend >= 80
            THEN 'High Value - Retention Priority'

        WHEN PurchaseCount < 5
            THEN 'High Retention Risk'

        WHEN TotalSpend >= 80
            THEN 'High Value Customer'

        ELSE 'Standard Customer'
    END AS CustomerPriority

FROM CustomerHistory

ORDER BY
    CASE
        WHEN PurchaseCount < 5
             AND TotalSpend >= 80
            THEN 1

        WHEN PurchaseCount < 5
            THEN 2

        WHEN TotalSpend >= 80
            THEN 3

        ELSE 4
    END,

    TotalSpend DESC;


/* ==========================================================
15. PAYMENT METHOD ANALYSIS
========================================================== */

SELECT
    PaymentMethod,
    COUNT(*) AS Transactions,
    SUM(PurchaseAmount) AS Revenue,
    ROUND(AVG(PurchaseAmount), 2) AS AveragePurchase
FROM FACT_Sales
GROUP BY PaymentMethod
ORDER BY Revenue DESC;


/* ==========================================================
16. CUSTOMER AGE GROUP ANALYSIS
========================================================== */

SELECT
    CASE
        WHEN c.Age < 25
            THEN '18-24'

        WHEN c.Age < 35
            THEN '25-34'

        WHEN c.Age < 45
            THEN '35-44'

        WHEN c.Age < 55
            THEN '45-54'

        ELSE '55+'
    END AS AgeGroup,

    COUNT(DISTINCT c.CustomerID) AS Customers,
    SUM(f.PurchaseAmount) AS Revenue,
    ROUND(AVG(f.PurchaseAmount), 2) AS AveragePurchase

FROM DIM_Customers c

JOIN FACT_Sales f
    ON c.CustomerID = f.CustomerID

GROUP BY

    CASE
        WHEN c.Age < 25
            THEN '18-24'

        WHEN c.Age < 35
            THEN '25-34'

        WHEN c.Age < 45
            THEN '35-44'

        WHEN c.Age < 55
            THEN '45-54'

        ELSE '55+'
    END

ORDER BY Revenue DESC;


/* ==========================================================
17. CUSTOMER REVIEWS
========================================================== */

SELECT
    r.CustomerID,
    r.ReviewDate,
    r.ReviewText
FROM Customer_Reviews r
ORDER BY r.ReviewDate DESC;


/* ==========================================================
18. CUSTOMER + PRODUCT + TRANSACTION DETAIL
========================================================== */

SELECT
    f.TransactionID,
    c.CustomerID,
    c.Age,
    c.Gender,
    c.Location,
    c.IsSubscriber,
    p.ItemName,
    p.Category,
    f.PurchaseAmount,
    f.PurchaseDate,
    f.ReviewRating,
    f.PaymentMethod

FROM FACT_Sales f

JOIN DIM_Customers c
    ON f.CustomerID = c.CustomerID

JOIN DIM_Products p
    ON f.ProductID = p.ProductID

ORDER BY f.PurchaseDate DESC;


/* ==========================================================
19. EXECUTIVE REVENUE SUMMARY
========================================================== */

WITH CategoryRevenue AS (

    SELECT
        p.Category,
        SUM(f.PurchaseAmount) AS Revenue

    FROM FACT_Sales f

    JOIN DIM_Products p
        ON f.ProductID = p.ProductID

    GROUP BY p.Category
),

OverallRevenue AS (

    SELECT
        SUM(PurchaseAmount) AS TotalRevenue
    FROM FACT_Sales
)

SELECT
    cr.Category,
    cr.Revenue,
    ROUND(
        cr.Revenue * 100.0 /
        o.TotalRevenue,
        2
    ) AS RevenueSharePct

FROM CategoryRevenue cr

CROSS JOIN OverallRevenue o

ORDER BY cr.Revenue DESC;


/* ==========================================================
20. BUSINESS PRIORITY CUSTOMERS
========================================================== */

WITH CustomerMetrics AS (

    SELECT
        c.CustomerID,
        c.Age,
        c.Location,
        c.IsSubscriber,
        COUNT(f.TransactionID) AS PurchaseCount,
        SUM(f.PurchaseAmount) AS TotalSpend,
        AVG(f.PurchaseAmount) AS AveragePurchase

    FROM DIM_Customers c

    JOIN FACT_Sales f
        ON c.CustomerID = f.CustomerID

    GROUP BY
        c.CustomerID,
        c.Age,
        c.Location,
        c.IsSubscriber
)

SELECT
    CustomerID,
    Age,
    Location,
    IsSubscriber,
    PurchaseCount,
    ROUND(TotalSpend, 2) AS TotalSpend,
    ROUND(AveragePurchase, 2) AS AveragePurchase,

    CASE
        WHEN PurchaseCount < 5
             AND TotalSpend >= 80
            THEN 'Priority Retention'

        WHEN TotalSpend >= 80
            THEN 'High Value'

        WHEN PurchaseCount < 5
            THEN 'At Risk'

        ELSE 'Standard'
    END AS BusinessPriority

FROM CustomerMetrics

ORDER BY
    CASE
        WHEN PurchaseCount < 5
             AND TotalSpend >= 80
            THEN 1

        WHEN TotalSpend >= 80
            THEN 2

        WHEN PurchaseCount < 5
            THEN 3

        ELSE 4
    END,

    TotalSpend DESC;