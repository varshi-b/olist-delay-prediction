-- [Q1_OVERALL_DELAY_SUMMARY]
-- Overall delivery volume and delay rate across all delivered orders
SELECT
    COUNT(*) AS total_delivered_orders,
    SUM(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) AS delayed_orders,
    ROUND(
        100.0 * SUM(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS delay_rate_pct,
    ROUND(
        AVG(EXTRACT(EPOCH FROM (order_delivered_customer_date - order_purchase_timestamp)) / 86400.0)::numeric,
        2
    ) AS avg_actual_delivery_days,
    ROUND(
        AVG(EXTRACT(EPOCH FROM (order_estimated_delivery_date - order_purchase_timestamp)) / 86400.0)::numeric,
        2
    ) AS avg_estimated_delivery_days
FROM orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL
  AND order_estimated_delivery_date IS NOT NULL;

-- [Q2_DELAY_RATE_BY_STATE]
-- Top 10 customer states with highest delivery delay rates (minimum 500 orders)
SELECT
    c.customer_state,
    COUNT(o.order_id) AS total_orders,
    SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) AS late_orders,
    ROUND(
        100.0 * SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) / COUNT(o.order_id),
        2
    ) AS late_rate_pct,
    ROUND(
        AVG(EXTRACT(EPOCH FROM (o.order_delivered_customer_date - o.order_estimated_delivery_date)) / 86400.0)::numeric,
        2
    ) AS avg_days_diff_from_estimate
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
GROUP BY c.customer_state
HAVING COUNT(o.order_id) >= 500
ORDER BY late_rate_pct DESC
LIMIT 10;

-- [Q3_MONTHLY_DELAY_TREND_WITH_LAG]
-- Monthly order volume, delay rate, and month-over-month change using LAG() window function
WITH monthly_stats AS (
    SELECT
        DATE_TRUNC('month', order_purchase_timestamp)::date AS order_month,
        COUNT(order_id) AS total_orders,
        ROUND(
            100.0 * SUM(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) / COUNT(order_id),
            2
        ) AS delay_rate_pct
    FROM orders
    WHERE order_status = 'delivered'
      AND order_delivered_customer_date IS NOT NULL
      AND order_purchase_timestamp >= '2017-01-01'
      AND order_purchase_timestamp < '2018-09-01'
    GROUP BY 1
)
SELECT
    order_month,
    total_orders,
    delay_rate_pct,
    LAG(delay_rate_pct, 1) OVER (ORDER BY order_month) AS prev_month_delay_rate_pct,
    ROUND(
        delay_rate_pct - LAG(delay_rate_pct, 1) OVER (ORDER BY order_month),
        2
    ) AS mom_delay_rate_change
FROM monthly_stats
ORDER BY order_month;

-- [Q4_SELLER_PERFORMANCE_RANKING]
-- Rank sellers with >= 100 orders by highest delay rate using RANK() window function
WITH seller_metrics AS (
    SELECT
        oi.seller_id,
        s.seller_state,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) AS delayed_orders,
        ROUND(
            100.0 * SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END)
            / COUNT(DISTINCT o.order_id),
            2
        ) AS seller_delay_rate_pct
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN sellers s ON oi.seller_id = s.seller_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
    GROUP BY oi.seller_id, s.seller_state
    HAVING COUNT(DISTINCT o.order_id) >= 100
)
SELECT
    seller_id,
    seller_state,
    total_orders,
    delayed_orders,
    seller_delay_rate_pct,
    RANK() OVER (ORDER BY seller_delay_rate_pct DESC) AS worst_delay_rank
FROM seller_metrics
ORDER BY worst_delay_rank
LIMIT 10;

-- [Q5_CATEGORY_DELAY_VS_WEIGHT]
-- Top 10 English product categories by delay rate alongside average product weight
SELECT
    COALESCE(ct.product_category_name_english, p.product_category_name, 'unknown') AS category_english,
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND(AVG(p.product_weight_g)::numeric, 0) AS avg_weight_grams,
    ROUND(
        100.0 * SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END)
        / COUNT(DISTINCT o.order_id),
        2
    ) AS delay_rate_pct
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
JOIN products p ON oi.product_id = p.product_id
LEFT JOIN category_translation ct ON p.product_category_name = ct.product_category_name
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
GROUP BY 1
HAVING COUNT(DISTINCT o.order_id) >= 300
ORDER BY delay_rate_pct DESC
LIMIT 10;

-- [Q6_INTERSTATE_VS_INTRASTATE_SHIPPING]
-- Compare delay rates and transit times when seller and customer are in the same vs different states
WITH order_route AS (
    SELECT DISTINCT
        o.order_id,
        o.order_purchase_timestamp,
        o.order_delivered_customer_date,
        o.order_estimated_delivery_date,
        CASE WHEN s.seller_state = c.customer_state THEN 'Intrastate (Same State)' ELSE 'Interstate (Cross State)' END AS route_type
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN sellers s ON oi.seller_id = s.seller_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
)
SELECT
    route_type,
    COUNT(order_id) AS total_orders,
    ROUND(
        100.0 * SUM(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) / COUNT(order_id),
        2
    ) AS delay_rate_pct,
    ROUND(
        AVG(EXTRACT(EPOCH FROM (order_delivered_customer_date - order_purchase_timestamp)) / 86400.0)::numeric,
        2
    ) AS avg_transit_days
FROM order_route
GROUP BY route_type;

-- [Q7_DELAY_SEVERITY_VS_REVIEW_SCORE]
-- How delivery timing buckets impact customer review scores and 1-star review percentages
WITH delivery_buckets AS (
    SELECT
        o.order_id,
        r.review_score,
        CASE
            WHEN o.order_delivered_customer_date <= o.order_estimated_delivery_date THEN '1. On Time or Early'
            WHEN EXTRACT(EPOCH FROM (o.order_delivered_customer_date - o.order_estimated_delivery_date)) / 86400.0 <= 3 THEN '2. Late (1 to 3 days)'
            WHEN EXTRACT(EPOCH FROM (o.order_delivered_customer_date - o.order_estimated_delivery_date)) / 86400.0 <= 7 THEN '3. Late (4 to 7 days)'
            ELSE '4. Very Late (> 7 days)'
        END AS timing_bucket
    FROM orders o
    JOIN order_reviews r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
)
SELECT
    timing_bucket,
    COUNT(order_id) AS total_reviews,
    ROUND(AVG(review_score)::numeric, 2) AS avg_review_score,
    ROUND(
        100.0 * SUM(CASE WHEN review_score = 1 THEN 1 ELSE 0 END) / COUNT(order_id),
        2
    ) AS one_star_pct
FROM delivery_buckets
GROUP BY timing_bucket
ORDER BY timing_bucket;

-- [Q8_CARRIER_VS_SELLER_BOTTLENECK]
-- Decompose delayed orders: did the seller hand off late to the carrier, or did the carrier delay in transit?
WITH bottleneck_flags AS (
    SELECT
        o.order_id,
        CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END AS is_order_late,
        CASE WHEN o.order_delivered_carrier_date > MAX(oi.shipping_limit_date) THEN 1 ELSE 0 END AS seller_handoff_late
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
      AND o.order_delivered_carrier_date IS NOT NULL
    GROUP BY o.order_id, o.order_delivered_customer_date, o.order_estimated_delivery_date, o.order_delivered_carrier_date
)
SELECT
    seller_handoff_late,
    COUNT(order_id) AS total_orders,
    SUM(is_order_late) AS late_customer_deliveries,
    ROUND(100.0 * SUM(is_order_late) / COUNT(order_id), 2) AS customer_delay_rate_pct
FROM bottleneck_flags
GROUP BY seller_handoff_late
ORDER BY seller_handoff_late DESC;

-- [Q9_FREIGHT_RATIO_QUARTILES_NTILE]
-- Group orders into 4 quartiles by freight-to-price ratio using NTILE(4) and check delay rate per quartile
WITH order_financials AS (
    SELECT
        o.order_id,
        CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END AS is_late,
        SUM(oi.freight_value) / NULLIF(SUM(oi.price), 0) AS freight_ratio
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
    GROUP BY o.order_id, o.order_delivered_customer_date, o.order_estimated_delivery_date
),
quartiled AS (
    SELECT
        order_id,
        is_late,
        freight_ratio,
        NTILE(4) OVER (ORDER BY freight_ratio) AS freight_ratio_quartile
    FROM order_financials
    WHERE freight_ratio IS NOT NULL
)
SELECT
    freight_ratio_quartile,
    COUNT(order_id) AS total_orders,
    ROUND(MIN(freight_ratio)::numeric, 3) AS min_freight_ratio,
    ROUND(MAX(freight_ratio)::numeric, 3) AS max_freight_ratio,
    ROUND(100.0 * SUM(is_late) / COUNT(order_id), 2) AS delay_rate_pct
FROM quartiled
GROUP BY freight_ratio_quartile
ORDER BY freight_ratio_quartile;