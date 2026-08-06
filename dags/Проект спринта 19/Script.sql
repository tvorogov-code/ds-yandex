WITH params AS
(
	SELECT date('2025-10-01') AS run_date
),
event_types_7 AS 
(
	SELECT 
		customer_id, 
		SUM(CASE WHEN event_type = 'page_view' THEN 1 ELSE 0 END) AS page_views_7,
		SUM(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) AS add_to_carts_7,
		SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchases_7,
		COUNT(DISTINCT product_id) AS distinct_products_7
	FROM 
		public.events 
		INNER JOIN public.sessions USING(session_id)
		CROSS JOIN params
	WHERE
		date_trunc('day', timestamp) between run_date - 7 AND run_date - 1
	GROUP BY 
		customer_id
),
event_types_30 AS 
(
	SELECT 
		customer_id, 
		SUM(CASE WHEN event_type = 'page_view' THEN 1 ELSE 0 END) AS page_views_30,
		SUM(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) AS add_to_carts_30,
		SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchases_30,
		COUNT(DISTINCT product_id) AS distinct_products_30
	FROM 
		public.events 
		INNER JOIN public.sessions USING(session_id)
		CROSS JOIN params
	WHERE
		date_trunc('day', timestamp) between run_date - 30 AND run_date - 1
	GROUP BY 
		customer_id
),
sessions_7 AS
(
	SELECT 	
		customer_id,
		count(*) AS sessions_count_7
	FROM public_sessions
	WHERE
		date_trunc('day', start_time) between run_date - 7 AND run_date - 1
	GROUP BY customer_id
		
),
sessions_30 AS
(
	SELECT 	
		customer_id,
		count(*) AS sessions_count_30,
		
	FROM 
		public.sessions
	WHERE
		date_trunc('day', start_time) between run_date - 30 AND run_date - 1
	GROUP BY customer_id		
)
SELECT
	page_views_7,
	page_views_30,
	add_to_carts_7,
	add_to_carts_30,
	add_to_carts_7::numeric / NULLIF(page_views_7, 0) AS cart_conv_7,
	add_to_carts_30::numeric / NULLIF(page_views_30, 0) AS cart_conv_30,
	purchases_7::numeric / NULLIF(add_to_carts_7, 0) AS purchase_conv_7,
	purchases_30::numeric / NULLIF(add_to_carts_30, 0) AS purchase_conv_30,
	distinct_products_7,
	distinct_products_30
FROM 
	event_types_7
	INNER JOIN event_types_30 USING(customer_id)


