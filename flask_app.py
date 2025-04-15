from flask import Flask, render_template, jsonify, request
import pandas as pd
import networkx as nx
import os
import io
import colorsys

app = Flask(__name__)

# Configure paths
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(APP_ROOT, 'network_data.csv')
CLUSTER_PATH = os.path.join(APP_ROOT, 'CLUSTER_DF.csv')
STATS_PATH = os.path.join(APP_ROOT, 'node_stats.csv')

HEADLINE_DATA_CSV = os.path.join(APP_ROOT, 'headline_data.csv')
CHAIN_DATA_CSV = os.path.join(APP_ROOT, 'chain_data.csv')

# Replacing old top_mint_burn.csv usage with net_mint_burn.csv (timeframe-based)
NET_MINT_BURN_CSV = os.path.join(APP_ROOT, 'net_mint_burn.csv')

SUPPLY_CSV = os.path.join(APP_ROOT, 'fdusd_supply.csv')


def load_data():
    """
    Example function that loads basic network-related CSVs
    (used by the / and /get_network routes).
    """
    try:
        network_df = pd.read_csv(DATA_PATH)
        cluster_df = pd.read_csv(CLUSTER_PATH)
        stats_df = pd.read_csv(STATS_PATH)
        return network_df, cluster_df, stats_df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback data if CSVs can’t be loaded
        network_df = pd.DataFrame({
            'DATE': ['2024-01-15','2024-01-16','2024-01-17','2024-01-18','2024-01-19','2024-01-20'],
            'FROM_ENTITY': ['A','B','A','C','D','E'],
            'TO_ENTITY':   ['B','C','D','E','F','A'],
            'VOLUME':      [100, 200, 150, 300, 250, 175]
        })
        cluster_df = pd.DataFrame({
            'label': ['A', 'B', 'C', 'D', 'E', 'F'],
            'cluster': [1, 1, 2, 2, 3, 3]
        })
        stats_df = pd.DataFrame({
            'address': ['A', 'B', 'C', 'D', 'E', 'F'],
            'total_volume': [1000, 2000, 1500, 3000, 2500, 1750],
            'total_transactions': [10, 20, 15, 30, 25, 17],
            'total_neighbors': [3, 2, 2, 2, 2, 1]
        })
        return network_df, cluster_df, stats_df


df, cluster_df, stats_df = load_data()

node_cluster_map = cluster_df.set_index('label')['cluster'].to_dict()
node_stats_map = stats_df.set_index('address').to_dict('index')


def format_stats(stats):
    return (
        f"Total Volume: {stats['total_volume']:,.2f}\n"
        f"Total Transactions: {stats['total_transactions']:,}\n"
        f"Total Neighbors: {stats['total_neighbors']:,}"
    )


def generate_cluster_color(cluster_id, total_clusters):
    """
    Generates a unique color for each cluster.
    """
    golden_ratio = 0.618033988749895
    hue = (cluster_id * golden_ratio) % 1
    saturation = 0.7
    lightness = 0.6
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    hex_color = "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )
    return hex_color


def get_cluster_color(cluster_id):
    total_clusters = len(set(cluster_df['cluster']))
    # Subtract 1 so cluster 1 => color index 0, cluster 2 => color index 1, etc.
    return generate_cluster_color(cluster_id - 1, total_clusters)


def aggregate_transactions(input_df):
    return (
        input_df
        .groupby(['FROM_ENTITY', 'TO_ENTITY'], as_index=False)
        .agg({
            'DATE': 'max',
            'VOLUME': 'sum'
        })
    )


def get_full_network():
    """
    Returns the entire (aggregated) network for rendering in the graph view.
    """
    try:
        agg_df = aggregate_transactions(df)
        G = nx.from_pandas_edgelist(
            agg_df,
            source='FROM_ENTITY',
            target='TO_ENTITY',
            edge_attr=True,
            create_using=nx.DiGraph()
        )

        nodes_data = []
        for node in G.nodes():
            cluster = node_cluster_map.get(node, 0)
            color = get_cluster_color(cluster)
            stats = node_stats_map.get(node, {
                'total_volume': 0,
                'total_transactions': 0,
                'total_neighbors': 0
            })
            nodes_data.append({
                "id": node,
                "label": node,
                "cluster": cluster,
                "color": color,
                "title": format_stats(stats)
            })

        edges_data = [
            {
                "from": u,
                "to": v,
                "value": data["VOLUME"],
                "date": data["DATE"],
                "title": f"Volume: {data['VOLUME']:,.2f}\nDate: {data['DATE']}"
            }
            for u, v, data in G.edges(data=True)
        ]

        return {"nodes": nodes_data, "edges": edges_data}
    except Exception as e:
        print(f"Error in get_full_network: {e}")
        return {"nodes": [], "edges": [], "error": str(e)}


def get_filtered_network(addresses, min_volume=0):
    """
    Filters the network to edges with VOLUME >= min_volume
    and nodes connected to the given addresses.

    'addresses' can be a list of 1 or more addresses.
    We gather all edges from or to each address, plus the addresses themselves.
    """
    try:
        df_filtered = df[df['VOLUME'] >= min_volume]
        agg_df = aggregate_transactions(df_filtered)
        G = nx.from_pandas_edgelist(
            agg_df,
            source='FROM_ENTITY',
            target='TO_ENTITY',
            edge_attr=True,
            create_using=nx.DiGraph()
        )

        # Start with the given addresses; add their neighbors
        nodes = set(addresses)
        for address in addresses:
            if address in G:
                # Add direct successors and predecessors
                nodes.update(list(G.successors(address)))
                nodes.update(list(G.predecessors(address)))

        subgraph = G.subgraph(nodes)

        nodes_data = []
        for node in subgraph.nodes():
            cluster = node_cluster_map.get(node, 0)
            color = get_cluster_color(cluster)
            stats = node_stats_map.get(node, {
                'total_volume': 0,
                'total_transactions': 0,
                'total_neighbors': 0
            })
            nodes_data.append({
                "id": node,
                "label": node,
                "cluster": cluster,
                "color": color,
                "title": format_stats(stats)
            })

        edges_data = [
            {
                "from": u,
                "to": v,
                "value": data["VOLUME"],
                "date": data["DATE"],
                "title": f"Volume: {data['VOLUME']:,.2f}\nDate: {data['DATE']}"
            }
            for u, v, data in subgraph.edges(data=True)
        ]

        return {"nodes": nodes_data, "edges": edges_data}
    except Exception as e:
        print(f"Error in get_filtered_network: {e}")
        return {"nodes": [], "edges": [], "error": str(e)}


def get_n_hops_network(addresses, n_hops, direction='both', min_volume=0):
    """
    Builds a subnetwork including n_hops around each address in 'addresses'.
    direction can be 'incoming', 'outgoing', or 'both'.
    """
    try:
        df_filtered = df[df['VOLUME'] >= min_volume]
        agg_df = aggregate_transactions(df_filtered)
        G = nx.from_pandas_edgelist(
            agg_df,
            source='FROM_ENTITY',
            target='TO_ENTITY',
            edge_attr=True,
            create_using=nx.DiGraph()
        )

        # Start set of nodes with all given addresses
        nodes = set(addresses)
        current_nodes = set(addresses)

        for _ in range(int(n_hops)):
            next_nodes = set()
            for node in current_nodes:
                if node in G:
                    if direction == 'outgoing' or direction == 'both':
                        next_nodes.update(list(G.successors(node)))
                    if direction == 'incoming' or direction == 'both':
                        next_nodes.update(list(G.predecessors(node)))
            nodes.update(next_nodes)
            current_nodes = next_nodes

        subgraph = G.subgraph(nodes)

        nodes_data = []
        for node in subgraph.nodes():
            cluster = node_cluster_map.get(node, 0)
            color = get_cluster_color(cluster)
            stats = node_stats_map.get(node, {
                'total_volume': 0,
                'total_transactions': 0,
                'total_neighbors': 0
            })
            nodes_data.append({
                "id": node,
                "label": node,
                "cluster": cluster,
                "color": color,
                "title": format_stats(stats)
            })

        edges_data = [
            {
                "from": u,
                "to": v,
                "value": data["VOLUME"],
                "date": data["DATE"],
                "title": f"Volume: {data['VOLUME']:,.2f}\nDate: {data['DATE']}"
            }
            for u, v, data in subgraph.edges(data=True)
        ]

        return {"nodes": nodes_data, "edges": edges_data}
    except Exception as e:
        print(f"Error in get_n_hops_network: {e}")
        return {"nodes": [], "edges": [], "error": str(e)}


@app.route('/')
def index():
    """
    Render the main network graph page (index.html).
    """
    entities = sorted(list(set(df['FROM_ENTITY'].unique()) | set(df['TO_ENTITY'].unique())))
    clusters = sorted(cluster_df['cluster'].unique())
    total_clusters = len(clusters)

    cluster_colors = {
        cluster: generate_cluster_color(i, total_clusters)
        for i, cluster in enumerate(clusters)
    }

    return render_template('index.html', entities=entities, clusters=cluster_colors)


@app.route('/get_network')
def get_network():
    """
    JSON endpoint for returning the entire network or a subnetwork.
    We use request.args.getlist('address') so multiple addresses can be selected.
    """
    addresses = request.args.getlist('address')  # returns a list of addresses
    n_hops = request.args.get('hops')
    direction = request.args.get('direction', 'both')
    min_volume = float(request.args.get('minVolume', 0))

    # If no addresses were specified, return the full network
    if not addresses:
        return jsonify(get_full_network())

    # If user specified n_hops, build n-hops subnetwork
    if n_hops:
        return jsonify(get_n_hops_network(addresses, int(n_hops), direction, min_volume))
    else:
        # Otherwise, just filter around the given addresses
        return jsonify(get_filtered_network(addresses, min_volume))


@app.route('/upload_addresses', methods=['POST'])
def upload_addresses():
    """
    POST endpoint for uploading a CSV of addresses to filter the network.
    CSV must have an 'address' column, each row has one address.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        content = file.read().decode('utf-8')
        df_addresses = pd.read_csv(io.StringIO(content))

        if 'address' not in df_addresses.columns:
            return jsonify({"error": "CSV must contain an 'address' column"}), 400

        addresses = df_addresses['address'].tolist()
        min_volume = float(request.form.get('minVolume', 0))

        network_data = get_filtered_network(addresses, min_volume)
        return jsonify(network_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# --------------------------------
# Overview Route
# --------------------------------
@app.route('/overview')
def overview():
    """
    Renders 'overview.html', showing:
      - "headline" data from headline_data.csv
      - chain amounts from chain_data.csv
      - usage charts (now included in overview.html)
    """
    # 1) Load "headline" data (assume only one row).
    try:
        headline_df = pd.read_csv(HEADLINE_DATA_CSV)
        row = headline_df.iloc[0].to_dict()
        headline = {
            'issued': row.get('issued', 0),
            'date': row.get('date', 'Unknown'),
            'past_month_amount': row.get('past_month_amount', 'N/A'),
            'past_month_change': row.get('past_month_change', '0%'),
            'past_week_amount': row.get('past_week_amount', 'N/A'),
            'past_week_change': row.get('past_week_change', '0%'),
            'past_24h_amount': row.get('past_24h_amount', 'N/A'),
            'past_24h_change': row.get('past_24h_change', '0%'),
        }
    except Exception as e:
        print(f"Error reading headline_data.csv: {e}")
        # Fallback values if CSV can’t be loaded
        headline = {
            'issued': 123456,
            'date': '2025-02-13',
            'past_month_amount': '9,876',
            'past_month_change': '2.3%',
            'past_week_amount': '5,000',
            'past_week_change': '1.1%',
            'past_24h_amount': '1,200',
            'past_24h_change': '0.5%'
        }

    # 2) Load chain data (multi-row). Assume columns: chain, amount
    try:
        chain_df = pd.read_csv(CHAIN_DATA_CSV)
        chains = chain_df.to_dict(orient='records')
    except Exception as e:
        print(f"Error reading chain_data.csv: {e}")
        # Fallback
        chains = [
            {'chain': 'Ethereum', 'amount': '80,000'},
            {'chain': 'BSC', 'amount': '40,000'},
            {'chain': 'Solana', 'amount': '10,500'},
        ]

    # We no longer load or pass net mint/burn data. That table is removed.

    # Render the (new) 'overview.html' template,
    # now including usage charts & table inside it.
    return render_template(
        'overview.html',
        headline=headline,
        chains=chains
    )


@app.route('/usage')
def usage():
    """
    Renders usage.html, passing in the list of distinct chain names
    from fdusd_supply.csv so the user can select them from the dropdown.
    """
    try:
        supply_df = pd.read_csv(SUPPLY_CSV)
        chains = sorted(supply_df['CHAIN'].dropna().unique().tolist())
    except Exception as e:
        print(f"Error reading supply CSV: {e}")
        chains = []

    return render_template('usage.html', chains=chains)


@app.route('/get_usage_data')
def get_usage_data():
    metric = request.args.get('metric', 'SUPPLY')
    valid_metrics = ['SUPPLY', 'VOLUME', 'TRANSACTIONS']
    if metric not in valid_metrics:
        metric = 'SUPPLY'

    try:
        # If the metric is SUPPLY, use total_supply.csv
        if metric == 'SUPPLY':
            csv_path = os.path.join(APP_ROOT, 'total_supply.csv')
        else:
            # For VOLUME or TRANSACTIONS, keep reading from fdusd_supply.csv
            csv_path = SUPPLY_CSV

        supply_df = pd.read_csv(csv_path)
        # Ensure DATE is a proper datetime
        supply_df['DATE'] = pd.to_datetime(supply_df['DATE'], errors='coerce')
    except Exception as e:
        print(f"Error reading supply CSV: {e}")
        return jsonify({"dates": [], "dataset": []}), 500

    if supply_df.empty:
        return jsonify({"dates": [], "dataset": []})

    # Drop rows with invalid or missing DATE
    supply_df = supply_df.dropna(subset=['DATE'])

    # Create a "MONTH" column in 'YYYY-MM' format
    supply_df['MONTH'] = supply_df['DATE'].dt.to_period('M').astype(str)

    # Group by [MONTH, CHAIN]
    if metric == 'SUPPLY':
        # For supply, take the monthly AVERAGE
        grouped = supply_df.groupby(['MONTH', 'CHAIN'], as_index=False)[metric].mean()
    else:
        # For volume or transactions, take the monthly SUM
        grouped = supply_df.groupby(['MONTH', 'CHAIN'], as_index=False)[metric].sum()

    # Pivot so that columns = CHAIN, rows = MONTH, values = metric
    pivot_df = grouped.pivot(index='MONTH', columns='CHAIN', values=metric).fillna(0)

    # Sort by month labels (which are strings like "2025-01"), so they appear chronologically
    pivot_df = pivot_df.sort_index()

    # Convert the row index (MONTH) to a list of x-axis labels
    dates = pivot_df.index.tolist()
    chains = pivot_df.columns.tolist()

    # Create a random color map for each chain
    import random
    random.seed(42)
    color_map = {}
    for ch in chains:
        r = random.randint(50, 200)
        g = random.randint(50, 200)
        b = random.randint(50, 200)
        color_map[ch] = f"rgba({r},{g},{b},0.8)"

    # Build the dataset for Chart.js
    dataset = []
    for ch in chains:
        values = pivot_df[ch].tolist()
        dataset.append({
            "label": ch,
            "data": values,
            "backgroundColor": color_map[ch]
        })

    return jsonify({
        "dates": dates,       # e.g. ["2024-01", "2024-02", ...]
        "dataset": dataset
    })

@app.route('/get_fdusd_supply_max_date')
def get_fdusd_supply_max_date():
    import pandas as pd
    import os

    csv_path = os.path.join(APP_ROOT, 'fdusd_supply_max_date.csv')
    try:
        df = pd.read_csv(csv_path)

        # We only need columns: CHAIN, APPLICATION, STABLECOIN_SUPPLY
        df = df[['CHAIN', 'APPLICATION', 'STABLECOIN_SUPPLY']]

        # Convert to a list of dicts
        data = df.to_dict(orient='records')
        return jsonify(data)

    except Exception as e:
        print("Error reading fdusd_supply_max_date.csv:", e)
        return jsonify({"error": str(e)}), 500


@app.route('/get_application_supply')
def get_application_supply():
    """
    Returns a JSON list of { 'APPLICATION': str, 'SUPPLY': float }
    summing the SUPPLY column across all dates/chains.
    """
    try:
        supply_df = pd.read_csv(SUPPLY_CSV)
    except Exception as e:
        print(f"Error reading supply CSV: {e}")
        return jsonify([]), 500

    if supply_df.empty:
        return jsonify([])

    grouped = supply_df.groupby('APPLICATION', as_index=False)['SUPPLY'].sum()
    grouped = grouped.sort_values(by='SUPPLY', ascending=False)

    result = grouped.to_dict(orient='records')
    return jsonify(result)


@app.route('/overlap')
def overlap():
    selected_timeframe = request.args.get('timeframe', '7d')

    try:
        nm_df = pd.read_csv(NET_MINT_BURN_CSV)
        nm_df = nm_df[nm_df['timeframe'] == selected_timeframe]
    except Exception as e:
        print(f"Error reading net_mint_burn.csv: {e}")
        nm_df = pd.DataFrame(columns=['address','timeframe','net_amount'])

    if nm_df.empty:
        return render_template('overlap.html', timeframe=selected_timeframe, data=[])

    try:
        network_df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Error reading network_data.csv: {e}")
        network_df = pd.DataFrame(columns=['DATE','FROM_ENTITY','TO_ENTITY','VOLUME'])

    G = nx.from_pandas_edgelist(
        network_df,
        source='FROM_ENTITY',
        target='TO_ENTITY',
        edge_attr=True,
        create_using=nx.DiGraph()
    )

    # Exclude 'binance' or 'FDUSD Treasury' nodes
    G_excluded = G.copy()
    excluded_nodes = [
        node for node in G_excluded.nodes
        if 'binance' in node.lower() or 'fdusd treasury' in node.lower()
    ]
    G_excluded.remove_nodes_from(excluded_nodes)

    address_list = nm_df['address'].unique().tolist()
    net_amount_map = nm_df.set_index('address')['net_amount'].to_dict()

    results = []
    for addr in address_list:
        row_result = {}
        row_result['address'] = addr
        row_result['net_amount'] = net_amount_map.get(addr, 0)

        # "Paths With Binance" => count of neighbors in original G
        if addr in G:
            in_neighbors = set(G.predecessors(addr))
            out_neighbors = set(G.successors(addr))
            row_result['paths_with_binance'] = len(in_neighbors.union(out_neighbors))
        else:
            row_result['paths_with_binance'] = 0

        # "Direct Paths (No Binance/FDUSD Treasury)" => shortest paths in G_excluded
        paths_info = []
        for other in address_list:
            if other == addr:
                continue
            if (addr not in G_excluded) or (other not in G_excluded):
                continue

            try:
                path_nodes = nx.shortest_path(G_excluded, source=addr, target=other)
                path_len = len(path_nodes) - 1
                # Create a space-delimited string like: "0x0c688 0x4f997 0x82b3e ..."
                path_str = " ".join(path_nodes)
                paths_info.append(f"{addr} -> {other}: length={path_len}, path={path_str}")
            except nx.NetworkXNoPath:
                pass

        row_result['direct_paths_no_binance'] = "\n".join(paths_info) if paths_info else "No paths"
        results.append(row_result)

    return render_template('overlap.html', timeframe=selected_timeframe, data=results)

@app.route('/get_net_burn_data')
def get_net_burn_data():
    """
    Return a single-series dataset of net burn amounts by month.
    CSV columns: MONTH, AMOUNT
    e.g.  2025-01,  120000
          2025-02,  90000
    """
    import pandas as pd
    import os

    net_burn_csv = os.path.join(APP_ROOT, 'net_burn_amount_time.csv')
    try:
        df = pd.read_csv(net_burn_csv)
        # We assume MONTH is a string like '2025-01'
        # and AMOUNT is numeric.

        # Convert MONTH column to string just in case:
        df['MONTH'] = df['MONTH'].astype(str)
        # Replace NaN in AMOUNT with 0
        df['AMOUNT'] = df['AMOUNT'].fillna(0)

        # Convert the columns into lists
        months = df['MONTH'].tolist()
        amounts = df['AMOUNT'].tolist()

        # Create a single chart.js dataset
        dataset = [
            {
                "label": "Net Burn",
                "data": amounts,
                "backgroundColor": "rgba(100, 150, 220, 0.7)"
            }
        ]

        return jsonify({
            "dates": months,    # x-axis labels (list of MONTH strings)
            "dataset": dataset  # a single dataset for net burn
        })

    except Exception as e:
        print("Error reading net_burn_amount_time.csv:", e)
        return jsonify({"dates": [], "dataset": []}), 500

@app.route('/get_dex_data')
def get_dex_data():
    """
    Returns data for a stacked bar chart of FDUSD DEX volume (USD_AMOUNT),
    grouped by (BLOCK_MONTH, BLOCKCHAIN).
    JSON format:
      {
        "dates": [...],
        "dataset": [
          {
            "label": "BSC",
            "data": [...],
            "backgroundColor": "rgba(...)"
          },
          ...
        ]
      }
    """
    import pandas as pd
    import os

    dex_csv = os.path.join(APP_ROOT, 'fdusd_dex_trades.csv')

    try:
        df = pd.read_csv(dex_csv)
        # We assume columns: BLOCK_MONTH, BLOCKCHAIN, USD_AMOUNT, TRADES
        # We'll use USD_AMOUNT only.

        # Convert BLOCK_MONTH to string or parse as date if you prefer
        df['BLOCK_MONTH'] = df['BLOCK_MONTH'].astype(str)

        # Group by (BLOCK_MONTH, BLOCKCHAIN) and sum USD_AMOUNT
        grouped = df.groupby(['BLOCK_MONTH', 'BLOCKCHAIN'], as_index=False)['USD_AMOUNT'].sum()

        # Pivot so columns = BLOCKCHAIN, rows = BLOCK_MONTH, values = USD_AMOUNT
        pivot_df = grouped.pivot(index='BLOCK_MONTH', columns='BLOCKCHAIN', values='USD_AMOUNT').fillna(0)

        # Sort by the month label (assuming "YYYY-MM" format)
        pivot_df = pivot_df.sort_index()

        # Convert pivot result into chart-friendly format
        dates = pivot_df.index.tolist()
        chains = pivot_df.columns.tolist()

        import random
        random.seed(42)
        dataset = []
        for ch in chains:
            # Create a random color for each chain
            r = random.randint(50, 200)
            g = random.randint(50, 200)
            b = random.randint(50, 200)
            color = f"rgba({r},{g},{b},0.8)"
            dataset.append({
                "label": ch,
                "data": pivot_df[ch].tolist(),
                "backgroundColor": color
            })

        return jsonify({
            "dates": dates,
            "dataset": dataset
        })

    except Exception as e:
        print("Error reading fdsusd_dex_trades.csv:", e)
        return jsonify({"dates": [], "dataset": []}), 500

@app.route('/get_binance_trading_data')
def get_binance_trading_data():
    """
    Returns line chart data from binance_trading_volume.csv, which has columns:
      - year_month (string, e.g. '2025-01')
      - stable_pair (e.g. 'FDUSD/USDT')
      - quote_volume (float)
    We pivot so each stable_pair is a separate line, and x-axis = year_month.
    """
    import pandas as pd
    import os

    csv_path = os.path.join(APP_ROOT, 'binance_trading_data.csv')
    try:
        df = pd.read_csv(csv_path)

        # Make sure columns exist
        # df has: year_month, stable_pair, quote_volume
        df['year_month'] = df['year_month'].astype(str)
        df['quote_volume'] = df['quote_volume'].fillna(0)

        # If year_month is something like '2025-01', you can keep it as a string
        # or parse as datetime:
        # df['year_month'] = pd.to_datetime(df['year_month'], format='%Y-%m', errors='coerce')
        # df['year_month'] = df['year_month'].dt.to_period('M').astype(str)

        # Pivot so columns = stable_pair, index = year_month, values = quote_volume
        pivot_df = df.pivot(
            index='year_month',
            columns='stable_pair',
            values='quote_volume'
        ).fillna(0)

        # Sort the rows by ascending year_month label
        pivot_df = pivot_df.sort_index()

        # X-axis labels
        dates = pivot_df.index.tolist()   # e.g. ["2025-01","2025-02", ...]

        # stable_pair columns
        pairs = pivot_df.columns.tolist()

        # Build the line chart datasets
        import random
        random.seed(42)
        dataset = []
        for p in pairs:
            # random color
            r = random.randint(50, 200)
            g = random.randint(50, 200)
            b = random.randint(50, 200)
            color = f"rgba({r},{g},{b},1.0)"  # full opacity for line stroke

            # For a line chart, we use "borderColor" for the stroke,
            # and optionally "backgroundColor" for the fill (e.g. transparent).
            dataset.append({
                "label": p,
                "data": pivot_df[p].tolist(),
                "borderColor": color,
                "backgroundColor": "rgba(0,0,0,0)",  # transparent fill
                "tension": 0.2,   # slight curve if you like
                "fill": False     # no fill under the line
            })

        return jsonify({
            "dates": dates,
            "dataset": dataset
        })

    except Exception as e:
        print("Error reading binance_trading_volume.csv:", e)
        return jsonify({"dates": [], "dataset": []}), 500


@app.route('/get_holders_data')
def get_holders_data():
    """
    Reads holders_over_time.csv, which has columns:
      DATE, CHAIN, HOLDERS
    Groups daily, pivoting so each CHAIN is its own line, x-axis = DATE, y-axis = HOLDERS.
    Returns JSON for a Chart.js line chart.
    """
    import pandas as pd
    import os

    csv_path = os.path.join(APP_ROOT, 'holders_over_time.csv')
    try:
        df = pd.read_csv(csv_path)

        # Ensure date is properly parsed
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        # Drop rows with invalid or missing DATE
        df = df.dropna(subset=['DATE'])

        # Sort by date ascending, just to keep things in order
        df = df.sort_values(by='DATE')

        # Pivot so columns = CHAIN, index = DATE, values = HOLDERS
        pivot_df = df.pivot(index='DATE', columns='CHAIN', values='HOLDERS').fillna(0)

        # Convert the DATE index to string for the chart's x-axis labels
        # If you want daily labels in "YYYY-MM-DD" format:
        pivot_df.index = pivot_df.index.strftime('%Y-%m-%d')

        # Create the lists needed for Chart.js
        dates = pivot_df.index.tolist()     # e.g. ["2025-01-01","2025-01-02",...]
        chains = pivot_df.columns.tolist()

        # Build a line dataset for each chain
        import random
        random.seed(42)
        dataset = []
        for ch in chains:
            # Random color
            r = random.randint(50, 200)
            g = random.randint(50, 200)
            b = random.randint(50, 200)
            color = f"rgba({r},{g},{b},1.0)"  # line color

            dataset.append({
                "label": ch,
                "data": pivot_df[ch].tolist(),       # list of holder counts
                "borderColor": color,
                "backgroundColor": "rgba(0,0,0,0)",  # transparent fill
                "tension": 0.2,  # slight curve
                "fill": False
            })

        return jsonify({
            "dates": dates,
            "dataset": dataset
        })

    except Exception as e:
        print("Error reading holders_over_time.csv:", e)
        return jsonify({"dates": [], "dataset": []}), 500



if __name__ == '__main__':
    app.run(debug=True)
