"""
Shipment Optimizer - Streamlit Web Application
Upload CSV files and run optimization to generate callback schedules
"""

import streamlit as st
import pandas as pd
import os
import sys
import tempfile
import time
from io import StringIO

# Ensure stdout/stderr do not crash on non-ASCII console encodings
try:
    sys.stdout.reconfigure(errors="ignore")
    sys.stderr.reconfigure(errors="ignore")
except Exception:
    pass

# ================================================================
# CONFIGURATION
# ================================================================

# Skip Vertica lookups for web app (can be toggled)
SKIP_VERTICA_LOOKUPS = True

# Optimization parameters
PLANNING_HORIZON_DAYS = 150
UNLIMITED_CAPACITY = 999999

# Objective function weights
ALPHA_EARLY = 1
BETA_DET = 50
GAMMA_DEM = 40
RUNOUT_LATE_PENALTY = 100000

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def normalize_dayofweek(s: pd.Series) -> pd.Series:
    """Normalize day of week to capitalized format"""
    return s.astype(str).str.strip().str.capitalize()

def normalize_key(s: pd.Series) -> pd.Series:
    """Normalize string keys by stripping whitespace"""
    return s.astype(str).str.strip()

def dow(ts: pd.Timestamp) -> str:
    """Get day of week name from timestamp"""
    return ts.strftime("%A")

# ================================================================
# DATA LOADING FUNCTIONS
# ================================================================

def load_shipments_from_upload(uploaded_file):
    """Load and preprocess shipment data from uploaded CSV"""
    try:
        shipments_raw = pd.read_csv(uploaded_file, skip_blank_lines=True)
        shipments_raw.dropna(how='all', inplace=True)
        shipments_raw.columns = shipments_raw.columns.str.strip()
        
        # Normalize key columns
        shipments_raw["Shipment Number"] = normalize_key(shipments_raw["Shipment Number"])
        shipments_raw["Ship To Location Name"] = normalize_key(shipments_raw["Ship To Location Name"])
        shipments_raw["Drayageparty"] = normalize_key(shipments_raw["Drayageparty"])
        
        # Create a cleaned copy for optimization
        shipments = shipments_raw.copy()
        
        # Drop rows with missing critical fields
        initial_count = len(shipments)
        shipments.dropna(subset=["Ship To Location Name", "Drayageparty"], inplace=True)
        shipments.reset_index(drop=True, inplace=True)
        
        dropped = initial_count - len(shipments)
        
        return shipments, shipments_raw, dropped
        
    except Exception as e:
        raise Exception(f"Failed to load shipments: {e}")

def load_warehouse_capacity_from_upload(uploaded_file):
    """Load warehouse capacity data from uploaded CSV"""
    try:
        warehouse_cap = pd.read_csv(uploaded_file, skip_blank_lines=True)
        warehouse_cap.dropna(how='all', inplace=True)
        warehouse_cap.columns = warehouse_cap.columns.str.strip()
        warehouse_cap["Warehouse"] = normalize_key(warehouse_cap["Warehouse"])
        warehouse_cap["DayOfWeek"] = normalize_dayofweek(warehouse_cap["DayOfWeek"])
        warehouse_cap["MaxContainers"] = pd.to_numeric(
            warehouse_cap["MaxContainers"], errors="coerce"
        ).fillna(0).astype(int)
        return warehouse_cap
    except Exception as e:
        raise Exception(f"Failed to load warehouse capacity: {e}")

def load_carrier_capacity_from_upload(uploaded_file):
    """Load carrier capacity data from uploaded CSV"""
    try:
        carrier_cap = pd.read_csv(uploaded_file, skip_blank_lines=True)
        carrier_cap.dropna(how='all', inplace=True)
        carrier_cap.columns = carrier_cap.columns.str.strip()
        carrier_cap["Drayageparty"] = normalize_key(carrier_cap["Drayageparty"])
        carrier_cap["DayOfWeek"] = normalize_dayofweek(carrier_cap["DayOfWeek"])
        carrier_cap["MaxContainers"] = pd.to_numeric(
            carrier_cap["MaxContainers"], errors="coerce"
        ).fillna(0).astype(int)
        return carrier_cap
    except Exception as e:
        raise Exception(f"Failed to load carrier capacity: {e}")

# ================================================================
# PREPROCESSING
# ================================================================

def preprocess_shipments(shipments, shipments_original=None):
    """Preprocess shipment data"""
    
    if shipments_original is None:
        shipments_original = shipments.copy()
    
    # Ensure Edray column exists
    if "Edray Gate Out Full Date" not in shipments.columns:
        shipments["Edray Gate Out Full Date"] = pd.NaT
    
    # Create source tracking column
    shipments["Min Date Source"] = shipments["Edray Gate Out Full Date"].apply(
        lambda x: "Edray Gate Out Full Date" if pd.notna(x) else "Min of Delivery Date"
    )
    
    # Convert date columns
    date_cols = [
        "Min of Delivery Date",
        "Run out date",
        "Detention Last Free date",
        "Demurrage Last Free date"
    ]
    
    for col in date_cols:
        if col in shipments.columns:
            shipments[col] = pd.to_datetime(shipments[col], errors="coerce")
    
    # Create Effective Min Date
    shipments["Effective_Min_Date"] = shipments["Edray Gate Out Full Date"].fillna(
        shipments["Min of Delivery Date"]
    )
    
    # Ensure DestinationLocationKey column exists
    if "DestinationLocationKey" not in shipments.columns:
        shipments["DestinationLocationKey"] = None
    
    # Consolidate by Shipment Number
    agg_dict = {
        "Effective_Min_Date": "min",
        "Min of Delivery Date": "min",
        "Run out date": "min",
        "Detention Last Free date": "min",
        "Demurrage Last Free date": "min",
        "Ship To Location Name": "first",
        "Drayageparty": "first",
        "DestinationLocationKey": "first",
        "Min Date Source": "first"
    }
    
    # Only include columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in shipments.columns}
    
    shipments_consolidated = shipments.groupby("Shipment Number", as_index=False).agg(agg_dict)
    
    # Copy other columns from first occurrence
    for col in shipments.columns:
        if col not in shipments_consolidated.columns and col != "Shipment Number":
            first_vals = shipments.groupby("Shipment Number")[col].first()
            shipments_consolidated[col] = shipments_consolidated["Shipment Number"].map(first_vals)
    
    shipments = shipments_consolidated
    
    # Flag invalid shipments
    shipments["Invalid_MinAfterRunout"] = (
        shipments["Effective_Min_Date"].notna() &
        shipments["Run out date"].notna() &
        (shipments["Effective_Min_Date"] > shipments["Run out date"])
    )
    
    # Separate excluded shipments
    shipments_excluded = shipments[shipments["Invalid_MinAfterRunout"]].copy()
    shipments_excluded["Callback Reason"] = "Min Delivery Date after Runout Date"
    shipments_excluded["Callback Date"] = pd.NaT
    shipments_excluded["LateDays_vs_Runout"] = pd.NA
    shipments_excluded["Allocation_Source"] = "N/A"
    
    # Keep valid shipments
    shipments_opt = shipments[~shipments["Invalid_MinAfterRunout"]].copy()
    shipments_opt = shipments_opt.reset_index(drop=True)
    
    return shipments_opt, shipments_excluded, shipments_original

# ================================================================
# OPTIMIZATION MODEL
# ================================================================

def build_and_solve_model(shipments_opt, warehouse_cap, carrier_cap, planning_days, time_limit_minutes, progress_callback=None):
    """Build and solve the optimization model"""
    from ortools.linear_solver import pywraplp
    
    today = pd.Timestamp.today().normalize()
    horizon_end = today + pd.Timedelta(days=planning_days)
    all_days = pd.date_range(start=today, end=horizon_end, freq="D")
    
    warehouses = shipments_opt["Ship To Location Name"].dropna().unique().tolist()
    carriers = shipments_opt["Drayageparty"].dropna().unique().tolist()
    
    if progress_callback:
        progress_callback(0.1, "Building capacity maps...")
    
    # Build capacity maps
    warehouse_capacity = {}
    for d in all_days:
        wd = dow(d)
        for wh in warehouses:
            row = warehouse_cap[
                (warehouse_cap["Warehouse"] == wh) & 
                (warehouse_cap["DayOfWeek"] == wd)
            ]
            warehouse_capacity[(wh, d)] = (
                int(row["MaxContainers"].iloc[0]) if not row.empty else UNLIMITED_CAPACITY
            )
    
    carrier_capacity = {}
    for d in all_days:
        wd = dow(d)
        for car in carriers:
            row = carrier_cap[
                (carrier_cap["Drayageparty"] == car) & 
                (carrier_cap["DayOfWeek"] == wd)
            ]
            carrier_capacity[(car, d)] = (
                int(row["MaxContainers"].iloc[0]) if not row.empty else UNLIMITED_CAPACITY
            )
    
    # Pre-build mappings
    shipment_warehouse = shipments_opt["Ship To Location Name"].to_dict()
    shipment_carrier = shipments_opt["Drayageparty"].to_dict()
    
    # Build carrier LiveUnload flag map
    carrier_live_map = {}
    if "LiveUnload" in carrier_cap.columns:
        for car in carriers:
            row = carrier_cap[carrier_cap["Drayageparty"] == car]
            if not row.empty:
                live_val = row["LiveUnload"].iloc[0]
                if isinstance(live_val, bool):
                    carrier_live_map[car] = live_val
                else:
                    s = str(live_val).strip().lower()
                    carrier_live_map[car] = s in ("y", "yes", "true", "1", "t")
            else:
                carrier_live_map[car] = False
    else:
        for car in carriers:
            carrier_live_map[car] = False
    
    if progress_callback:
        progress_callback(0.2, "Creating solver and variables...")
    
    # Create solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise RuntimeError("Failed to create SCIP solver")
    
    ship_list = shipments_opt.index.tolist()
    day_list = list(all_days)
    
    # Create binary variables
    x = {}
    for s in ship_list:
        for d in day_list:
            x[s, d] = solver.BoolVar(f'x_{s}_{d}')
    
    if progress_callback:
        progress_callback(0.3, "Adding constraints...")
    
    # Constraint 1: Each shipment assigned to exactly one day
    for s in ship_list:
        solver.Add(sum(x[s, d] for d in day_list) == 1)
    
    # Constraint 2: Respect minimum delivery date
    for s in ship_list:
        M = shipments_opt.loc[s, "Effective_Min_Date"]
        if pd.notna(M):
            for d in day_list:
                if d < M:
                    solver.Add(x[s, d] == 0)
    
    # Constraint 3: Warehouse daily capacity
    shipments_by_warehouse = {}
    for s in ship_list:
        wh = shipment_warehouse.get(s)
        if pd.notna(wh):
            if wh not in shipments_by_warehouse:
                shipments_by_warehouse[wh] = []
            shipments_by_warehouse[wh].append(s)
    
    for d in day_list:
        for wh in warehouses:
            cap = warehouse_capacity.get((wh, d), UNLIMITED_CAPACITY)
            if cap >= UNLIMITED_CAPACITY:
                continue
            ships_for_wh = shipments_by_warehouse.get(wh, [])
            if ships_for_wh:
                solver.Add(sum(x[s, d] for s in ships_for_wh) <= cap)
    
    # Constraint 4: Carrier daily capacity
    shipments_by_carrier = {}
    for s in ship_list:
        car = shipment_carrier.get(s)
        if pd.notna(car):
            if car not in shipments_by_carrier:
                shipments_by_carrier[car] = []
            shipments_by_carrier[car].append(s)
    
    for d in day_list:
        for car in carriers:
            cap = carrier_capacity.get((car, d), UNLIMITED_CAPACITY)
            if cap >= UNLIMITED_CAPACITY:
                continue
            ships_for_car = shipments_by_carrier.get(car, [])
            if ships_for_car:
                solver.Add(sum(x[s, d] for s in ships_for_car) <= cap)
    
    if progress_callback:
        progress_callback(0.4, "Building objective function...")
    
    # Objective function
    objective = solver.Objective()
    
    for s in ship_list:
        runout = shipments_opt.loc[s, "Run out date"]
        det_lfd = shipments_opt.loc[s, "Detention Last Free date"]
        dem_lfd = shipments_opt.loc[s, "Demurrage Last Free date"]
        
        for d in day_list:
            day_index = (d - today).days
            det_pen = max(0, (d - det_lfd).days) if pd.notna(det_lfd) else 0
            dem_pen = max(0, (d - dem_lfd).days) if pd.notna(dem_lfd) else 0
            runout_late_days = max(0, (d - runout).days) if pd.notna(runout) else 0
            runout_pen = RUNOUT_LATE_PENALTY * runout_late_days
            
            cost = (ALPHA_EARLY * day_index + 
                    BETA_DET * det_pen + 
                    GAMMA_DEM * dem_pen + 
                    runout_pen)
            
            objective.SetCoefficient(x[s, d], cost)
    
    objective.SetMinimization()
    
    if progress_callback:
        progress_callback(0.5, f"Solving (time limit: {time_limit_minutes} minutes)...")
    
    # Solve
    solver.SetTimeLimit(time_limit_minutes * 60 * 1000)
    status = solver.Solve()
    
    if progress_callback:
        progress_callback(0.8, "Extracting solution...")
    
    # Extract solution
    solution_data = []
    for s in ship_list:
        for d in day_list:
            if x[s, d].solution_value() > 0.5:
                carrier = shipment_carrier.get(s)
                is_live_unload = carrier_live_map.get(carrier, False)
                allocation_source = "Appointments" if is_live_unload else "Warehouse fixed capacity"
                
                solution_data.append({
                    'Shipment_Index': s,
                    'Assigned_Date': d,
                    'Warehouse': shipment_warehouse.get(s),
                    'Carrier': carrier,
                    'Allocation_Source': allocation_source
                })
                break
    
    solution_df = pd.DataFrame(solution_data)
    
    return solution_df, status, shipment_warehouse, shipment_carrier, carrier_live_map

# ================================================================
# OUTPUT PREPARATION
# ================================================================

def prepare_final_output(shipments_opt, shipments_excluded, solution_df, shipments_original):
    """Prepare final output with callback dates"""
    
    shipments_opt_final = shipments_opt.copy()
    shipments_opt_final["Callback Date"] = pd.NaT
    shipments_opt_final["Callback Reason"] = "Scheduled"
    shipments_opt_final["LateDays_vs_Runout"] = 0
    shipments_opt_final["Allocation_Source"] = "N/A"
    
    if len(solution_df) > 0:
        date_mapping = solution_df.set_index('Shipment_Index')['Assigned_Date']
        shipments_opt_final.loc[date_mapping.index, "Callback Date"] = date_mapping
        
        source_mapping = solution_df.set_index('Shipment_Index')['Allocation_Source']
        shipments_opt_final.loc[source_mapping.index, "Allocation_Source"] = source_mapping
    
    mask_runout = (
        shipments_opt_final["Run out date"].notna() & 
        shipments_opt_final["Callback Date"].notna()
    )
    if mask_runout.any():
        shipments_opt_final.loc[mask_runout, "LateDays_vs_Runout"] = (
            (shipments_opt_final.loc[mask_runout, "Callback Date"] - 
             shipments_opt_final.loc[mask_runout, "Run out date"]).dt.days
        ).clip(lower=0)
    
    # Build mappings
    callback_date_map = shipments_opt_final.set_index("Shipment Number")["Callback Date"].to_dict()
    callback_reason_map = shipments_opt_final.set_index("Shipment Number")["Callback Reason"].to_dict()
    allocation_source_map = shipments_opt_final.set_index("Shipment Number")["Allocation_Source"].to_dict()
    late_days_map = shipments_opt_final.set_index("Shipment Number")["LateDays_vs_Runout"].to_dict()
    
    # Expand excluded shipments
    excluded_shipnums = shipments_excluded["Shipment Number"].unique()
    excluded_expanded = shipments_original[shipments_original["Shipment Number"].isin(excluded_shipnums)].copy()
    excluded_expanded["Callback Date"] = pd.NaT
    excluded_expanded["Callback Reason"] = "Min Delivery Date after Runout Date"
    excluded_expanded["LateDays_vs_Runout"] = pd.NA
    excluded_expanded["Allocation_Source"] = "N/A"
    
    # Build final output
    final_output_parts = shipments_original[~shipments_original["Shipment Number"].isin(excluded_shipnums)].copy()
    final_output_parts["Callback Date"] = pd.NaT
    final_output_parts["Callback Reason"] = "Scheduled"
    final_output_parts["LateDays_vs_Runout"] = 0
    final_output_parts["Allocation_Source"] = "N/A"
    
    for ship_num in final_output_parts["Shipment Number"].unique():
        if ship_num in callback_date_map and pd.notna(callback_date_map[ship_num]):
            mask = final_output_parts["Shipment Number"] == ship_num
            final_output_parts.loc[mask, "Callback Date"] = callback_date_map[ship_num]
            final_output_parts.loc[mask, "Callback Reason"] = callback_reason_map.get(ship_num, "Scheduled")
            final_output_parts.loc[mask, "Allocation_Source"] = allocation_source_map.get(ship_num, "N/A")
            final_output_parts.loc[mask, "LateDays_vs_Runout"] = late_days_map.get(ship_num, 0)
    
    final_output = pd.concat([
        final_output_parts,
        excluded_expanded
    ], axis=0, ignore_index=True)
    
    return final_output

def create_output_files(final_output):
    """Create all output dataframes"""
    
    # Main output
    main_output = final_output.copy()
    
    # Scheduled shipments
    scheduled_df = final_output[final_output['Callback Reason'] == 'Scheduled'].copy()
    
    # Callback shipments (excluded/problematic)
    callbacks_df = final_output[final_output['Callback Reason'] != 'Scheduled'].copy()
    
    # Daily schedule
    daily_schedule = pd.DataFrame()
    if len(scheduled_df) > 0:
        daily_schedule = scheduled_df.groupby('Callback Date').agg({
            'Shipment Number': 'count'
        }).rename(columns={'Shipment Number': 'Total_Shipments'}).reset_index()
    
    return main_output, scheduled_df, callbacks_df, daily_schedule

# ================================================================
# STREAMLIT APP
# ================================================================

def main():
    st.set_page_config(
        page_title="Shipment Optimizer",
        page_icon="📦",
        layout="wide"
    )
    
    st.title("📦 Shipment Optimizer")
    st.markdown("Upload your input files and run the optimization to generate callback schedules.")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    planning_days = st.sidebar.slider(
        "Planning Horizon (days)",
        min_value=30,
        max_value=365,
        value=150,
        help="Number of days to plan ahead"
    )
    
    time_limit = st.sidebar.slider(
        "Solver Time Limit (minutes)",
        min_value=1,
        max_value=60,
        value=10,
        help="Maximum time for the solver to run"
    )
    
    # File upload section
    st.header("📁 Upload Input Files")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Callback Input")
        shipments_file = st.file_uploader(
            "Upload Callbackinput.csv",
            type=['csv'],
            key='shipments',
            help="Main shipment data file"
        )
        if shipments_file:
            st.success("✅ Uploaded")
    
    with col2:
        st.subheader("Warehouse Capacity")
        warehouse_file = st.file_uploader(
            "Upload Warehousecapacity.csv",
            type=['csv'],
            key='warehouse',
            help="Warehouse capacity constraints"
        )
        if warehouse_file:
            st.success("✅ Uploaded")
    
    with col3:
        st.subheader("Carrier Capacity")
        carrier_file = st.file_uploader(
            "Upload DrayageCarriercapacity.csv",
            type=['csv'],
            key='carrier',
            help="Carrier capacity constraints"
        )
        if carrier_file:
            st.success("✅ Uploaded")
    
    # Preview uploaded files
    if shipments_file or warehouse_file or carrier_file:
        st.header("👀 Preview Uploaded Data")
        
        preview_col1, preview_col2, preview_col3 = st.columns(3)
        
        with preview_col1:
            if shipments_file:
                try:
                    shipments_file.seek(0)
                    preview_df = pd.read_csv(shipments_file, nrows=5)
                    st.write("**Shipments Preview:**")
                    st.dataframe(preview_df, use_container_width=True)
                    shipments_file.seek(0)
                except Exception as e:
                    st.error(f"Error previewing: {e}")
        
        with preview_col2:
            if warehouse_file:
                try:
                    warehouse_file.seek(0)
                    preview_df = pd.read_csv(warehouse_file, nrows=5)
                    st.write("**Warehouse Capacity Preview:**")
                    st.dataframe(preview_df, use_container_width=True)
                    warehouse_file.seek(0)
                except Exception as e:
                    st.error(f"Error previewing: {e}")
        
        with preview_col3:
            if carrier_file:
                try:
                    carrier_file.seek(0)
                    preview_df = pd.read_csv(carrier_file, nrows=5)
                    st.write("**Carrier Capacity Preview:**")
                    st.dataframe(preview_df, use_container_width=True)
                    carrier_file.seek(0)
                except Exception as e:
                    st.error(f"Error previewing: {e}")
    
    # Run optimization
    st.header("🚀 Run Optimization")
    
    all_files_uploaded = shipments_file and warehouse_file and carrier_file
    
    if not all_files_uploaded:
        st.warning("⚠️ Please upload all three required files to run the optimization.")
    
    run_button = st.button(
        "Run Optimization",
        disabled=not all_files_uploaded,
        type="primary",
        use_container_width=True
    )
    
    if run_button and all_files_uploaded:
        try:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(value, text):
                progress_bar.progress(value)
                status_text.text(text)
            
            # Load data
            update_progress(0.05, "Loading shipment data...")
            shipments_file.seek(0)
            shipments, shipments_raw, dropped = load_shipments_from_upload(shipments_file)
            
            update_progress(0.1, "Loading warehouse capacity...")
            warehouse_file.seek(0)
            warehouse_cap = load_warehouse_capacity_from_upload(warehouse_file)
            
            update_progress(0.15, "Loading carrier capacity...")
            carrier_file.seek(0)
            carrier_cap = load_carrier_capacity_from_upload(carrier_file)
            
            # Display data summary
            st.info(f"""
            **Data Summary:**
            - Total shipments loaded: {len(shipments_raw)}
            - Shipments for optimization: {len(shipments)}
            - Dropped (missing data): {dropped}
            - Warehouses: {warehouse_cap['Warehouse'].nunique()}
            - Carriers: {carrier_cap['Drayageparty'].nunique()}
            """)
            
            # Preprocess
            update_progress(0.2, "Preprocessing shipments...")
            shipments_opt, shipments_excluded, shipments_original = preprocess_shipments(
                shipments, shipments_raw
            )
            
            st.info(f"""
            **Preprocessing Summary:**
            - Shipments for optimization: {len(shipments_opt)}
            - Excluded (min > runout): {len(shipments_excluded)}
            """)
            
            # Build and solve model
            solution_df, status, shipment_warehouse, shipment_carrier, carrier_live_map = build_and_solve_model(
                shipments_opt, warehouse_cap, carrier_cap, 
                planning_days, time_limit, update_progress
            )
            
            # Check solver status
            from ortools.linear_solver import pywraplp
            if status == pywraplp.Solver.OPTIMAL:
                st.success("✅ OPTIMAL solution found!")
            elif status == pywraplp.Solver.FEASIBLE:
                st.success("✅ FEASIBLE solution found!")
            else:
                st.error(f"❌ Optimization failed with status: {status}")
                return
            
            update_progress(0.9, "Preparing output files...")
            
            # Prepare outputs
            final_output = prepare_final_output(
                shipments_opt, shipments_excluded, solution_df, shipments_original
            )
            
            main_output, scheduled_df, callbacks_df, daily_schedule = create_output_files(final_output)
            
            update_progress(1.0, "Complete!")
            
            # Store results in session state
            st.session_state['main_output'] = main_output
            st.session_state['scheduled_df'] = scheduled_df
            st.session_state['callbacks_df'] = callbacks_df
            st.session_state['daily_schedule'] = daily_schedule
            st.session_state['optimization_complete'] = True
            
            st.success("🎉 Optimization complete!")
            
        except Exception as e:
            st.error(f"❌ Error during optimization: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Display results and download buttons
    if st.session_state.get('optimization_complete', False):
        st.header("📊 Results")
        
        # Summary metrics
        main_output = st.session_state['main_output']
        scheduled_df = st.session_state['scheduled_df']
        callbacks_df = st.session_state['callbacks_df']
        daily_schedule = st.session_state['daily_schedule']
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Shipments", len(main_output))
        
        with metric_col2:
            st.metric("Scheduled", len(scheduled_df))
        
        with metric_col3:
            st.metric("Callbacks/Excluded", len(callbacks_df))
        
        with metric_col4:
            if len(scheduled_df) > 0 and 'LateDays_vs_Runout' in scheduled_df.columns:
                late = scheduled_df[scheduled_df['LateDays_vs_Runout'] > 0]
                st.metric("Late vs Runout", len(late))
            else:
                st.metric("Late vs Runout", 0)
        
        # Download section
        st.header("📥 Download Output Files")
        
        download_col1, download_col2, download_col3, download_col4 = st.columns(4)
        
        with download_col1:
            csv = main_output.to_csv(index=False)
            st.download_button(
                label="📄 Callback_Output.csv",
                data=csv,
                file_name="Callback_Output.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.caption(f"{len(main_output)} rows")
        
        with download_col2:
            if len(scheduled_df) > 0:
                csv = scheduled_df.to_csv(index=False)
                st.download_button(
                    label="📄 Callback_Output_scheduled.csv",
                    data=csv,
                    file_name="Callback_Output_scheduled.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption(f"{len(scheduled_df)} rows")
            else:
                st.button("📄 No scheduled shipments", disabled=True, use_container_width=True)
        
        with download_col3:
            if len(callbacks_df) > 0:
                csv = callbacks_df.to_csv(index=False)
                st.download_button(
                    label="📄 Callback_Output_callbacks.csv",
                    data=csv,
                    file_name="Callback_Output_callbacks.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption(f"{len(callbacks_df)} rows")
            else:
                st.button("📄 No callback shipments", disabled=True, use_container_width=True)
        
        with download_col4:
            if len(daily_schedule) > 0:
                csv = daily_schedule.to_csv(index=False)
                st.download_button(
                    label="📄 Callback_Output_daily_schedule.csv",
                    data=csv,
                    file_name="Callback_Output_daily_schedule.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption(f"{len(daily_schedule)} rows")
            else:
                st.button("📄 No daily schedule", disabled=True, use_container_width=True)
        
        # Preview results
        st.header("👁️ Preview Results")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Main Output", "Scheduled", "Callbacks", "Daily Schedule"
        ])
        
        with tab1:
            st.dataframe(main_output.head(100), use_container_width=True)
        
        with tab2:
            if len(scheduled_df) > 0:
                st.dataframe(scheduled_df.head(100), use_container_width=True)
            else:
                st.info("No scheduled shipments")
        
        with tab3:
            if len(callbacks_df) > 0:
                st.dataframe(callbacks_df.head(100), use_container_width=True)
            else:
                st.info("No callback shipments")
        
        with tab4:
            if len(daily_schedule) > 0:
                st.dataframe(daily_schedule, use_container_width=True)
                
                # Chart
                st.subheader("📈 Daily Schedule Chart")
                st.bar_chart(daily_schedule.set_index('Callback Date')['Total_Shipments'])
            else:
                st.info("No daily schedule data")

if __name__ == "__main__":
    main()