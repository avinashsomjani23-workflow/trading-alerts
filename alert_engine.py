# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 & 3: EXECUTION ENGINE (Replaces old main loop)
# ══════════════════════════════════════════════════════════════════════════════
import smc_detector

print(f"Phase 2 Engine started {ist_str()} ({utc_str()} UTC)")

# Load Step 1 Output
active_obs = load_json("active_obs.json", {})
watch_state = load_json("active_watch_state.json", {})

for pair_conf in config["pairs"]:
    symbol = pair_conf["symbol"]
    name = pair_conf["name"]
    entry_model = pair_conf.get("entry_model", "limit")
    
    # Check if Phase 1 mapped any OBs for this pair
    pair_obs = active_obs.get(name, [])
    if not pair_obs: continue

    # Fetch live execution timeframe data
    trigger_interval = pair_conf.get("trigger_tf", "15m")
    df_trigger = clean_df(yf.download(symbol, period="5d", interval=trigger_interval, progress=False))
    df_h1 = clean_df(yf.download(symbol, period="15d", interval="1h", progress=False))
    
    if df_trigger is None or df_h1 is None or df_h1.empty: continue
    
    current_price = float(df_trigger['Close'].iloc[-1])
    h1_atr = get_atr(df_h1)
    if not h1_atr: continue
    
    # Calculate Dynamic ATR Warning Distance
    warning_dist = pair_conf["atr_multiplier"] * h1_atr

    for ob in pair_obs:
        ob_proximal = float(ob['proximal_line'])
        distance = abs(current_price - ob_proximal)
        
        # Phase 2 Radar: Is price approaching?
        if distance <= warning_dist:
            bias = "LONG" if "Demand" in ob['direction'] else "SHORT"
            
            # Step 2: Scorecard
            fvg_data = ob.get("fvg", {"exists": False}) # Assuming Step 1 saves this
            score_res = smc_detector.run_scorecard(bias, df_h1, ob, fvg_data, current_price)
            
            # Hard Gate: Min Confidence 7.0
            if score_res['total'] < pair_conf["min_confidence"]:
                print(f"  [X] {name} Score {score_res['total']} < 7.0. Aborted.")
                continue
                
            # Step 3: Risk Math & Entry Cascade
            levels = smc_detector.compute_dynamic_levels(pair_conf, bias, ob, fvg_data, current_price, df_trigger)
            
            if not levels['valid']:
                print(f"  [X] {name} {levels['reason']}. Aborted.")
                continue

            trade_data = {
                "pair": name,
                "bias": bias,
                "score": score_res['total'],
                "levels": levels,
                "ob": ob
            }

            # Routing
            if entry_model == "limit":
                # FOREX: Fire Email Alert (Placeholder for Step 4 formatting)
                print(f"  [✓] TRADE READY (FOREX): {name} | Entry: {levels['entry']} | R:R: {levels['rr']}")
                # send_email(...) 
            
            elif entry_model == "ltf_choch":
                # GOLD/NAS: Push to Phase 3 Watch State
                watch_state[f"{name}_{ob_proximal}"] = trade_data
                print(f"  [>] LOGGED FOR PHASE 3 (CHoCH): {name} approaching {ob_proximal}")

# Save state for Phase 3
save_json("active_watch_state.json", watch_state)
