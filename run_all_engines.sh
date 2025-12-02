#!/usr/bin/env fish

# Shell script to run rl.py with all reward engines

set -l engines \
    "infogain_only" \
    "info_gain_max_difference_only" \
    "info_gain_max_range_only" \
    "cos_sim_only" \
    "entropy_only" \
    "max_entropy_only" \
    "text_align_only" \
    "text_align_fixed_embedding_only" \
    "infogain_only_depth" \
    "info_gain_max_difference_only_depth" \
    "info_gain_max_range_only_depth" \
    "cos_sim_only_depth" \
    "entropy_only_depth" \
    "max_entropy_only_depth" \
    "text_align_only_depth" \
    "text_align_fixed_embedding_only_depth" \
    "vision_contrast" \
    "vision_structural" \
    "multimodal_text_driven" \
    "multimodal_entropy" \
    "multimodal_with_vision" \
    "balanced_standard" \
    "balanced_comprehensive" \
    "balanced_diversity" \
    "depth_limited" \
    "fast_exploration" \
    "quality_assurance" \
    "ablation_infogain_only" \
    "ablation_max_difference_only" \
    "ablation_cosine_range_only" \
    "ablation_entropy_only" \
    "ablation_max_entropy_only" \
    "ablation_text_align_only" \
    "ablation_no_vision" \
    "ablation_no_multimodal"

for engine in $engines
    echo "========================================="
    echo "Running with reward engine: $engine"
    echo "========================================="
    # Ensure benchmark directory and CSV exist; add header once
    set -l csv "data/benchmark/training_time.csv"
    mkdir -p (dirname $csv)
    if not test -f $csv
        echo "engine,start_time,end_time,duration_seconds,status" > $csv
    end

    # Capture start time (epoch seconds)
    set -l start (date +%s)

    # Run training while showing output; capture status
    python src/rl.py --reward-engine $engine
    set -l status_code $status

    # Capture end time and compute duration
    set -l end (date +%s)
    set -l duration (math $end - $start)

    # Append CSV row
    printf "%s,%s,%s,%s,%s\n" "$engine" "$start" "$end" "$duration" "$status_code" >> $csv

    if test $status -ne 0
        echo "ERROR: Training failed for engine: $engine"
        echo "Continuing to next engine..."
    else
        echo "SUCCESS: Training completed for engine: $engine"
    end
    
    echo ""
end

echo "All engines processed!"
