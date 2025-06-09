#!/bin/bash

# Street Sparse 3D Gaussian Splatting Training Pipeline
# This script automates the entire training process and logs timing information

# Set the dataset directory
DATASET_DIR="/host"
LOG_FILE="${DATASET_DIR}/training_pipeline_timing.txt"
RESULTS_FILE="${DATASET_DIR}/output/results.txt"

PARAMS=( 0.9 0.3 )

# Initialize the log file
echo "Street Sparse 3D Gaussian Splatting Training Pipeline Timing Log" > $LOG_FILE
echo "Started at: $(date)" >> $LOG_FILE
echo "-----------------------------------" >> $LOG_FILE

# Function to run a command and log its execution time
run_and_log() {
    local cmd="$1"
    local description="$2"
    
    echo "Running: $description..."
    echo "Command: $cmd"
    
    # Log the start time
    echo "-----------------------------------" >> $LOG_FILE
    echo "$description" >> $LOG_FILE
    echo "Start time: $(date)" >> $LOG_FILE
    
    # Measure the execution time
    start_time=$(date +%s)
    
    # Execute the command
    eval $cmd
    exit_code=$?
    
    # Calculate execution time
    end_time=$(date +%s)
    execution_time=$((end_time - start_time))
    hours=$((execution_time / 3600))
    minutes=$(( (execution_time % 3600) / 60 ))
    seconds=$((execution_time % 60))
    
    # Log the results
    echo "End time: $(date)" >> $LOG_FILE
    echo "Execution time: ${hours}h ${minutes}m ${seconds}s" >> $LOG_FILE
    
    if [ $exit_code -eq 0 ]; then
        echo "Status: Completed successfully" >> $LOG_FILE
    else
        echo "Status: Failed with exit code $exit_code" >> $LOG_FILE
    fi
    
    # Print status to console
    if [ $exit_code -eq 0 ]; then
        echo "Completed in ${hours}h ${minutes}m ${seconds}s"
    else
        echo "Failed with exit code $exit_code after ${hours}h ${minutes}m ${seconds}s"
    fi
    
    echo ""
}

for PARAM in "${PARAMS[@]}"; do
  echo "Running with additional depth weight: ${PARAM}"

  mkdir -p "${DATASET_DIR}/output"

  # Step 9: Train and Evaluate the Model with current WEIGHT
  run_and_log "python scripts/full_train.py --project_dir ${DATASET_DIR} --gt_point_cloud_constraints --constraint_treshold 0.15 --additional_depth_maps --additional_depth_maps_weight ${PARAM} --extra_training_args '--exposure_lr_init 0.0 --eval' > ${RESULTS_FILE}" "Step 9: Train and Evaluate the Model w/o additional and constraint (weight ${PARAM})"

  # Step 10: Render Evaluation Results
  run_and_log "python render_hierarchy_final.py --segmentation_root_folder ${DATASET_DIR}/camera_calibration/rectified/semantic_segmentation/masks -s ${DATASET_DIR}/camera_calibration/aligned --model_path ${DATASET_DIR}/output --hierarchy ${DATASET_DIR}/output/merged.hier --out_dir ${DATASET_DIR}/output/renders --images ${DATASET_DIR}/camera_calibration/rectified/images --depths ${DATASET_DIR}/camera_calibration/rectified/depths --eval --scaffold_file ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000 >> ${RESULTS_FILE} 2>&1" "Step 10: Render Evaluation Results (weight ${PARAM})"

  # Rename the output folder
  NEW_OUTPUT_DIR="${DATASET_DIR}/output_ADD_DEPTH_WEIGHT_${PARAM//./_}" # Replaces '.' with '_' for folder name
  echo "Renaming ${DATASET_DIR}/output to ${NEW_OUTPUT_DIR}"
  mv "${DATASET_DIR}/output" "${NEW_OUTPUT_DIR}"

done

# Final summary
echo "-----------------------------------" >> $LOG_FILE
echo "Pipeline completed at: $(date)" >> $LOG_FILE
echo "Total steps executed: ${step_count}" >> $LOG_FILE

echo "Training pipeline complete!"
echo "Timing log saved to: $LOG_FILE"