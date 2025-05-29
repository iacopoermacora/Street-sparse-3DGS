#!/bin/bash

# Street Sparse 3D Gaussian Splatting Training Pipeline
# This script automates the entire training process and logs timing information

# Set the dataset directory
DATASET_DIR="/host"
LOG_FILE="${DATASET_DIR}/training_pipeline_timing.txt"
RESULTS_FILE="${DATASET_DIR}/output/results.txt"

CONSTRAINT_THRESHOLDS=(0.05 0.1 0.3)

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

# # Step 1: Generate Colmap Calibration
# run_and_log "python ss_utils/generate_colmap_calibration.py --project_dir ${DATASET_DIR} --eval" "Step 1: Generate Colmap Calibration"

# # Step 2: Prepare Inputs
# run_and_log "python ss_utils/create_inputs.py --project_dir ${DATASET_DIR}" "Step 2: Prepare Inputs"

# # Step 3: Mask Images
# echo "Step 3: Mask Images requires manual intervention in X2GO environment"
# echo "Please run the following command manually in X2GO:"
# echo "python ss_utils/mask_images.py --project_dir ${DATASET_DIR}"
# echo "Once complete, press Enter to continue..."
# read -p "Press Enter to continue after masking images..."

# # Step 4: Generate Colmap
# run_and_log "python preprocess/generate_colmap.py --project_dir ${DATASET_DIR}" "Step 4: Generate Colmap"

# # Step 5: Fix Colmap to work with only 6 images
# run_and_log "python ss_utils/colmap_fix.py --project_dir ${DATASET_DIR}" "Step 5: Fix Colmap for 6 images"

# # Step 6: Generate Depth Maps
# run_and_log "python ss_utils/ss_generate_depths.py --project_dir ${DATASET_DIR}" "Step 6: Generate Depth Maps"

# # Step 7: Generate Chunks
# run_and_log "python preprocess/generate_chunks.py --project_dir ${DATASET_DIR} --LiDAR_initialisation --LiDAR_downsample_density 5000" "Step 7: Generate Chunks"

# # Step 8: Copy test.txt
# run_and_log "python ss_utils/copy_test_and_depth_params_files.py --project_dir ${DATASET_DIR}" "Step 8: Copy test.txt"

for THRESHOLD in "${CONSTRAINT_THRESHOLDS[@]}"; do
  echo "Running with constraint_threshold: ${THRESHOLD}"

  # Step 9: Train and Evaluate the Model with current constraint_treshold
  run_and_log "python scripts/full_train.py --project_dir ${DATASET_DIR} --extra_training_args --gt_point_cloud_constraints --constraint_treshold ${THRESHOLD} '--exposure_lr_init 0.0 --eval' > ${RESULTS_FILE}" "Step 9: Train and Evaluate the Model w/o additional and constraint (threshold ${THRESHOLD})"

  # Step 10: Render Evaluation Results
  run_and_log "python render_hierarchy.py -s ${DATASET_DIR}/camera_calibration/aligned --model_path ${DATASET_DIR}/output --hierarchy ${DATASET_DIR}/output/merged.hier --out_dir ${DATASET_DIR}/output/renders --images ${DATASET_DIR}/camera_calibration/rectified/images --eval --scaffold_file ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000 >> ${RESULTS_FILE} 2>&1" "Step 10: Render Evaluation Results (threshold ${THRESHOLD})"

  # Rename the output folder
  NEW_OUTPUT_DIR="${DATASET_DIR}/output_CONSTRAINT_${THRESHOLD//./_}" # Replaces '.' with '_' for folder name
  echo "Renaming ${DATASET_DIR}/output to ${NEW_OUTPUT_DIR}"
  mv "${DATASET_DIR}/output" "${NEW_OUTPUT_DIR}"

done

# Final summary
echo "-----------------------------------" >> $LOG_FILE
echo "Pipeline completed at: $(date)" >> $LOG_FILE
echo "Total steps executed: ${step_count}" >> $LOG_FILE

echo "Training pipeline complete!"
echo "Timing log saved to: $LOG_FILE"