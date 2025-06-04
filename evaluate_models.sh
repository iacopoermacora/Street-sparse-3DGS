#!/bin/bash

# Street Sparse 3D Gaussian Splatting Training Pipeline
# This script automates the entire training process and logs timing information

# Set the dataset directory
DATASET_DIR="/host"
LOG_FILE="${DATASET_DIR}/training_pipeline_timing.txt"

OUTPUT_FOLDERS=(
    "TESTS_VARIOUS/constraint_init_2000/output_CONSTRAINT_0_01"
    "TESTS_VARIOUS/constraint_init_2000/output_CONSTRAINT_0_025"
    "TESTS_VARIOUS/constraint_init_2000/output_CONSTRAINT_0_05"
    "TESTS_VARIOUS/constraint_init_2000/output_CONSTRAINT_0_1"
    "TESTS_VARIOUS/constraint_init_2000/output_CONSTRAINT_0_3"
    "TESTS_VARIOUS/constraint_init_2000/output_CONSTRAINT_0_5"
)

    # "TESTS_VARIOUS/test_lidar_init/Initialisation_1000/output"
    # "TESTS_VARIOUS/test_lidar_init/Initialisation_2000/output"
    # "TESTS_VARIOUS/test_lidar_init/Initialisation_5000/output"
    # "TESTS_VARIOUS/test_lidar_init/Initialisation_10000/output"
    # "TESTS_VARIOUS/test_lidar_init/Initialisation_20000/output"

CAMERA_CALIBRATION_FOLDERS=(
    "camera_calibration"
    "camera_calibration"
    "camera_calibration"
    "camera_calibration"
    "camera_calibration"
    "camera_calibration"
)

    # "TESTS_VARIOUS/test_lidar_init/Initialisation_1000/camera_calibration"
    # "TESTS_VARIOUS/test_lidar_init/Initialisation_2000/camera_calibration"
    # "TESTS_VARIOUS/test_lidar_init/Initialisation_5000/camera_calibration"
    # "TESTS_VARIOUS/test_lidar_init/Initialisation_10000/camera_calibration"
    # "TESTS_VARIOUS/test_lidar_init/Initialisation_20000/camera_calibration"

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

for i in "${!OUTPUT_FOLDERS[@]}"; do

  OUTPUT_FOLDER="${OUTPUT_FOLDERS[$i]}"
  CAMERA_CALIBRATION_FOLDER="${CAMERA_CALIBRATION_FOLDERS[$i]}"

  RESULTS_FILE="${DATASET_DIR}/${OUTPUT_FOLDER}/evaluation_results.txt"

  # Step 10: Render Evaluation Results
  run_and_log "python render_hierarchy_final.py --segmentation_root_folder ${DATASET_DIR}/${CAMERA_CALIBRATION_FOLDER}/rectified/semantic_segmentation/masks -s ${DATASET_DIR}/${CAMERA_CALIBRATION_FOLDER}/aligned --model_path ${DATASET_DIR}/${OUTPUT_FOLDER} --hierarchy ${DATASET_DIR}/${OUTPUT_FOLDER}/merged.hier --out_dir ${DATASET_DIR}/${OUTPUT_FOLDER}/renders --images ${DATASET_DIR}/${CAMERA_CALIBRATION_FOLDER}/rectified/images --depths ${DATASET_DIR}/${CAMERA_CALIBRATION_FOLDER}/rectified/depths --eval --scaffold_file ${DATASET_DIR}/${OUTPUT_FOLDER}/scaffold/point_cloud/iteration_30000 >> ${RESULTS_FILE} 2>&1" "Step 10: Render Evaluation Results (${OUTPUT_FOLDER})"

done

# Final summary
echo "-----------------------------------" >> $LOG_FILE
echo "Pipeline completed at: $(date)" >> $LOG_FILE
echo "Total steps executed: ${step_count}" >> $LOG_FILE

echo "Training pipeline complete!"
echo "Timing log saved to: $LOG_FILE"