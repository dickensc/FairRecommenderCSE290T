#!/usr/bin/env bash

# run weight learning performance experiments,
#i.e. collects runtime and evaluation statistics of various weight learning methods

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_DIR="${THIS_DIR}/.."
readonly BASE_OUT_DIR="${BASE_DIR}/results/fairness"

readonly STUDY_NAME='noisy_attribute_fairness_threshold_study'
readonly SUPPORTED_DATASETS='movielens movielens_corrupt'
readonly SUPPORTED_FAIRNESS_MODELS='base rank non_parity value non_parity_value nmf nb non_parity_nmf_retro_fit value_nmf_retro_fit mutual_information'

readonly DATASETS=('movielens' 'movielens_corrupt')

#readonly FAIRNESS_MODELS='base rank non_parity value non_parity_value nb nmf non_parity_nmf_retro_fit value_nmf_retro_fit mutual_information'
readonly FAIRNESS_MODELS='non_parity'
declare -A FAIRNESS_THRESHOLDS
FAIRNESS_THRESHOLDS['non_parity']='0.002 0.004 0.006 0.008 0.010 0.012 0.014 0.016 0.018 0.020 0.022 0.024 0.026 0.028 0.030'

readonly WL_METHODS='UNIFORM'
readonly SEED=22
readonly TRACE_LEVEL='TRACE'

# Evaluators to be use for each example
declare -A DATASET_EVALUATORS
DATASET_EVALUATORS[movielens]='Continuous AUC'

readonly STANDARD_OPTIONS='-D reasoner.tolerance=1.0e-5f -D inference.relax.squared=false -D inference.normalize=false -D weightlearning.inference=SGDInference -D sgd.extension=ADAM -D sgd.inversescaleexp=2.0'

# Evaluators to be use for each example
declare -A DATASET_FOLDS
DATASET_FOLDS[movielens]=5

function run_example() {
    local example_name=$1
    local wl_method=$2
    local fairness_model=$3
    local fold=$4
    local fair_threshold=$5
    local evaluator=$6

    echo "Running example ${example_name} : ${fairness_model} : ${fold} : ${wl_method} : tau=${fair_threshold}"

    local cli_directory="${BASE_DIR}/psl-datasets/${example_name}/cli"

    out_directory="${BASE_OUT_DIR}/psl/${STUDY_NAME}/${example_name}/${wl_method}/${evaluator}/${fairness_model}/${fair_threshold}"/${fold}

    # Only make a new out directory if it does not already exist
    [[ -d "$out_directory" ]] || mkdir -p "$out_directory"

    # Setup experiment cli and data directory
    setup_fairness_experiment "${example_name}" "${fairness_model}" "${cli_directory}"

    # Write the fairness weight
    write_fairness_threshold "$fair_threshold" "$fairness_model" "$example_name" "$wl_method" "$cli_directory"

    ##### WEIGHT LEARNING #####
    run_weight_learning "${example_name}" "${evaluator}" "${wl_method}" "${fairness_model}" "${fair_threshold}" "${fold}" "${cli_directory}" "${out_directory}"

    ##### EVALUATION #####
    run_evaluation "${example_name}" "${evaluator}" "${fairness_model}" "${fair_threshold}" "${fold}" "${out_directory}"

    return 0
}

function setup_fairness_experiment() {
      local example_name=$1
      local fairness_model=$2
      local cli_directory=$3

      local fairness_model_directory="${BASE_DIR}/psl-datasets/${example_name}/${example_name}_${fairness_model}"

      # copy the .data and .psl files into the cli directory
      cp "${fairness_model_directory}/${example_name}.psl" "${cli_directory}/${example_name}.psl"
      cp "${fairness_model_directory}/${example_name}-eval.data" "${cli_directory}/${example_name}-eval.data"
      cp "${fairness_model_directory}/${example_name}-learn.data" "${cli_directory}/${example_name}-learn.data"
}

function run_evaluation() {
    local example_name=$1
    local evaluator=$2
    local fairness_model=$3
    local fair_threshold=$4
    local fold=$5
    local out_directory=$6

    # path to output files
    local out_path="${out_directory}/eval_out.txt"
    local err_path="${out_directory}/eval_out.err"

    if [[ -e "${out_path}" ]]; then
        echo "Output file already exists, skipping: ${out_path}"
    else
        echo "Running ${wl_method} Evaluation for ${example_name} ${evaluator} ${fairness_model} ${fair_threshold} (#${fold})."
        # call inference script for SRL model type
        pushd . > /dev/null
            cd "psl_scripts" || exit
            ./run_inference.sh "${example_name}" "${evaluator}" "${fairness_model}" "${fold}" "${out_directory}"> "$out_path" 2> "$err_path"
        popd > /dev/null
    fi
}

function run_weight_learning() {
    local example_name=$1
    local evaluator=$2
    local wl_method=$3
    local fairness_model=$4
    local fair_threshold=$5
    local fold=$6
    local cli_directory=$7
    local out_directory=$8

    # path to output files
    local out_path="${out_directory}/learn_out.txt"
    local err_path="${out_directory}/learn_out.err"

    if [[ -e "${out_path}" ]]; then
        echo "Output file already exists, skipping: ${out_path}"
        echo "Copying cached learned model from earlier run into cli"
        # copy the learned weights into the cli directory for inference
        cp "${out_directory}/${example_name}-learned.psl" "${cli_directory}/"
    else
        echo "Running ${wl_method} Weight Learning for ${example_name} ${evaluator} ${fairness_model} ${fair_threshold} (#${fold})."
        # call weight learning script for SRL model type
        pushd . > /dev/null
            cd "psl_scripts" || exit
            ./run_wl.sh "${example_name}" "${evaluator}" "${wl_method}" "${fairness_model}" "${fold}" "${SEED}" "${out_directory}" "${TRACE_LEVEL}" > "$out_path" 2> "$err_path"
        popd > /dev/null
    fi
}

function write_fairness_threshold() {
    local fairness_threshold=$1
    local fairness_model=$2
    local example_name=$3
    local wl_method=$4
    local cli_directory=$5

    # write fairness threshold for constrarint in psl file
    pushd . > /dev/null
        cd "${cli_directory}" || exit

        local rule

        if [[ ${fairness_model} != 'base' ]]; then
          if [[ ${wl_method} == 'UNIFORM' ]]; then
            # set the fairness related constraint thresholds in the learned file to the fairness_threshold value and write to learned.psl file
            if [[ ${fairness_model} == 'non_parity' || ${fairness_model} == 'non_parity_nmf_retro_fit' ]]; then
              rule="group1_avg_rating\(c\) - group2_avg_rating\(c\)"
            elif [[ ${fairness_model} == 'value' || ${fairness_model} == 'value_nmf_retro_fit' ]]; then
              rule="pred_group_average_item_rating\(G1, I\) - obs_group_average_item_rating\(G1, I\) = pred_group_average_item_rating\(G2, I\) - obs_group_average_item_rating\(G2, I\)"
            elif [[ ${fairness_model} == 'non_parity_value' ]]; then
              rule="group1_avg_rating\(c\) = group2_avg_rating\(c\)"
              rule="pred_group_average_item_rating\(G1, I\) - obs_group_average_item_rating\(G1, I\) = pred_group_average_item_rating\(G2, I\) - obs_group_average_item_rating\(G2, I\)"
            elif [[ ${fairness_model} == 'mutual_information' ]]; then
              rule="@MI\[rating\(\+U1, I\), group_member\(\+U2, \+G\)\] \{U1: rated\(U1, I\)\}"
            fi
            sed -i -r "s/^${rule} <= TAU .|${rule} <= [0-9]+.[0-9]+ ./${rule} <= ${fairness_threshold} ./g"  "${example_name}.psl"
            sed -i -r "s/^${rule} >= TAU .|${rule} >= [0-9]+.[0-9]+ ./${rule} >= ${fairness_threshold} ./g"  "${example_name}.psl"
          else
            if [[ ${fairness_model} == 'non_parity' || ${fairness_model} == 'non_parity_nmf_retro_fit' ]]; then
              rule="1.0 \* GROUP1_AVG_RATING\(c\) \+ -1.0 \* GROUP2_AVG_RATING\(c\) = 0.0"
            elif [[ ${fairness_model} == 'value' || ${fairness_model} == 'value_nmf_retro_fit' ]]; then
              rule="1.0 \* PRED_GROUP_AVERAGE_ITEM_RATING\(G1, I\) \+ -1.0 \* OBS_GROUP_AVERAGE_ITEM_RATING\(G1, I\) \+ -1.0 \* PRED_GROUP_AVERAGE_ITEM_RATING\(G2, I\) \+ 1.0 \* OBS_GROUP_AVERAGE_ITEM_RATING\(G2, I\) = 0.0"
            elif [[ ${fairness_model} == 'non_parity_value' ]]; then
              rule="group1_avg_rating\(c\) = group2_avg_rating\(c\)"
              rule="pred_group_average_item_rating\(G1, I\) - obs_group_average_item_rating\(G1, I\) = pred_group_average_item_rating\(G2, I\) - obs_group_average_item_rating\(G2, I\)"
            elif [[ ${fairness_model} == 'mutual_information' ]]; then
              rule="@MI\[rating\(\+U1, I\), group_member\(\+U2, \+G\)\] \{U1: rated\(U1, I\)\}"
            fi
            sed -i -r "s/^${rule} <= TAU .|${rule} <= [0-9]+.[0-9]+ ./${rule} <= ${fairness_threshold} ./g"  "${example_name}.psl"
            sed -i -r "s/^${rule} >= TAU .|${rule} >= [0-9]+.[0-9]+ ./${rule} >= ${fairness_threshold} ./g"  "${example_name}.psl"
          fi
        fi

    popd > /dev/null
}

function main() {
    trap exit SIGINT

    # if [[ $# -eq 0 ]]; then
    #     echo "USAGE: $0 <example dir> ..."
    #     exit 1
    # fi

    for example_name in ${DATASETS[@]}; do
      echo "${example_name}";
      for wl_method in ${WL_METHODS}; do
        for fairness_model in ${FAIRNESS_MODELS}; do
          for fair_threshold in ${FAIRNESS_THRESHOLDS[${fairness_model}]}; do
            for evaluator in ${DATASET_EVALUATORS[${example_name}]}; do
              for ((fold=0; fold<${DATASET_FOLDS[${example_name}]}; fold++)) do
                if [[ "${SUPPORTED_DATASETS}" == *"${example_name}"* ]]; then
                  if [[ "${SUPPORTED_FAIRNESS_MODELS}" == *"${fairness_model}"* ]]; then
                    run_example "${example_name}" "${wl_method}" "${fairness_model}" "${fold}" "${fair_threshold}" "${evaluator}"
                  fi
                 fi
              done
            done
          done
        done
      done
    done

    return 0
}

main #"$@"
