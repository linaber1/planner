#python -m pts.train.generate_latents

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#cd /home/abdulrahman.mahmoud/HEAKL/PTS
#source ~/.bashrc
#conda activate .heakl
#module load nvidia/cuda/11.8

#set -a
#source .venv
#set +a


#source llada_env/bin/activate

export PYTHONPATH="$(pwd):${PYTHONPATH}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATASETS=(arc_easy arc_challenge dart-1 dart-2 dart-3 dart-4 dart-5)
GPUS=(0 1)

pids=()
for i in "${!DATASETS[@]}"; do
	ds="${DATASETS[$i]}"
	gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
	CUDA_VISIBLE_DEVICES="$gpu" python pts/train/generate_latents.py \
		--dataset "$ds" \
		--num_samples 5000 \
		--start_index 0 \
		--output_suffix s5000 &
	pids+=("$!")

	if [[ ${#pids[@]} -ge ${#GPUS[@]} ]]; then
		wait "${pids[0]}"
		pids=("${pids[@]:1}")
	fi
done

wait

#export CUDA_VISIBLE_DEVICES=1
#python pts/train/generate_latents.py --dataset "dart-5" --num_samples 5000

# for dataset in "arc_challenge" "dart-1" "dart-2" "dart-3" "dart-4" "dart-5" "gsm8k":
# do
#export CUDA_VISIBLE_DEVICES=0
#python pts/train/generate_latents.py --dataset "dart-4" --num_samples 5000


#export CUDA_VISIBLE_DEVICES=1
#python pts/train/generate_latents.py --dataset "dart-5" --num_samples 5000

#export CUDA_VISIBLE_DEVICES=1
#python pts/train/generate_latents.py --dataset "dart-2" --num_samples 5000


#export CUDA_VISIBLE_DEVICES=3
#python pts/train/generate_latents.py --dataset "dart-3" --num_samples 5000