python llm_train_v2-a.py \
    --model_name google/gemma-2-2b-it \
    --dataset_name mahfuzh74/news_multiclass_no_pad \
    --output_dir ./instruction-tuned-model \
    --batch_size 8 \
    --num_epochs 1 \
    --learning_rate 5e-5 \
    --max_length 512 \
    --save_steps 500 \
    --eval_steps 500
