pip install imblearn
mv ./train2.out ./train2.out_$(date +%F-%H:%M)
mv ./RandomizedSearchCV_result_df.csv ./RandomizedSearchCV_result_df.csv_$(date +%F-%H:%M)
export CUDA_VISIBLE_DEVICES=""
nohup python3 ./train2.py > train2.out 2>&1 &
